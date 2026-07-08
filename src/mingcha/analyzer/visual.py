"""VISUAL_LOCATE 分析 —— FR-5.4 / §6.6 三级由粗到细。

级1 像素预筛（本地零依赖）：similarity.rank 取 Top-K 候选，把逐帧多模态调用降到 K 次。
级2 语义确认（多模态）：参考图 + 候选帧成对给 vision 模型，判 same/similar/no（参考图设
cacheable，支持 caching 的 provider 自动省钱 §6.5）。级3 精扫：对最早命中的候选做区间密采
精确到秒。区分「同一个体」与「同类外观」，相似≠同一时如实降级措辞与 confidence（R-5）。
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from .. import assembler, llm, rescan, similarity, timestamps
from ..config import (
    JUDGE_MAX_WORKERS, RESCAN_STEP, RESCAN_WINDOW, RESCAN_SCALE_WIDTH,
    TOPK, VISUAL_THRESH,
)
from ..prompts import VISUAL_INSTRUCTION, visual_system
from ..types import Answer, Plan, VisualHitSchema

# same 比 similar 更可信：命中优先级排序用
_RANK = {"same": 2, "similar": 1, "no": 0}


def _pair_judge(query_image: str, frame_path: str, desc: str, vision_model) -> VisualHitSchema:
    """参考图 + 候选帧成对语义确认；异常（含拒答）降级为 no，不中断整体。"""
    try:
        return llm.vision_structured(
            "vision", visual_system(desc),
            images=[query_image, frame_path], instruction=VISUAL_INSTRUCTION,
            schema=VisualHitSchema, override=vision_model, cacheable_images=[query_image])
    except Exception:  # noqa: BLE001
        return VisualHitSchema(verdict="no", note="语义确认失败")


def _judge_many(query_image, items, desc, vision_model):
    """items: [(frame_path, t), ...] → [(frame_path, t, VisualHitSchema), ...]，并发、保序。"""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(JUDGE_MAX_WORKERS, len(items))) as ex:
        objs = list(ex.map(
            lambda it: _pair_judge(query_image, it[0], desc, vision_model), items))
    return [(p, t, o) for (p, t), o in zip(items, objs)]


def _caveats(step: float) -> str:
    return (f"以图搜视频基于像素预筛 + 语义确认（精扫每 {step:.1f} 秒一帧）；"
            f"预筛未进 Top-{TOPK} 的帧不会被确认，更短暂的出现可能漏检。")


def analyze(out_dir: str, plan: Plan, query_image: str, source_video: str, *,
            vision_model=None) -> Answer:
    """级1 预筛 → 级2 语义确认 → 级3 精扫。见设计文档 §7.4。"""
    stamps = timestamps.load(out_dir)
    frames_dir = os.path.join(out_dir, "frames")
    path_t = {os.path.join(frames_dir, s.frame): s.t for s in stamps
              if os.path.exists(os.path.join(frames_dir, s.frame))}
    if not path_t:
        return assembler.not_found(out_dir, None, "无可用关键帧（预处理未产出帧）。",
                                   intent="VISUAL_LOCATE")

    # 级1 像素预筛取 Top-K
    scored = similarity.rank(query_image, list(path_t))
    topk = scored[:TOPK]

    # 参考图文字描述（FR-2.3，辅助交叉验证；失败为空不影响）
    desc = llm.describe("vision", query_image, override=vision_model)

    # 级2 语义确认
    cand = [(p, path_t[p]) for p, _ in topk]
    results = _judge_many(query_image, cand, desc, vision_model)
    confirmed = [(p, t, o) for (p, t, o) in results
                 if o.verdict in ("same", "similar") and o.confidence >= VISUAL_THRESH]
    if not confirmed:
        return assembler.not_found(out_dir, desc or None,
                                   _caveats(RESCAN_STEP), intent="VISUAL_LOCATE")

    # 选最早命中：先按 verdict 档次（same>similar），同档取最早时间
    confirmed.sort(key=lambda r: (-_RANK[r[2].verdict], r[1]))
    b_path, b_t, b_obj = confirmed[0]

    # 级3 精扫：命中帧 ±WINDOW 秒密采，成对判定求首次出现（不低于当前档次）
    fine = rescan.dense_extract(source_video, out_dir,
                                t0=b_t - RESCAN_WINDOW, t1=b_t + RESCAN_WINDOW,
                                step=RESCAN_STEP, scale_width=RESCAN_SCALE_WIDTH)
    if fine:
        fine_res = _judge_many(query_image, fine, desc, vision_model)
        floor = _RANK[b_obj.verdict]
        fine_hits = [(p, t, o) for (p, t, o) in fine_res
                     if _RANK[o.verdict] >= floor and o.confidence >= VISUAL_THRESH]
        if fine_hits:
            fine_hits.sort(key=lambda r: r[1])
            b_path, b_t, b_obj = fine_hits[0]

    return assembler.from_visual(
        query_image, desc, b_obj.verdict, b_path, b_t,
        b_obj.similarity, b_obj.confidence, b_obj.note, out_dir, _caveats(RESCAN_STEP))
