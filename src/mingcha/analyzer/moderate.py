"""MODERATE 分析 —— FR-5.3 / §6.3 高召回 + §8 合规。

关去重的分段密采后帧很多，逐帧判定「是否存在」（judge_frames 并发，只判布尔+置信度，
不细节化露骨内容 §8），按较低阈值高召回，再把相邻命中点合并为时间区间。命中涉未成年人
或模型拒答 → 标注建议高优先级人工复核（C-2 / §6.7）。
"""
from __future__ import annotations

import os

from .. import assembler, llm, timestamps
from ..config import MODERATE_MERGE_GAP, MODERATE_THRESH
from ..prompts import MODERATE_INSTRUCTION, moderate_system
from ..types import Answer, Evidence, ModerateSchema, Plan


def _merge(times: list[float], gap: float) -> list[tuple[float, float]]:
    """把命中时间点按最大间隔 gap 合并为 [(start, end), ...]。"""
    if not times:
        return []
    times = sorted(times)
    ranges = [[times[0], times[0]]]
    for t in times[1:]:
        if t - ranges[-1][1] <= gap:
            ranges[-1][1] = t
        else:
            ranges.append([t, t])
    return [(a, b) for a, b in ranges]


def analyze(out_dir: str, plan: Plan, target: str, *, vision_model=None) -> Answer:
    """分段高召回逐帧审核，合并命中片段为时间区间。见设计文档 §7.3 + §8。"""
    stamps = timestamps.load(out_dir)
    frames_dir = os.path.join(out_dir, "frames")
    items = [(os.path.join(frames_dir, s.frame), s.t) for s in stamps
             if os.path.exists(os.path.join(frames_dir, s.frame))]
    if not items:
        return assembler.from_moderate(target, [], [], out_dir,
                                       "无可用关键帧（预处理未产出帧）。")

    verdicts = llm.judge_frames("vision", moderate_system(target), items, ModerateSchema,
                                override=vision_model, instruction=MODERATE_INSTRUCTION)

    hit_times: list[float] = []
    evidences: list[Evidence] = []
    need_review = False
    for path, t, o in verdicts:
        note = o.note or ""
        if "人工复核" in note or "未成年" in note:
            need_review = True
        if o.present and o.confidence >= MODERATE_THRESH:
            hit_times.append(t)
            evidences.append(Evidence(frame=os.path.basename(path), t=round(t, 3),
                                      hms=timestamps.hms(t), confidence=round(o.confidence, 3),
                                      note=note))

    ranges = _merge(hit_times, MODERATE_MERGE_GAP)
    review_note = "⚠ 存在需高优先级人工复核的帧（疑似敏感/模型拒答）。" if need_review else ""
    caveats = ("高召回自动初筛（阈值偏低以少漏），可能有误报；"
               "涉未成年人或敏感内容以人工复核为准。")
    return assembler.from_moderate(target, ranges, evidences, out_dir, caveats, review_note)
