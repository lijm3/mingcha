"""LOCATE 分析 —— FR-5.2 / §6.2 两阶段（粗扫定区间 → 精扫到秒）。

粗扫：按时间序对每个关键帧逐帧判定目标是否出现（judge_frames 并发），取**最早**命中帧
定位大致落点。精扫：仅在该帧 ±WINDOW 秒区间用 rescan.dense_extract 密采重抽，逐帧判定
首次出现，把时间精确到秒级。找不到 → 诚实否定（FR-6.2）+ 采样局限声明（NFR-1）。
"""
from __future__ import annotations

import os

from .. import assembler, llm, rescan, timestamps
from ..config import (
    LOCATE_THRESH, RESCAN_STEP, RESCAN_WINDOW, RESCAN_SCALE_WIDTH,
)
from ..prompts import LOCATE_INSTRUCTION, locate_system
from ..types import Answer, HitSchema, Plan


def _caveats(step: float) -> str:
    return (f"定位基于关键帧采样（精扫每 {step:.1f} 秒一帧）；"
            f"更短暂的出现可能未被抽到，时间点为最接近的采样帧。")


def analyze(out_dir: str, plan: Plan, target: str, source_video: str, *,
            vision_model=None) -> Answer:
    """按时间序扫描带时间戳的关键帧，找目标首次出现，两阶段精确到秒。见设计文档 §7.2。"""
    stamps = timestamps.load(out_dir)
    frames_dir = os.path.join(out_dir, "frames")
    system = locate_system(target)

    # 阶段1 粗扫：逐帧判定，取最早命中
    coarse_items = [(os.path.join(frames_dir, s.frame), s.t) for s in stamps
                    if os.path.exists(os.path.join(frames_dir, s.frame))]
    if not coarse_items:
        return assembler.not_found(out_dir, target, "无可用关键帧（预处理未产出帧）。")

    coarse = llm.judge_frames("vision", system, coarse_items, HitSchema,
                              override=vision_model, instruction=LOCATE_INSTRUCTION)
    hits = [(p, t, o) for (p, t, o) in coarse if o.present and o.confidence >= LOCATE_THRESH]
    if not hits:
        return assembler.not_found(out_dir, target, _caveats(plan.fps_floor or 1.0))

    hits.sort(key=lambda x: x[1])
    c_path, c_t, c_obj = hits[0]

    # 阶段2 精扫：命中帧 ±WINDOW 秒密采，求首次出现
    fine = rescan.dense_extract(source_video, out_dir,
                                t0=c_t - RESCAN_WINDOW, t1=c_t + RESCAN_WINDOW,
                                step=RESCAN_STEP, scale_width=RESCAN_SCALE_WIDTH)
    if fine:
        fine_verdicts = llm.judge_frames("vision", system, fine, HitSchema,
                                         override=vision_model, instruction=LOCATE_INSTRUCTION)
        fine_hits = [(p, t, o) for (p, t, o) in fine_verdicts
                     if o.present and o.confidence >= LOCATE_THRESH]
        if fine_hits:
            fine_hits.sort(key=lambda x: x[1])
            f_path, f_t, f_obj = fine_hits[0]
            return assembler.from_locate(target, f_path, f_t, f_obj.confidence,
                                         f_obj.note, out_dir, _caveats(RESCAN_STEP))

    # 精扫无果（区间抽帧失败或未复现）→ 回退到粗扫命中帧
    return assembler.from_locate(target, c_path, c_t, c_obj.confidence,
                                 c_obj.note, out_dir, _caveats(plan.fps_floor or 1.0))
