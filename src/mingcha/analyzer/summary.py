"""SUMMARY 分析 —— FR-5.1。把关键帧拼图 + 转写喂给 vision 档模型，产出结构化摘要。
等价于当前“人读 MANIFEST”的自动化。对应设计文档 §7.1。
"""
from __future__ import annotations

import glob
import os

from .. import llm
from ..prompts import SUMMARY_SYSTEM, summary_user
from ..types import Answer, Plan, SummarySchema


def analyze(out_dir: str, plan: Plan, prompt: str, *, vision_model=None) -> Answer:
    grids = sorted(glob.glob(os.path.join(out_dir, "grids", "grid_*.jpg")))
    frames = sorted(glob.glob(os.path.join(out_dir, "frames", "frame_*.jpg")))
    images = grids or frames  # 无拼图则退化为直接喂关键帧

    transcript = ""
    tpath = os.path.join(out_dir, "transcript.txt")
    if os.path.exists(tpath):
        transcript = open(tpath, encoding="utf-8", errors="ignore").read()

    result: SummarySchema = llm.vision_structured(
        role="vision", system=SUMMARY_SYSTEM, images=images,
        instruction=summary_user(transcript, why=prompt),
        schema=SummarySchema, override=vision_model)

    body = (result.summary or result.topic or "").strip()
    if result.segments:
        body += "\n\n分段脉络：\n" + "\n".join(f"- {s}" for s in result.segments)
    if result.key_points:
        body += "\n\n关键点：\n" + "\n".join(f"- {k}" for k in result.key_points)
    return Answer(intent="SUMMARY", answer=body.strip(), confidence=0.8,
                  artifacts_dir=out_dir)
