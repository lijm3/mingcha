"""把内核落盘的 answer.json 读出来，包装成带产物 URL 的 AnswerOut（§5.3）。
*_url 由 task_token 在此拼装，不落进 answer.json。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .schemas import AnswerOut, EvidenceOut, SummaryDetail


def build_answer_out(out_dir: Path, task_id: str, token: str) -> AnswerOut | None:
    ans_path = out_dir / "answer.json"
    if not ans_path.exists():
        return None
    data = json.loads(ans_path.read_text(encoding="utf-8"))

    def art_url(rel: str) -> str:
        return f"/artifacts/{task_id}/{rel}?token={token}"

    evidence = []
    for e in data.get("evidence", []):
        frame = e.get("frame", "")
        evidence.append(EvidenceOut(
            frame=frame,
            frame_url=art_url(f"frames/{frame}") if frame else "",
            t=e.get("t", 0.0), hms=e.get("hms", ""),
            confidence=e.get("confidence", 0.0),
            similarity=e.get("similarity"), verdict=e.get("verdict"),
            note=e.get("note", ""),
        ))

    # SUMMARY 三段式（内核已透传 topic/segments/key_points）
    detail = None
    if data.get("topic") or data.get("segments") or data.get("key_points"):
        detail = SummaryDetail(
            topic=data.get("topic", ""),
            segments=data.get("segments", []),
            key_points=data.get("key_points", []),
        )

    # 视频 URL：产物目录里找视频文件
    video_url = None
    for name in os.listdir(out_dir):
        low = name.lower()
        if low.endswith((".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts")):
            video_url = art_url(name)
            break

    # 参考图回显（VISUAL_LOCATE）
    query_image_url = None
    if (out_dir / "query_image.jpg").exists():
        query_image_url = art_url("query_image.jpg")

    # 拼图
    grids = []
    grids_dir = out_dir / "grids"
    if grids_dir.exists():
        for g in sorted(os.listdir(grids_dir)):
            if g.lower().endswith(".jpg"):
                grids.append(art_url(f"grids/{g}"))

    return AnswerOut(
        intent=data.get("intent", ""), target=data.get("target"),
        answer=data.get("answer", ""), summary_detail=detail,
        evidence=evidence, confidence=data.get("confidence", 0.0),
        caveats=data.get("caveats", ""), video_url=video_url,
        query_image_url=query_image_url, grids=grids,
    )
