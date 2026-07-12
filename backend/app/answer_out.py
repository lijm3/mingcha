"""把内核落盘的 answer.json 读出来，包装成带产物 URL 的 AnswerOut（§5.3）。
*_url 由 task_token 在此拼装，不落进 answer.json。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from mingcha import timestamps as _ts

from .schemas import AnswerOut, BBoxOut, EvidenceOut, PlateTrackOut, SummaryDetail


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
        bb = e.get("bbox")
        evidence.append(EvidenceOut(
            frame=frame,
            frame_url=art_url(f"frames/{frame}") if frame else "",
            t=e.get("t", 0.0), hms=e.get("hms", ""),
            confidence=e.get("confidence", 0.0),
            similarity=e.get("similarity"), verdict=e.get("verdict"),
            note=e.get("note", ""),
            bbox=BBoxOut(**bb) if bb else None,        # PLATE 空间证据
            track_id=e.get("track_id"), plate_text=e.get("plate_text"),
            plate_color=e.get("plate_color"),
        ))

    # SUMMARY 三段式（内核已透传 topic/segments/key_points）
    detail = None
    if data.get("topic") or data.get("segments") or data.get("key_points"):
        detail = SummaryDetail(
            topic=data.get("topic", ""),
            segments=data.get("segments", []),
            key_points=data.get("key_points", []),
        )

    # 标注视频（PLATE）：单独字段，供前端主播放器优先使用
    annotated_video_url = None
    if (out_dir / "annotated.mp4").exists():
        annotated_video_url = art_url("annotated.mp4")

    # 原视频 URL：产物目录里找视频文件（排除 annotated.mp4，那是标注结果）
    video_url = None
    for name in os.listdir(out_dir):
        low = name.lower()
        if low == "annotated.mp4":
            continue
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

    # 车牌/车辆轨迹（PLATE）：展示有标签的轨迹（车牌号，或车辆模式的「车辆N」）；
    # 无标签、或无牌且单帧（噪声）→ 不展示（诚实，只列有意义的）。
    plate_tracks = []
    for tk in data.get("plate_tracks", []):
        label = tk.get("label") or tk.get("plate_text", "")
        has_plate = bool(tk.get("plate_text"))
        if not label or (not has_plate and tk.get("n_frames", 0) < 2):
            continue
        dets = tk.get("detections", [])
        best = max(dets, key=lambda d: d.get("ocr_confidence", 0.0), default=None)
        best_frame = best.get("frame", "") if best else ""
        first_t, last_t = tk.get("first_t", 0.0), tk.get("last_t", 0.0)
        plate_tracks.append(PlateTrackOut(
            track_id=tk.get("track_id", 0), plate_text=tk.get("plate_text", ""),
            label=label, confidence=tk.get("confidence", 0.0),
            plate_color=tk.get("plate_color"),
            first_t=first_t, last_t=last_t,
            hms_range=f"{_ts.hms(first_t)}–{_ts.hms(last_t)}",
            n_frames=tk.get("n_frames", 0), method=tk.get("method", "vote"),
            best_frame_url=art_url(f"frames/{best_frame}") if best_frame else "",
            caveats=tk.get("caveats", ""),
        ))

    return AnswerOut(
        intent=data.get("intent", ""), target=data.get("target"),
        answer=data.get("answer", ""), summary_detail=detail,
        evidence=evidence, confidence=data.get("confidence", 0.0),
        caveats=data.get("caveats", ""), video_url=video_url,
        query_image_url=query_image_url, grids=grids,
        annotated_video_url=annotated_video_url, plate_tracks=plate_tracks,
    )
