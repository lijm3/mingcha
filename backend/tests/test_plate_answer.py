"""PLATE 结果组装：build_answer_out 把含 plate_tracks/annotated_video 的 answer.json
包装成带产物 URL 的 AnswerOut（区分原视频/标注视频、填空间证据、构造轨迹列表）。"""
import json


def _write_answer(out, data):
    (out / "answer.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_build_answer_out_plate(tmp_path):
    from app.answer_out import build_answer_out
    out = tmp_path
    (out / "frames").mkdir()
    (out / "annotated.mp4").write_bytes(b"")
    (out / "source.mp4").write_bytes(b"")
    _write_answer(out, {
        "intent": "PLATE",
        "answer": "识别到 1 辆车的车牌：京A12345。",
        "evidence": [{
            "frame": "frame_003.jpg", "t": 3.0, "hms": "00:00:03.000", "confidence": 0.92,
            "bbox": {"x": 100, "y": 200, "w": 120, "h": 40}, "track_id": 1,
            "plate_text": "京A12345", "plate_color": "蓝", "note": "vote·20帧",
        }],
        "confidence": 0.92, "caveats": "采样局限声明",
        "annotated_video": str(out / "annotated.mp4"),
        "plate_tracks": [{
            "track_id": 1, "plate_text": "京A12345", "confidence": 0.92, "plate_color": "蓝",
            "first_t": 3.0, "last_t": 7.0, "n_frames": 20, "method": "vote", "caveats": "",
            "detections": [{"frame": "frame_003.jpg", "ocr_confidence": 0.92,
                            "bbox": {"x": 100, "y": 200, "w": 120, "h": 40}, "t": 3.0}],
        }],
    })

    ao = build_answer_out(out, "tid", "tok")
    assert ao is not None and ao.intent == "PLATE"
    # 标注视频单独字段；原视频排除 annotated.mp4
    assert ao.annotated_video_url and "annotated.mp4" in ao.annotated_video_url
    assert ao.video_url and "source.mp4" in ao.video_url
    assert "annotated.mp4" not in (ao.video_url or "")
    # 车牌轨迹列表
    assert len(ao.plate_tracks) == 1
    tk = ao.plate_tracks[0]
    assert tk.plate_text == "京A12345"
    assert tk.hms_range == "00:00:03.000–00:00:07.000"
    assert "frames/frame_003.jpg" in tk.best_frame_url
    # evidence 空间字段
    ev = ao.evidence[0]
    assert ev.plate_text == "京A12345" and ev.track_id == 1
    assert ev.bbox is not None and ev.bbox.x == 100 and ev.bbox.w == 120


def test_build_answer_out_plate_blurry_track_excluded(tmp_path):
    """全模糊 track（plate_text 空）不进轨迹列表（诚实：只列可读的）。"""
    from app.answer_out import build_answer_out
    out = tmp_path
    _write_answer(out, {
        "intent": "PLATE", "answer": "检出 1 处车牌区域，但过于模糊。",
        "evidence": [], "confidence": 0.0, "caveats": "过于模糊",
        "plate_tracks": [{"track_id": 1, "plate_text": "", "confidence": 0.0,
                          "first_t": 1.0, "last_t": 2.0, "detections": []}],
    })
    ao = build_answer_out(out, "tid", "tok")
    assert ao.plate_tracks == [] and ao.annotated_video_url is None


def test_build_answer_out_summary_unaffected(tmp_path):
    """回归：SUMMARY（无 plate 字段）仍正常，plate 字段取默认空。"""
    from app.answer_out import build_answer_out
    out = tmp_path
    _write_answer(out, {
        "intent": "SUMMARY", "answer": "摘要", "topic": "主题",
        "evidence": [], "confidence": 0.8, "caveats": "",
    })
    ao = build_answer_out(out, "tid", "tok")
    assert ao.intent == "SUMMARY"
    assert ao.plate_tracks == [] and ao.annotated_video_url is None
