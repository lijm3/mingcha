"""车辆追踪模式（V 系列）单测 —— detect_and_track 选 YOLO、车内找牌、标签、from_vehicle 组装。

全 mock：不真调 ultralytics/HyperLPR3，不烧 token，不需 [vehicle]/[plate] extra。
"""
import mingcha.plate as plate
from mingcha import assembler, config
from mingcha.plate.boxes import PlateBox
from mingcha.types import BBox, PlateDetection, PlateTrack


def _vtrack(tid, n, first=0.0, plate_text="", conf=0.0, kind="vehicle"):
    dets = [PlateDetection(t=first + i, frame=f"frame_{i:03d}.jpg",
                           bbox=BBox(x=10 + i, y=20, w=40, h=30), ocr_confidence=0.8)
            for i in range(n)]
    return PlateTrack(track_id=tid, plate_text=plate_text, confidence=conf, first_t=first,
                      last_t=first + max(0, n - 1), n_frames=n, kind=kind, detections=dets)


def test_detect_and_track_uses_yolo_in_vehicle_mode(monkeypatch):
    monkeypatch.setattr(config, "PLATE_TRACK_MODE", "vehicle")
    from mingcha.plate import vehicle
    calls = {"n": 0}

    def fake_detect(path, *, gpu=True, det_thresh=None):
        calls["n"] += 1
        return [PlateBox(bbox=BBox(x=10 + calls["n"] * 3, y=20, w=40, h=30), det_conf=0.9)]
    monkeypatch.setattr(vehicle, "detect", fake_detect)

    tracks = plate.detect_and_track([(f"f{i}.jpg", float(i)) for i in range(5)])
    assert calls["n"] == 5                       # 逐帧走了 YOLO 车辆检测
    assert len(tracks) == 1 and tracks[0].n_frames == 5   # 大目标稳定关联成多帧


def test_assign_labels_vehicle_without_plate():
    tr = _vtrack(3, n=6)                          # 车辆多帧、无牌
    plate._assign_labels([tr])
    assert tr.label == "车辆3"
    assert tr.confidence > 0                      # 回退为车辆检测置信均值（0.8）


def test_assign_labels_with_plate():
    tr = _vtrack(1, n=6, plate_text="京A12345", conf=0.9)
    plate._assign_labels([tr])
    assert tr.label == "京A12345"


def test_from_vehicle_shows_tracked_cars():
    a = _vtrack(1, n=8, plate_text="京A12345", conf=0.9)   # 有牌
    b = _vtrack(2, n=5)                                    # 无牌多帧 → 展示
    c = _vtrack(3, n=1)                                    # 单帧噪声 → 剔除
    for t in (a, b, c):
        plate._assign_labels([t])
    ans = assembler.from_plate([a, b, c], "out/annotated.mp4", "out", caveats="")
    assert ans.intent == "PLATE"
    assert "追踪到 2 辆车" in ans.answer                    # a、b 稳定；c 被剔
    assert "京A12345" in ans.answer
    assert any(e.plate_text == "京A12345" for e in ans.evidence)
    assert ans.annotated_video == "out/annotated.mp4"


def test_from_vehicle_all_unplated():
    b = _vtrack(2, n=5)
    plate._assign_labels([b])
    ans = assembler.from_plate([b], None, "out", caveats="")
    assert "未能读出车牌号" in ans.answer                   # 只框选跟随、无牌，如实说明


def test_find_plates_in_vehicles_fills_detections(monkeypatch, tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(3):
        (frames_dir / f"frame_{i:03d}.jpg").write_bytes(b"")
    tr = _vtrack(1, n=3)
    monkeypatch.setattr(plate, "_detect_plate_in_box",
                        lambda fp, bbox, gpu=True: ("京A12345", 0.9, "蓝"))
    plate.find_plates_in_vehicles([tr], str(frames_dir))
    assert all(d.text == "京A12345" and d.plate_color == "蓝" for d in tr.detections)
