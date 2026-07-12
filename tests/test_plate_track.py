"""PLATE 多目标追踪单测 —— §6.3（G6 空间-时序追踪地基）。

纯 CPU、确定性、零重依赖：不需 GPU/[plate] extra。验证跨帧 bbox 关联为稳定 track_id、
短暂遮挡续链、多车不混轨。
"""
from mingcha.plate import track
from mingcha.plate.boxes import PlateBox
from mingcha.types import BBox


def _box(x, y, w=40, h=15, text="京A12345", conf=0.9):
    return PlateBox(bbox=BBox(x=x, y=y, w=w, h=h), det_conf=conf, det_text=text)


def test_single_car_moving_stays_one_track():
    # 一辆车 bbox 每帧右移 5px（相邻帧高 IoU + 线速外推）→ 单一 track
    frames = [(float(i), f"frame_{i:03d}.jpg", [_box(10 + i * 5, 20)]) for i in range(6)]
    tracks = track.associate(frames)
    assert len(tracks) == 1
    assert tracks[0].n_frames == 6
    assert tracks[0].first_t == 0.0 and tracks[0].last_t == 5.0


def test_two_cars_not_mixed():
    frames = [(float(i), f"f{i}.jpg", [_box(10 + i * 3, 20), _box(300 + i * 3, 200)])
              for i in range(5)]
    tracks = track.associate(frames)
    assert len(tracks) == 2
    assert all(t.n_frames == 5 for t in tracks)
    # 两条轨迹分别稳定在左上 / 右下，未串轨
    xs = sorted(t.detections[0].bbox.x for t in tracks)
    assert xs[0] < 100 < xs[1]


def test_short_gap_bridged():
    # 第 3 帧漏检（空框），gap=1 ≤ MAX_GAP → 续上为单 track，不拆成两条
    frames = []
    for i in range(6):
        boxes = [] if i == 3 else [_box(10 + i * 5, 20)]
        frames.append((float(i), f"f{i}.jpg", boxes))
    tracks = track.associate(frames)
    assert len(tracks) == 1
    assert tracks[0].n_frames == 5          # 漏检帧不计入检测数


def test_iou_basic():
    assert track.iou(BBox(x=0, y=0, w=10, h=10), BBox(x=0, y=0, w=10, h=10)) == 1.0
    assert track.iou(BBox(x=0, y=0, w=10, h=10), BBox(x=100, y=100, w=10, h=10)) == 0.0
