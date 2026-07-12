"""轻量多目标追踪 —— FR-5.5.3 / §6.3（G6 空间-时序追踪地基）。

SORT / ByteTrack 简化版：略去卡尔曼，用「上一帧 bbox + 线速外推」预测 + IoU 贪心关联 +
断链容忍（gap ≤ PLATE_TRACK_MAX_GAP 内可续上，对付短暂遮挡）。纯 Python，不引 torch / onnx
（§16 风险项「优先轻量独立实现，避免 torch 进 [plate] 主 extra」），CPU 可跑、确定性可单测。

追踪把「同一车牌的多帧读数」聚成一条 track，是 §6.5 多帧投票的前提——没有 track，
多帧信息无从关联。此处只做分组与时间区间，plate_text 尚未定案（交 fuse.fuse_tracks 融合）。
"""
from __future__ import annotations

from .. import config
from ..types import BBox, PlateDetection, PlateTrack
from .boxes import PlateBox

# 一帧的检测结果：(绝对时间 t, 帧文件名, 该帧所有车牌框)
FrameDets = tuple


def iou(a: BBox, b: BBox) -> float:
    """两个 bbox 的交并比（IoU）。"""
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2 = min(a.x + a.w, b.x + b.w)
    iy2 = min(a.y + a.h, b.y + b.h)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


class _Track:
    """内部可变追踪状态（不外泄；associate 结束后转成不可变的 PlateTrack）。"""
    __slots__ = ("id", "bbox", "vx", "vy", "gap", "dets", "first_t", "last_t")

    def __init__(self, tid: int, t: float, frame: str, box: PlateBox):
        self.id = tid
        self.bbox = box.bbox
        self.vx = 0.0                 # 中心 x 速度（像素/帧）
        self.vy = 0.0
        self.gap = 0                  # 连续未匹配帧数
        self.dets = [_to_det(t, frame, box)]
        self.first_t = t
        self.last_t = t

    def predict(self) -> BBox:
        """线速外推：中心按 (vx,vy)×(gap+1) 平移，宽高不变（快速驶过的车更易续上）。"""
        shift = self.gap + 1
        return BBox(x=int(self.bbox.x + self.vx * shift),
                    y=int(self.bbox.y + self.vy * shift),
                    w=self.bbox.w, h=self.bbox.h)

    def update(self, t: float, frame: str, box: PlateBox) -> None:
        span = self.gap + 1
        self.vx = (box.bbox.x - self.bbox.x) / span
        self.vy = (box.bbox.y - self.bbox.y) / span
        self.bbox = box.bbox
        self.gap = 0
        self.last_t = t
        self.dets.append(_to_det(t, frame, box))


def associate(dets_by_frame: list, *, iou_thresh: float | None = None,
              max_gap: int | None = None) -> list[PlateTrack]:
    """按时间序的逐帧检测 [(t, frame, [PlateBox,...]), ...] → 若干 track。

    每帧：预测各活跃 track 的 bbox → 与本帧检测框按 IoU 降序贪心一对一匹配 → 未匹配 track
    累加 gap（超 max_gap 归档）、未匹配框新建 track。plate_text 未定案，交 fuse 融合。
    """
    iou_thresh = config.PLATE_TRACK_IOU if iou_thresh is None else iou_thresh
    max_gap = config.PLATE_TRACK_MAX_GAP if max_gap is None else max_gap

    active: list[_Track] = []
    finished: list[_Track] = []
    next_id = 1

    for t, frame, boxes in dets_by_frame:
        pairs = []  # (iou, track_idx, box_idx)
        for ti, tr in enumerate(active):
            pred = tr.predict()
            for bi, box in enumerate(boxes):
                v = iou(pred, box.bbox)
                if v >= iou_thresh:
                    pairs.append((v, ti, bi))
        pairs.sort(key=lambda p: p[0], reverse=True)

        used_t: set = set()
        used_b: set = set()
        for _, ti, bi in pairs:
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            active[ti].update(t, frame, boxes[bi])

        for ti, tr in enumerate(active):
            if ti not in used_t:
                tr.gap += 1
        for bi, box in enumerate(boxes):
            if bi not in used_b:
                active.append(_Track(next_id, t, frame, box))
                next_id += 1

        keep: list[_Track] = []
        for tr in active:
            (finished if tr.gap > max_gap else keep).append(tr)
        active = keep

    finished.extend(active)
    finished.sort(key=lambda tr: (tr.first_t, tr.id))
    return [_to_plate_track(tr) for tr in finished]


def _to_det(t: float, frame: str, box: PlateBox) -> PlateDetection:
    return PlateDetection(t=t, frame=frame, bbox=box.bbox, text=box.det_text,
                          ocr_confidence=box.det_conf, plate_color=box.color)


def _to_plate_track(tr: _Track) -> PlateTrack:
    return PlateTrack(track_id=tr.id, plate_text="", confidence=0.0,
                      first_t=tr.first_t, last_t=tr.last_t,
                      n_frames=len(tr.dets), detections=list(tr.dets))
