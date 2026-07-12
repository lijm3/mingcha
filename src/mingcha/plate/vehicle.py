"""车辆检测 —— YOLOv8（COCO 预训练）检测 car/bus/truck/motorcycle，用于「追踪车辆」模式。

远距离俯拍车流里车牌只有十几像素、HyperLPR3 检不到连续帧 → 无法追踪（表现为「全单帧轨迹」）。
车辆是大目标、召回高、移动连续，改追车辆即可稳定关联出多帧轨迹、回写出跟随效果；车牌退化为
车辆框内的可选二次识别（见 plate.run_pipeline 的 vehicle 模式）。

重依赖 ultralytics（含 torch）隔离在 [vehicle] extra、惰性 import：顶层只依赖 dataclass 与
types.BBox；缺依赖 → 调用时抛 ImportError，由 analyzer.plate 兜底降级（永不崩）。输出复用
PlateBox（bbox + 置信 + 车型名），使 track.associate / annotate 无需改动即可复用。
"""
from __future__ import annotations

from ..types import BBox
from .boxes import PlateBox

# COCO 类编号 → 中文车型名（PlateBox.color 复用来承载）
_COCO_VEHICLE = {2: "小汽车", 3: "摩托车", 5: "公交车", 7: "卡车"}

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        from .. import config
        _MODEL = YOLO(config.PLATE_VEHICLE_MODEL)   # 首次自动下载 COCO 预训练权重
    return _MODEL


def detect(frame_path: str, *, gpu: bool = True, det_thresh=None) -> list[PlateBox]:
    """一帧 → 若干车辆框（bbox + 车型名 + 检测置信）。

    未装 [vehicle] → 惰性 import 抛 ImportError（由上层降级）；无车/失败 → []（不抛，追踪照常）。
    det_thresh 缺省用 config.PLATE_VEHICLE_CONF；gpu 由 ultralytics/torch 自动选 device。
    """
    from .. import config
    conf = config.PLATE_VEHICLE_CONF if det_thresh is None else det_thresh
    classes = list(config.PLATE_VEHICLE_CLASSES)
    results = _get_model()(frame_path, conf=conf, classes=classes, verbose=False)
    out: list[PlateBox] = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for b in boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist()[:4])
            # det_text/color 留空：车辆框不含车牌读数（框内二次识别时才填），避免污染 fuse 投票
            out.append(PlateBox(
                bbox=BBox(x=x1, y=y1, w=max(0, x2 - x1), h=max(0, y2 - y1)),
                det_conf=float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0))
    return out
