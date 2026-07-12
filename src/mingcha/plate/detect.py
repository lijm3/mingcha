"""车牌检测 —— FR-5.5.1 / §6.2。主引擎 HyperLPR3（国产、专为中国车牌，检测+识别一体，
onnxruntime 后端；蓝/黄/绿新能源/双层黄）。

门面稳定、内部引擎可换（§16：召回不足可切 YOLO-plate + PaddleOCR，不改 analyzer）。重依赖
（hyperlpr3 / opencv / onnxruntime）全部**惰性 import**：顶层只依赖 dataclass 与 types.BBox，
使无 [plate] extra 时 import 本模块也不炸——真正调用 detect() 才触发依赖探测（缺则 ImportError
由上层 analyzer 兜底降级）。
"""
from __future__ import annotations

from ..types import BBox
from .boxes import PlateBox

# HyperLPR3 车牌类型编号 → 颜色标签（以官方 typeDict 为准，门面隔离便于一处校正）。
_TYPE_COLOR = {0: "蓝", 1: "黄", 2: "白", 3: "绿(新能源)", 4: "黑", 9: "双层黄"}


def detect(frame_path: str, *, gpu: bool = True, det_thresh: float = 0.0) -> list[PlateBox]:
    """一帧 → 若干车牌框（bbox + 初检文本 + 颜色 + 检测置信度）。

    - 未装 [plate] / 引擎不可用 → 惰性 import 抛 ImportError，由上层 analyzer 兜底降级；
    - 单帧解码失败 / 无车牌 → 返回 []（不抛，追踪照常，丢一帧不断链，§11）。
    HyperLPR3 返回 [text, confidence, type, box=[x1,y1,x2,y2]]（真实字段以官方 API 为准）。
    """
    import cv2
    from .engine import get_catcher
    catcher = get_catcher(gpu)
    img = cv2.imread(frame_path)
    if img is None:
        return []
    out: list[PlateBox] = []
    for item in (catcher(img) or []):
        text, conf, ptype, box = _unpack(item)
        if float(conf) < det_thresh:
            continue
        x1, y1, x2, y2 = (int(v) for v in list(box)[:4])
        out.append(PlateBox(
            bbox=BBox(x=x1, y=y1, w=max(0, x2 - x1), h=max(0, y2 - y1)),
            det_conf=float(conf), det_text=str(text or ""),
            color=_TYPE_COLOR.get(int(ptype)) if ptype is not None else None))
    return out


def _unpack(item):
    """兼容 HyperLPR3 不同版本的返回形态（list/tuple 顺序 text,conf,type,box；或 dict）。"""
    if isinstance(item, (list, tuple)) and len(item) >= 4:
        return item[0], item[1], item[2], item[3]
    if isinstance(item, dict):
        return (item.get("text") or item.get("code", ""),
                item.get("confidence", 0.0), item.get("type"),
                item.get("box") or item.get("bbox", [0, 0, 0, 0]))
    return "", 0.0, None, [0, 0, 0, 0]
