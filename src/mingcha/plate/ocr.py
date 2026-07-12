"""车牌 OCR —— FR-5.5.2 / §6.4。对每个检测框做（二次精）识别，产出逐字符置信度。

HyperLPR3 检测已顺带出文本（填入 PlateBox.det_text）；本模块对**车牌裁剪小图**二次识别，
通常比全帧检测顺带的读数更准（尤其接在 §6.9 透视矫正 / §6.7 超分之后，P5.4）。逐字符
置信度是 §6.5 多帧投票的关键输入——HyperLPR3 未直接暴露 per-char 时，退回整体置信度铺平
（投票仍可用）。识别引擎与 detect **共享同一 HyperLPR3 单例**（见 engine.py，避免首次权重
下载冲突与重复加载）；单框识别失败返回空，不抛（永不崩，交多帧投票兜）。
"""
from __future__ import annotations

from ..types import BBox


def read(frame_path: str, bbox: BBox, *, gpu: bool = True):
    """裁剪 bbox 小图 → 二次 OCR。返回 (text, char_confidences, ocr_confidence, color)。
    解码失败 / 空框 / 读不出 → ("", [], 0.0, None)（不抛，永不崩）。"""
    import cv2
    img = cv2.imread(frame_path)
    if img is None:
        return "", [], 0.0, None
    h, w = img.shape[:2]
    x1, y1 = max(0, bbox.x), max(0, bbox.y)
    x2, y2 = min(w, bbox.x + bbox.w), min(h, bbox.y + bbox.h)
    if x2 <= x1 or y2 <= y1:
        return "", [], 0.0, None
    crop = img[y1:y2, x1:x2]
    try:
        from .engine import get_catcher
        results = get_catcher(gpu)(crop) or []
    except Exception:  # noqa: BLE001 —— 单框识别失败不拖垮整条 track
        return "", [], 0.0, None
    if not results:
        return "", [], 0.0, None
    best = max(results, key=lambda r: r[1] if len(r) > 1 else 0.0)
    text = str(best[0] or "")
    conf = float(best[1]) if len(best) > 1 else 0.0
    from .detect import _TYPE_COLOR
    color = _TYPE_COLOR.get(int(best[2])) if len(best) > 2 and best[2] is not None else None
    char_confs = [conf] * len(text)   # HyperLPR3 无 per-char → 用整体铺平（投票仍可用）
    return text, char_confs, conf, color
