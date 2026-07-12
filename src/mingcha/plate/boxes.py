"""PlateBox —— 检测阶段的单框产物（轻量 dataclass，零重依赖）。

detect / track / ocr 共享此类型。刻意放在独立轻量模块（而非 detect.py），使 track.py
能 import 它而**不触发** detect.py 里 hyperlpr3/onnxruntime 的惰性 import——从而 track 与
fuse 一样保持纯 CPU、可在无 [plate] extra 的环境里单测。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..types import BBox


@dataclass
class PlateBox:
    """一帧里一个车牌的检测框 + 初检信息（未追踪、未融合）。"""
    bbox: BBox                                  # G6 空间框（抽帧分辨率坐标系）
    det_conf: float = 0.0                       # 检测置信度
    det_text: str = ""                          # 检测顺带的初检文本（HyperLPR3 一体模型）
    color: str | None = None                    # 蓝 / 黄 / 绿(新能源) / 双层黄
    corners: list = field(default_factory=list)  # 四角点 [(x,y)×4]，透视矫正用（§6.9，P5.4）
