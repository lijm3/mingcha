"""管线规划 —— FR-3。把需求 §FR-3 的参数表编码为纯函数；数值集中便于实测调优。
对应设计文档 §5。
"""
from __future__ import annotations

from .types import Intent, Plan


def plan(intents: list[Intent]) -> Plan:
    """多意图取“最严格”并集：只要含 LOCATE/MODERATE/VISUAL_LOCATE 就要时间戳 + 密采样。"""
    dense = any(i in (Intent.LOCATE, Intent.MODERATE, Intent.VISUAL_LOCATE) for i in intents)
    moderate = Intent.MODERATE in intents
    if not dense:  # 纯 SUMMARY（首期统一走带时间戳路径，好让 G1 地基第一期即被验证）
        return Plan(intents=intents, fps_floor=1.0, scene=0.30, dedup_threshold=8,
                    max_frames=150, need_timestamps=True, grid_label_time=True,
                    keep_audio=False)
    return Plan(intents=intents, fps_floor=0.5, scene=0.20,
                dedup_threshold=0 if moderate else 3.5,      # MODERATE 关去重（§6.3）
                max_frames=None if moderate else 400,        # MODERATE 不封顶 → 分段
                need_timestamps=True, grid_label_time=True,
                keep_audio=moderate)                          # 暴力/尖叫等声学线索
