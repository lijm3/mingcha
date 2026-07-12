"""管线规划 —— FR-3。把需求 §FR-3 的参数表编码为纯函数；数值集中便于实测调优。
对应设计文档 §5。
"""
from __future__ import annotations

from . import config
from .types import Intent, Plan


def plan(intents: list[Intent]) -> Plan:
    """多意图取“最严格”并集：只要含 LOCATE/MODERATE/VISUAL_LOCATE 就要时间戳 + 密采样。"""
    # PLATE 采样诉求与其它意图根本不同：追踪要连续帧、关去重、关场景选择、高分辨率、
    # 不拼图（不走 VLM 拼图范式）。故独立分支、置于最前，不并入 dense 逻辑（§5.2）。
    if Intent.PLATE in intents:
        return Plan(intents=intents,
                    fps_floor=1.0 / config.PLATE_SAMPLE_FPS,  # 5fps → every_n=round(video_fps*0.2)
                    scene=0.0,                # 关场景选择 → extract 走纯均匀抽帧（scene<=0 分支）
                    dedup_threshold=0,        # 关去重（连续帧是追踪前提）
                    max_frames=None,          # 不封顶（受 MAX_FRAMES_HARD 护栏兜底）
                    need_timestamps=True,     # G1
                    grid_label_time=False,    # 不拼 grids（PLATE 不走拼图 VLM 范式）
                    keep_audio=True,          # 回写 annotated.mp4 保留原音轨
                    frame_width=config.PLATE_FRAME_WIDTH)
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
