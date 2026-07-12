"""PLATE 分析 —— FR-5.5 / §6（主编排）。与四个现有 analyzer 同构：消费预处理产出的连续帧
+ timestamps → 调 plate 门面（检测→追踪→OCR→融合→回写）→ 调 assembler 组装 Answer。

analyzer 只依赖 plate 门面 + assembler，不知背后是 HyperLPR3 还是 YOLO（与四个现有 analyzer
只依赖 llm 门面同构，换引擎不动此文件）。永不崩铺底（§11）：未装 [plate] → 可读降级；
其它异常 → 诚实否定 + 失败说明，绝不中断主流程。
"""
from __future__ import annotations

from .. import assembler, plate
from ..config import PLATE_SAMPLE_FPS
from ..types import Answer, Plan


def _caveats() -> str:
    return (f"车牌识别基于约 {PLATE_SAMPLE_FPS:.0f}fps 密采 + 多目标追踪 + 多帧投票；"
            f"极短暂出现、远处或严重模糊的车牌可能漏检或被压低置信度。")


def analyze(out_dir: str, plan: Plan, target: str, source_video: str, *,
            vision_model=None, progress=None) -> Answer:
    """PLATE 主入口。target 对车牌识别无实义（保持与其它 analyzer 同签名）。见设计文档 §6/§7。"""
    try:
        tracks, annotated = plate.run_pipeline(out_dir, plan, source_video,
                                               vision_model=vision_model, progress=progress)
    except ImportError as e:                # 未装 [plate] extra → 可读降级（永不崩）
        return assembler.plate_unavailable(out_dir, e)
    except Exception as e:  # noqa: BLE001 —— 任何失败降级为诚实否定，不中断主流程
        return assembler.from_plate([], None, out_dir, caveats=_failure_caveat(e))
    return assembler.from_plate(tracks, annotated, out_dir, caveats=_caveats())


def _failure_caveat(e: Exception) -> str:
    """把车牌分析失败转成诚实且可操作的 caveat。首次权重下载/解压占用（WinError 32）
    等临时错误给出「重试即可」的指引。"""
    msg = f"{type(e).__name__}：{e}"
    s = str(e).lower()
    if ".hyperlpr3" in s or "20230229" in s or "winerror 32" in s:
        return (f"车牌模型权重首次下载/解压被占用（{msg}）。多为首次运行的临时冲突，"
                f"请重试一次即可（权重已就绪时会跳过下载）；仍失败可手动预置权重到 ~/.hyperlpr3/。")
    return f"车牌分析未能完成（{msg}）。"
