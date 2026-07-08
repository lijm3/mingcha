"""明察的数据结构：意图、时间戳、管线配置、证据、答案，以及各分析器的
结构化输出 schema。统一用 Pydantic，供 llm/structured.py 跨 provider 校验 LLM 返回。

对应设计文档 §2。
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    SUMMARY = "SUMMARY"
    LOCATE = "LOCATE"
    MODERATE = "MODERATE"
    VISUAL_LOCATE = "VISUAL_LOCATE"


# ---- FR-2 意图分类的 LLM 结构化输出 ----
class IntentResult(BaseModel):
    intents: list[Intent] = Field(default_factory=list)  # 允许多标签
    target: str | None = Field(None, description="检测/定位目标实体，文字型来自 prompt")
    return_scope: str = Field("earliest", description="earliest | all_ranges | exists")
    reason: str = ""  # 可解释（FR-2.5）


# ---- FR-4 时间戳地基 (G1) ----
class FrameStamp(BaseModel):
    frame: str  # "frame_001.jpg"
    t: float    # 秒，保留小数
    hms: str    # "00:00:12.500"


# ---- FR-3 管线配置 ----
class Plan(BaseModel):
    intents: list[Intent]
    fps_floor: float
    scene: float
    dedup_threshold: float
    max_frames: int | None       # None = 不封顶（分段）
    need_timestamps: bool
    grid_label_time: bool
    keep_audio: bool


# ---- FR-5/FR-6 证据与答案 ----
class Evidence(BaseModel):
    frame: str
    t: float
    hms: str
    confidence: float
    similarity: float | None = None  # 仅 VISUAL_LOCATE
    note: str = ""


class Answer(BaseModel):
    intent: str                     # 主意图或 "SUMMARY+MODERATE"
    target: str | None = None
    query_image: str | None = None
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = 0.0
    caveats: str = ""               # NFR-1 采样局限；否定结论必填非空
    artifacts_dir: str = ""


# ---- 各分析器的结构化输出 schema（供 llm.vision_structured 校验）----
class SummarySchema(BaseModel):
    """FR-5.1 SUMMARY 的结构化摘要。"""
    topic: str = Field("", description="视频主题 / 一句话概括")
    segments: list[str] = Field(default_factory=list, description="分段脉络，按时间顺序")
    key_points: list[str] = Field(default_factory=list, description="关键结论 / 要点")
    summary: str = Field("", description="完整结构化摘要文本")


class HitSchema(BaseModel):
    """FR-5.2 LOCATE 逐帧判定。"""
    present: bool = False
    confidence: float = 0.0
    note: str = ""


class ModerateSchema(BaseModel):
    """FR-5.3 MODERATE 逐帧审核（只判定是否存在，不细节化）。"""
    present: bool = False
    confidence: float = 0.0
    t: float | None = None
    note: str = ""


class VisualHitSchema(BaseModel):
    """FR-5.4 VISUAL_LOCATE 语义确认：same / similar / no。"""
    verdict: str = "no"             # same | similar | no
    similarity: float = 0.0
    confidence: float = 0.0
    note: str = ""
