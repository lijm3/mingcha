"""明察的数据结构：意图、时间戳、管线配置、证据、答案，以及各分析器的
结构化输出 schema。统一用 Pydantic，供 llm/structured.py 跨 provider 校验 LLM 返回。

对应设计文档 §2；PLATE 车牌识别扩展见 docs/车牌识别-详细设计文档.md。
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    SUMMARY = "SUMMARY"
    LOCATE = "LOCATE"
    MODERATE = "MODERATE"
    VISUAL_LOCATE = "VISUAL_LOCATE"
    PLATE = "PLATE"                       # 车牌识别 + 高亮追踪（FR-5.5）


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
    frame_width: int = 640       # 抽帧宽度（PLATE 用 1280，车牌小目标；默认 640 向后兼容）


# ---- G6 空间-时序地基（PLATE，FR-5.5）----
class BBox(BaseModel):
    """车牌/目标在帧内的像素框（左上角 + 宽高）。"""
    x: int
    y: int
    w: int
    h: int


class PlateDetection(BaseModel):
    """单帧单车牌的检测 + OCR 结果（多帧融合的原料）。"""
    t: float                                            # G1 绝对时间
    frame: str                                          # 来源帧文件名
    bbox: BBox                                          # G6 空间框
    text: str = ""                                      # 该帧 OCR 读数（未融合）
    char_confidences: list[float] = Field(default_factory=list)  # 逐字符置信度
    ocr_confidence: float = 0.0
    plate_color: str | None = None


class PlateTrack(BaseModel):
    """一条车牌轨迹（跨帧同一车牌，多帧融合后定案）。"""
    track_id: int
    plate_text: str                                     # 多帧投票 + 规则约束后的车牌号
    confidence: float = 0.0                             # 融合置信度
    plate_color: str | None = None                      # 蓝 / 黄 / 绿(新能源) / 双层黄
    first_t: float = 0.0                                # 首次出现（G1）
    last_t: float = 0.0                                 # 末次出现
    n_frames: int = 0                                   # 检测帧数（投票样本量）
    method: str = "vote"                                # vote|rule_fixed|superres|vlm_corrected
    detections: list[PlateDetection] = Field(default_factory=list)  # 全部检测（回写/复核）
    caveats: str = ""                                   # 该车牌的个体局限
    label: str = ""                                     # 展示/回写标签：车牌号，或车辆模式无牌时的「车辆N」
    kind: str = "plate"                                 # plate=车牌轨迹 | vehicle=车辆轨迹（可能无牌）


# ---- FR-5/FR-6 证据与答案 ----
class Evidence(BaseModel):
    frame: str
    t: float
    hms: str
    confidence: float
    similarity: float | None = None  # 仅 VISUAL_LOCATE
    verdict: str | None = None       # 仅 VISUAL_LOCATE: "same"|"similar"，供前端徽章区分同一个体/同类外观
    note: str = ""
    # —— 仅 PLATE（FR-5.5）：空间证据 ——
    bbox: BBox | None = None         # 车牌在代表帧的位置
    track_id: int | None = None      # 跨帧同一车牌标识
    plate_text: str | None = None    # 最终车牌号
    plate_color: str | None = None   # 车牌颜色


class Answer(BaseModel):
    intent: str                     # 主意图或 "SUMMARY+MODERATE"
    target: str | None = None
    query_image: str | None = None
    answer: str
    topic: str = ""                 # 仅 SUMMARY：主题/一句话概括（来自 SummarySchema.topic）
    segments: list[str] = Field(default_factory=list)   # 仅 SUMMARY：分段脉络
    key_points: list[str] = Field(default_factory=list)  # 仅 SUMMARY：关键要点
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = 0.0
    caveats: str = ""               # NFR-1 采样局限；否定结论必填非空
    artifacts_dir: str = ""
    # —— 仅 PLATE（FR-5.5）——
    annotated_video: str | None = None                  # 高亮标注视频落盘路径
    plate_tracks: list[PlateTrack] = Field(default_factory=list)  # 结构化车牌轨迹


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


class PlateVLMSchema(BaseModel):
    """FR-5.5 PLATE 的 VLM 兜底纠错输出（P5.4 启用）。"""
    plate_text: str = ""
    plate_color: str | None = None
    confidence: float = 0.0
    reasoning: str = ""             # 可解释：为何这么读（规则依据）
