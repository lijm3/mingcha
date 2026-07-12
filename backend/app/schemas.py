"""API 请求/响应 Pydantic 模型（§5.3）。复用内核 types.Answer/Evidence，
在 API 层包一层产物 URL（*_url 由 task_token 拼装，不落进 answer.json）。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

TaskState = Literal["queued", "downloading", "extracting", "transcribing",
                    "analyzing", "assembling", "done", "error", "cancelled"]
IntentName = Literal["auto", "SUMMARY", "LOCATE", "MODERATE", "VISUAL_LOCATE", "PLATE"]


class ProviderOverride(BaseModel):
    provider: str | None = None          # 一次性切所有角色
    vision_model: str | None = None      # 形如 "openai:gpt-5.5"
    classify_model: str | None = None


class CreateTaskJson(BaseModel):
    """URL 源建任务（application/json）。上传源走 multipart，不用这个模型。"""
    url: str
    prompt: str = ""
    intent: IntentName = "auto"
    override: ProviderOverride = ProviderOverride()
    cookies_from_browser: str | None = None


class CreateTaskResponse(BaseModel):
    task_id: str
    task_token: str                      # 访问该任务产物/状态/SSE 的凭证


class TaskStatus(BaseModel):
    task_id: str
    state: TaskState
    progress: float = 0.0                # 0~1
    stage_note: str = ""                 # 人类可读，如 "去重后 40 帧"
    intent: str | None = None            # 分类后回填
    error: str | None = None
    created_at: float = 0.0
    caveats: str = ""


class BBoxOut(BaseModel):                # 仅 PLATE：车牌在代表帧的像素框（原分辨率坐标系）
    x: int
    y: int
    w: int
    h: int


class EvidenceOut(BaseModel):
    frame: str
    frame_url: str                       # /artifacts/{id}/frames/frame_007.jpg?token=
    t: float
    hms: str
    confidence: float
    similarity: float | None = None      # 仅 VISUAL_LOCATE
    verdict: str | None = None           # 仅 VISUAL_LOCATE: "same"|"similar"
    note: str = ""
    # —— 仅 PLATE：空间证据 ——
    bbox: BBoxOut | None = None
    track_id: int | None = None
    plate_text: str | None = None
    plate_color: str | None = None


class SummaryDetail(BaseModel):          # 仅 SUMMARY：三段式结构化
    topic: str = ""
    segments: list[str] = []
    key_points: list[str] = []


class PlateTrackOut(BaseModel):          # 仅 PLATE：一条车牌/车辆轨迹（前端轨迹列表）
    track_id: int
    plate_text: str
    label: str = ""                      # 展示标签：车牌号，或车辆模式无牌时「车辆N」
    confidence: float
    plate_color: str | None = None
    first_t: float
    last_t: float
    hms_range: str = ""                  # "00:00:03–00:00:07"，前端直显
    n_frames: int = 0
    method: str = "vote"                 # vote|rule_fixed|superres|vlm_corrected
    best_frame_url: str = ""             # 代表帧（最清晰）缩略图 URL
    caveats: str = ""


class AnswerOut(BaseModel):
    intent: str
    target: str | None = None
    answer: str
    summary_detail: SummaryDetail | None = None   # 仅 SUMMARY
    evidence: list[EvidenceOut] = []
    confidence: float = 0.0
    caveats: str = ""
    video_url: str | None = None         # 供播放器跳转（PLATE 下为原视频，标注视频见下）
    query_image_url: str | None = None   # 仅 VISUAL_LOCATE：参考图回显
    grids: list[str] = []                # 拼图 URL
    # —— 仅 PLATE ——
    annotated_video_url: str | None = None        # 高亮标注视频（框跟随移动 + 车牌号）
    plate_tracks: list[PlateTrackOut] = []


class HealthOut(BaseModel):
    status: str = "ok"
    ffmpeg: bool = False
    whisper: bool = False
    version: str = ""
