"""API 请求/响应 Pydantic 模型（§5.3）。复用内核 types.Answer/Evidence，
在 API 层包一层产物 URL（*_url 由 task_token 拼装，不落进 answer.json）。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

TaskState = Literal["queued", "downloading", "extracting", "transcribing",
                    "analyzing", "assembling", "done", "error", "cancelled"]
IntentName = Literal["auto", "SUMMARY", "LOCATE", "MODERATE", "VISUAL_LOCATE"]


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


class EvidenceOut(BaseModel):
    frame: str
    frame_url: str                       # /artifacts/{id}/frames/frame_007.jpg?token=
    t: float
    hms: str
    confidence: float
    similarity: float | None = None      # 仅 VISUAL_LOCATE
    verdict: str | None = None           # 仅 VISUAL_LOCATE: "same"|"similar"
    note: str = ""


class SummaryDetail(BaseModel):          # 仅 SUMMARY：三段式结构化
    topic: str = ""
    segments: list[str] = []
    key_points: list[str] = []


class AnswerOut(BaseModel):
    intent: str
    target: str | None = None
    answer: str
    summary_detail: SummaryDetail | None = None   # 仅 SUMMARY
    evidence: list[EvidenceOut] = []
    confidence: float = 0.0
    caveats: str = ""
    video_url: str | None = None         # 供播放器跳转
    query_image_url: str | None = None   # 仅 VISUAL_LOCATE：参考图回显
    grids: list[str] = []                # 拼图 URL


class HealthOut(BaseModel):
    status: str = "ok"
    ffmpeg: bool = False
    whisper: bool = False
    version: str = ""
