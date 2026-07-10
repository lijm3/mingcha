"""后端环境变量配置（§13.1）。全部有默认值，不设也能跑。"""
from __future__ import annotations

import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


DATA_DIR: Path = Path(os.environ.get("MINGCHA_DATA_DIR", "./data")).resolve()
TASKS_DIR: Path = DATA_DIR / "tasks"

MAX_CONCURRENCY: int = _int("MINGCHA_MAX_CONCURRENCY", 2)
TASK_TTL_HOURS: int = _int("MINGCHA_TASK_TTL_HOURS", 24)
MAX_UPLOAD_MB: int = _int("MINGCHA_MAX_UPLOAD_MB", 500)
TASK_TIMEOUT_MIN: int = _int("MINGCHA_TASK_TIMEOUT_MIN", 30)

# 前端构建产物目录（生产由后端托管）
FRONTEND_DIST: Path = Path(
    os.environ.get("MINGCHA_FRONTEND_DIST",
                   str(Path(__file__).resolve().parents[2] / "frontend" / "dist"))
).resolve()

# 允许的视频扩展名（上传校验，§11.1）
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv", ".m4v", ".ts", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def ensure_dirs() -> None:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
