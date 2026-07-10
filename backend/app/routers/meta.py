"""能力探测 / 健康检查（§5.1）。"""
from __future__ import annotations

import shutil

from fastapi import APIRouter

from ..schemas import HealthOut

router = APIRouter(prefix="/api", tags=["meta"])


def _has_whisper() -> bool:
    try:
        import whisper  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    from .. import __version__
    return HealthOut(
        status="ok",
        ffmpeg=bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
        whisper=_has_whisper(),
        version=__version__,
    )
