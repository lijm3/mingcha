"""FastAPI 应用入口（§4.4）：ffmpeg 注入 + 路由 + 产物服务 + SPA 托管。

生产模式单进程同时提供 API / SSE / 静态产物 / 前端页面，同源无跨域、无需 nginx。
"""
from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ① 启动时把 static-ffmpeg 的二进制注入 PATH，media.py 无需改动
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:  # noqa: BLE001 —— 未装则依赖系统 PATH 里的 ffmpeg
    pass

from . import settings
from .routers import meta as meta_router
from .routers import tasks as tasks_router
from .security import safe_join, token_matches
from .task_manager import TaskManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    mgr = TaskManager()
    mgr.bind_loop(asyncio.get_running_loop())
    app.state.task_manager = mgr
    cleaner = asyncio.create_task(_ttl_loop(mgr))
    try:
        yield
    finally:
        cleaner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleaner
        mgr.shutdown()


async def _ttl_loop(mgr: TaskManager):
    while True:
        try:
            mgr.cleanup_expired()
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(3600)


app = FastAPI(title="明察 MingCha", lifespan=lifespan)

# 本地开发：前端 dev server(5173) 跨域访问后端(8000)。生产同源不需要，但留着无害。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(meta_router.router)
app.include_router(tasks_router.router)


# ---- 产物静态服务（带 token 校验 + 路径穿越防护，§10.2）----
artifacts_router = APIRouter()


@artifacts_router.get("/artifacts/{task_id}/{path:path}")
def get_artifact(request: Request, task_id: str, path: str, token: str = Query(...)):
    mgr: TaskManager = request.app.state.task_manager
    rec = mgr.get(task_id)
    if rec is None:
        raise HTTPException(404, "任务不存在或已过期")
    if not token_matches(rec.token, token):
        raise HTTPException(403, "token 无效")
    target = safe_join(rec.out_dir, path)
    if target is None or not target.is_file():
        raise HTTPException(404, "产物不存在")
    return FileResponse(target)


app.include_router(artifacts_router)


# ---- SPA 托管（放最后：其余路径交给前端 index.html）----
if settings.FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(settings.FRONTEND_DIST), html=True),
              name="spa")
