"""任务路由（§5）：创建（上传/URL）+ 状态 + SSE 进度 + 结果 + 取消。"""
from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, Header, HTTPException, Query, Request, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse

from .. import settings
from ..answer_out import build_answer_out
from ..keys import parse_provider_keys
from ..schemas import CreateTaskResponse, TaskStatus
from ..security import token_matches

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def _mgr(request: Request):
    return request.app.state.task_manager


def _auth(request: Request, task_id: str, token: str):
    rec = _mgr(request).get(task_id)
    if rec is None:
        raise HTTPException(404, "任务不存在或已过期")
    if not token_matches(rec.token, token):
        raise HTTPException(403, "token 无效")
    return rec


# ---- 创建任务：URL 源（JSON）或上传源（multipart）----
@router.post("", status_code=202, response_model=CreateTaskResponse)
async def create_task(
    request: Request,
    x_provider_keys: str | None = Header(default=None, alias="X-Provider-Keys"),
    # multipart 字段（上传源时用；URL 源走 JSON body，这些为 None）
    video: UploadFile | None = None,
    image: UploadFile | None = None,
    prompt: str | None = Form(default=None),
    intent: str | None = Form(default=None),
    provider: str | None = Form(default=None),
    vision_model: str | None = Form(default=None),
    classify_model: str | None = Form(default=None),
):
    keys = parse_provider_keys(x_provider_keys)
    mgr = _mgr(request)
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        # URL 源
        body = await request.json()
        source = (body.get("url") or "").strip()
        if not source:
            raise HTTPException(400, "缺少视频 URL")
        payload = {
            "source": source, "prompt": body.get("prompt", ""),
            "image": None,
            "override": body.get("override") or {},
            "keys": keys,
            "cookies_from_browser": body.get("cookies_from_browser"),
            "use_cache": False,
        }
        rec = mgr.create(payload, has_upload=False)
        return CreateTaskResponse(task_id=rec.id, task_token=rec.token)

    # multipart 上传源
    if video is None:
        raise HTTPException(400, "缺少视频文件（video 字段）")
    override = {"provider": provider, "vision_model": vision_model,
                "classify_model": classify_model}
    # 先建记录+目录，落地上传文件，再提交 worker（保证 worker 拿到真实源路径）
    rec = mgr.create_record(has_upload=True)
    try:
        video_path = await _save_upload(video, rec.out_dir, settings.VIDEO_EXTS,
                                        "source", rec)
        image_path = None
        if image is not None:
            image_path = await _save_upload(image, rec.out_dir, settings.IMAGE_EXTS,
                                            "query_image", rec)
    except HTTPException:
        import shutil
        shutil.rmtree(rec.out_dir, ignore_errors=True)
        mgr._tasks.pop(rec.id, None)
        raise

    mgr.start(rec, {
        "source": str(video_path), "prompt": prompt or "",
        "image": str(image_path) if image_path else None,
        "override": override, "keys": keys, "use_cache": False,
    })
    return CreateTaskResponse(task_id=rec.id, task_token=rec.token)


async def _save_upload(upload: UploadFile, out_dir, allowed_exts, stem: str, rec):
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(400, f"不支持的文件类型: {ext or '(无扩展名)'}")
    dest = out_dir / f"{stem}{ext}"
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    with open(dest, "wb") as fh:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"文件超过 {settings.MAX_UPLOAD_MB}MB 上限")
            fh.write(chunk)
    return dest


@router.get("/{task_id}", response_model=TaskStatus)
def get_status(request: Request, task_id: str, token: str = Query(...)):
    rec = _auth(request, task_id, token)
    return TaskStatus(**rec.status_snapshot())


@router.get("/{task_id}/events")
async def events(request: Request, task_id: str, token: str = Query(...)):
    rec = _auth(request, task_id, token)
    mgr = _mgr(request)
    q = mgr.subscribe(rec)

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"      # 注释行，保活
                    continue
                etype = ev.pop("type", "state")
                yield f"event: {etype}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
                if etype in ("done", "fail"):
                    break
        finally:
            mgr.unsubscribe(rec, q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@router.get("/{task_id}/answer")
def get_answer(request: Request, task_id: str, token: str = Query(...)):
    rec = _auth(request, task_id, token)
    if not rec.done or rec.state != "done":
        raise HTTPException(409, "任务尚未完成")
    out = build_answer_out(rec.out_dir, rec.id, rec.token)
    if out is None:
        raise HTTPException(404, "结果文件缺失")
    return JSONResponse(out.model_dump())


@router.post("/{task_id}/cancel")
def cancel_task(request: Request, task_id: str, token: str = Query(...)):
    rec = _auth(request, task_id, token)
    ok = _mgr(request).cancel(rec)
    return {"cancelled": ok}
