"""任务管理器（§6.2）：内存任务表 + ProcessPoolExecutor + 每任务事件队列。

设计要点：
- 分析是 CPU 密集 + 阻塞（ffmpeg/whisper/多次 LLM），放**独立进程**执行，进程隔离
  才能真正并发、单任务崩溃不拖垮服务。
- worker 进程通过 multiprocessing.Manager().Queue() 把进度事件回传主进程；
  主进程每任务起一个 asyncio drain 协程，把事件翻译成状态更新 + 广播给 SSE 订阅者。
- 取消：设置 mp Event，内核在关键检查点抛 TaskCancelled。
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import settings
from .security import new_token, token_hash

import logging
_log = logging.getLogger("mingcha.backend")
if not _log.handlers:
    import sys
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False


# ---- 进程池 worker 入口（必须是模块级函数，才能被 spawn 序列化）----
def _worker_entry(payload: dict, event_queue, cancel_event) -> dict:
    from .service import run_task
    return run_task(
        payload,
        emit=lambda ev: event_queue.put(ev),
        cancel=lambda: cancel_event.is_set(),
    )


@dataclass
class TaskRecord:
    id: str
    token: str
    token_hash: str
    out_dir: Path
    state: str = "queued"
    progress: float = 0.0
    stage_note: str = ""
    intent: str | None = None
    error: str | None = None
    caveats: str = ""
    created_at: float = 0.0
    has_upload: bool = False              # 上传源 vs URL 源（前端两段式进度用不到，仅记录）
    event_queue: object = None            # mp Manager Queue
    cancel_event: object = None           # mp Event
    future: object = None
    last_event: dict = field(default_factory=dict)   # 最近状态快照，供 SSE 重连续传
    subscribers: list = field(default_factory=list)  # list[asyncio.Queue]
    done: bool = False

    def status_snapshot(self) -> dict:
        return {
            "task_id": self.id, "state": self.state, "progress": self.progress,
            "stage_note": self.stage_note, "intent": self.intent,
            "error": self.error, "created_at": self.created_at, "caveats": self.caveats,
        }


# 状态 → 进度百分比（§6.3 经验值），worker 未显式给 progress 时兜底
_STATE_PROGRESS = {
    "queued": 0.0, "downloading": 0.1, "extracting": 0.4, "transcribing": 0.55,
    "analyzing": 0.75, "assembling": 0.95, "done": 1.0,
}


class TaskManager:
    def __init__(self) -> None:
        ctx = mp.get_context("spawn")     # 跨平台一致（Windows 只有 spawn）
        self._pool = ProcessPoolExecutor(max_workers=settings.MAX_CONCURRENCY,
                                         mp_context=ctx)
        self._mgr = ctx.Manager()
        self._tasks: dict[str, TaskRecord] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ---- 创建任务 ----
    def create_record(self, *, has_upload: bool) -> TaskRecord:
        """只分配 task_id / token / 工作目录，尚不提交 worker。
        上传源需要先把文件落到 out_dir，再 start()。"""
        import uuid
        task_id = uuid.uuid4().hex[:16]
        token = new_token()
        out_dir = settings.TASKS_DIR / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = TaskRecord(
            id=task_id, token=token, token_hash=token_hash(token), out_dir=out_dir,
            created_at=time.time(), has_upload=has_upload,
            event_queue=self._mgr.Queue(), cancel_event=self._mgr.Event(),
        )
        self._tasks[task_id] = rec
        return rec

    def start(self, rec: TaskRecord, payload: dict) -> None:
        """提交 worker 进程并起 drain 协程。payload.source 此时必须已就绪。"""
        payload = dict(payload, out_dir=str(rec.out_dir))
        rec.future = self._pool.submit(
            _worker_entry, payload, rec.event_queue, rec.cancel_event)
        assert self._loop is not None
        self._loop.create_task(self._drain(rec))

    def create(self, payload: dict, *, has_upload: bool) -> TaskRecord:
        """便捷：URL 源可一步创建并启动。"""
        rec = self.create_record(has_upload=has_upload)
        self.start(rec, payload)
        return rec

    def get(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def cancel(self, rec: TaskRecord) -> bool:
        if rec.done:
            return False
        rec.cancel_event.set()
        rec.future.cancel()   # 若还在队列里没开始，直接取消
        return True

    # ---- SSE 订阅 ----
    def subscribe(self, rec: TaskRecord) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        rec.subscribers.append(q)
        # 立即补发当前快照，让刚连上的客户端不必等下一个事件（也用于重连续传，§5.4）
        q.put_nowait({"type": "state", **rec.status_snapshot()})
        if rec.done:
            q.put_nowait(rec.last_event or {"type": "done"})
        return q

    def unsubscribe(self, rec: TaskRecord, q: asyncio.Queue) -> None:
        try:
            rec.subscribers.remove(q)
        except ValueError:
            pass

    def _broadcast(self, rec: TaskRecord, event: dict) -> None:
        rec.last_event = event
        for q in list(rec.subscribers):
            q.put_nowait(event)

    # ---- 事件泵：mp Queue → 状态更新 + 广播 ----
    async def _drain(self, rec: TaskRecord) -> None:
        loop = asyncio.get_running_loop()
        timeout_at = rec.created_at + settings.TASK_TIMEOUT_MIN * 60
        while True:
            if time.time() > timeout_at and not rec.done:
                self._terminal(rec, "error", error="任务超时（超过 "
                               f"{settings.TASK_TIMEOUT_MIN} 分钟），已中止。")
                rec.cancel_event.set()
                break
            try:
                ev = await loop.run_in_executor(None, _q_get, rec.event_queue, 0.5)
            except _Empty:
                if rec.future.done() and not rec.done:
                    # 进程结束但没收到终止事件（如崩溃）→ 兜底转 error
                    self._terminal(rec, "error", error="分析进程异常退出。")
                    break
                continue
            self._apply(rec, ev)
            if rec.done:
                break

    def _apply(self, rec: TaskRecord, ev: dict) -> None:
        etype = ev.get("type")
        if etype == "state":
            rec.state = ev.get("state", rec.state)
            rec.progress = ev.get("progress",
                                  _STATE_PROGRESS.get(rec.state, rec.progress))
            rec.stage_note = ev.get("stage_note", "")
            self._broadcast(rec, {"type": "state", **rec.status_snapshot()})
        elif etype == "log":
            line = ev.get("line", "")
            _log.info("[task %s] %s", rec.id[:8], line)   # 回到 uvicorn 控制台
            self._broadcast(rec, {"type": "log", "line": line})
        elif etype == "done":
            rec.intent = ev.get("intent")
            rec.caveats = ev.get("caveats", "")
            self._terminal(rec, "done",
                           answer_url=f"/api/tasks/{rec.id}/answer?token={rec.token}")
        elif etype == "error":
            self._terminal(rec, "error", error=ev.get("error", "未知错误"))
        elif etype == "cancelled":
            self._terminal(rec, "cancelled")

    def _terminal(self, rec: TaskRecord, state: str, *, error=None, answer_url=None) -> None:
        rec.state = state
        rec.done = True
        rec.error = error
        rec.progress = 1.0 if state == "done" else rec.progress
        if state == "done":
            event = {"type": "done", "answer_url": answer_url}
        elif state == "cancelled":
            event = {"type": "fail", "error": "任务已取消", "state": "cancelled"}
        else:
            event = {"type": "fail", "error": error or "任务失败", "state": "error"}
        self._broadcast(rec, event)

    # ---- TTL 清理（§6.7）----
    def cleanup_expired(self) -> int:
        import shutil
        cutoff = time.time() - settings.TASK_TTL_HOURS * 3600
        removed = 0
        for tid, rec in list(self._tasks.items()):
            if rec.created_at < cutoff:
                shutil.rmtree(rec.out_dir, ignore_errors=True)
                self._tasks.pop(tid, None)
                removed += 1
        # 也扫磁盘上没有内存记录的过期目录（进程重启后残留）
        if settings.TASKS_DIR.exists():
            for d in settings.TASKS_DIR.iterdir():
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
        return removed

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        try:
            self._mgr.shutdown()
        except Exception:  # noqa: BLE001
            pass


# multiprocessing Queue 没有 asyncio 接口，用线程池阻塞式 get + 超时
import queue as _queue  # noqa: E402

_Empty = _queue.Empty


def _q_get(q, timeout: float):
    return q.get(timeout=timeout)
