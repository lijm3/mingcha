"""桥接层（§9.8）：在进程池 worker 中执行，调用改造后的 Orchestrator.ask()，
把内核进度/取消/结果经 multiprocessing 事件回传主进程。

注意：本函数运行在**独立进程**里，emit 通过 multiprocessing 管道回传。
"""
from __future__ import annotations

import logging
import os
from typing import Any

# 进程池 worker 的日志无法直接回到主控制台（Windows spawn 子进程 stderr 不共享），
# 故用一个 handler 把 mingcha 的日志记录经 emit 事件管道回传主进程 → 转 SSE「log」事件，
# 既能在后端控制台看到，也能在浏览器日志面板实时看到。
_CURRENT_EMIT = None
_BRIDGE_INSTALLED = False


class _QueueLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        fn = _CURRENT_EMIT
        if fn is None:
            return
        try:
            fn({"type": "log", "line": self.format(record)})
        except Exception:  # noqa: BLE001 —— 日志转发绝不能反过来搞崩任务
            pass


def _install_log_bridge() -> None:
    global _BRIDGE_INSTALLED
    if _BRIDGE_INSTALLED:
        return
    _BRIDGE_INSTALLED = True
    from mingcha._log import get_logger
    logger = get_logger("mingcha")          # 触发一次基础配置
    h = _QueueLogHandler()
    h.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    level = os.environ.get("MINGCHA_LOG_LEVEL", "INFO").upper()
    h.setLevel(getattr(logging, level, logging.INFO))
    logging.getLogger("mingcha").addHandler(h)


def run_task(payload: dict, emit, cancel) -> dict:
    """进程池 worker 入口。

    payload: {source, prompt, image, out_dir, override, keys, cookies_from_browser, use_cache}
    emit:    Callable[[dict], None]  —— 把事件送回主进程队列
    cancel:  Callable[[], bool]      —— 返回 True 表示已请求取消
    """
    global _CURRENT_EMIT
    from mingcha.orchestrator import Orchestrator, TaskCancelled

    _install_log_bridge()
    _CURRENT_EMIT = emit                    # 让日志桥接把本任务的日志转发到该 emit

    override = payload.get("override") or {}

    def progress(state: str, frac: float, note: str = "") -> None:
        emit({"type": "state", "state": state, "progress": frac, "stage_note": note})

    try:
        orch = Orchestrator(
            provider=override.get("provider"),
            vision_model=override.get("vision_model"),
            classify_model=override.get("classify_model"),
            runtime_keys=payload.get("keys") or None,
        )
        ans = orch.ask(
            payload["source"], payload.get("prompt", ""),
            query_image=payload.get("image"),
            out_dir=payload["out_dir"],
            cookies_from_browser=payload.get("cookies_from_browser"),
            use_cache=payload.get("use_cache", False),
            progress=progress,
            cancel=cancel,
        )
        emit({"type": "done", "intent": ans.intent, "caveats": ans.caveats})
        return {"ok": True}
    except TaskCancelled:
        emit({"type": "cancelled"})
        return {"ok": False, "cancelled": True}
    except Exception as e:  # noqa: BLE001 —— 任何内核异常都转成可读错误事件，不崩服务
        emit({"type": "error", "error": _friendly_error(e)})
        return {"ok": False}
    finally:
        _CURRENT_EMIT = None


def _friendly_error(e: Exception) -> Any:
    """把内核异常映射为人类可读文案（§5.5）。"""
    name = type(e).__name__
    msg = str(e)
    if name == "LLMError":
        return f"模型调用失败：{msg}（请检查对应 provider 的 Key 是否正确/有余额）"
    if name == "FileNotFoundError":
        return f"文件缺失：{msg}"
    if "ffmpeg" in msg.lower() or "ffprobe" in msg.lower():
        return f"ffmpeg 相关错误：{msg}（请确认已装 ffmpeg 或 static-ffmpeg）"
    if "download" in msg.lower() or "yt-dlp" in msg.lower():
        return f"下载失败：{msg}（私有视频可能需要 cookies）"
    return f"{name}: {msg}"
