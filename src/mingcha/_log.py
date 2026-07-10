"""统一日志。用标准 logging，默认输出到 stderr（CLI 与后端进程池 worker 都能看到）。

级别用环境变量控制：MINGCHA_LOG_LEVEL=DEBUG|INFO|WARNING（默认 INFO）。
DEBUG 会打印每次 HTTP 请求体大小/耗时等细节，排查模型调用问题时用。
"""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _ensure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    level = os.environ.get("MINGCHA_LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger("mingcha")
    logger.setLevel(getattr(logging, level, logging.INFO))
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    logger.propagate = False   # 不重复冒泡到 root（避免后端 uvicorn 双打）


def get_logger(name: str = "mingcha") -> logging.Logger:
    _ensure()
    return logging.getLogger(name)
