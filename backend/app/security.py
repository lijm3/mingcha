"""任务 token 与产物路径穿越防护（§10.2 / §11.1）。"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path


def new_token() -> str:
    """给每个任务生成随机访问凭证（访问产物/状态/SSE 都需带它）。"""
    return secrets.token_urlsafe(24)


def token_matches(expected: str | None, provided: str | None) -> bool:
    """恒定时间比较，避免时序侧信道。"""
    if not expected or not provided:
        return False
    return hmac.compare_digest(expected, provided)


def token_hash(token: str) -> str:
    """meta.json 里只存 token 的哈希，不存明文（即便产物泄露也拿不到 token）。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def safe_join(base: Path, relative: str) -> Path | None:
    """把 relative 拼到 base 下并做穿越防护：结果必须仍在 base 内，否则返回 None。"""
    base = base.resolve()
    try:
        target = (base / relative).resolve()
    except (ValueError, OSError):
        return None
    if base == target or base in target.parents:
        return target
    return None
