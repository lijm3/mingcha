"""用户自带 Key 的解析与脱敏（§7）。

Key 只经请求头 X-Provider-Keys 传入（JSON），绝不进 URL/query/日志/磁盘。
"""
from __future__ import annotations

import json


def parse_provider_keys(header_value: str | None) -> dict[str, str]:
    """解析 X-Provider-Keys 请求头（形如 {"claude":"sk-...","openai":"..."}）。
    非法/空 → 返回空 dict（退回后端环境变量 key，§7.2）。"""
    if not header_value:
        return {}
    try:
        data = json.loads(header_value)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str) and v.strip():
            out[k.strip()] = v.strip()
    return out


def redact(text: str) -> str:
    """日志脱敏：把疑似 key 的片段替换为 sk-****（§7.1）。"""
    import re
    return re.sub(r"(sk-[A-Za-z0-9_\-]{4,}|Bearer\s+[A-Za-z0-9_\-.]{8,})", "sk-****", text)
