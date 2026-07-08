"""LLM provider 抽象基类 + 统一请求/响应类型 + HTTP 重试。原生 HTTP，无官方 SDK。

对应设计文档 §6.1 / §6.7。
"""
from __future__ import annotations

import base64
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import requests

from ..config import HTTP_MAX_RETRIES, HTTP_TIMEOUT


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    """模型主动拒答（Anthropic stop_reason=='refusal' / OpenAI finish_reason=='content_filter'）。
    §6.7：统一映射为「无法判定，建议人工复核」，不重试原 prompt。"""


@dataclass
class ImageRef:
    path: str
    cacheable: bool = False       # 支持 caching 的 provider 会利用（如参考图固定前缀）


@dataclass
class Msg:
    role: str                     # "user" | "assistant"
    text: str = ""


@dataclass
class Capabilities:
    vision: bool = True
    json_schema: bool = False
    prompt_cache: bool = False
    batch: bool = False


@dataclass
class LLMResult:
    text: str                     # 原始文本（结构化时是 JSON 串）
    usage: dict = field(default_factory=dict)
    model: str = ""
    raw: dict = field(default_factory=dict)


def encode_image(path: str) -> str:
    """把本地图片读成 base64（各 provider 自行包装成线格式）。"""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def post_with_retry(url, headers, json_body, *, timeout=HTTP_TIMEOUT,
                    max_retries=HTTP_MAX_RETRIES) -> dict:
    """POST + 429/5xx 指数退避（读 Retry-After）。返回解析后的 JSON dict。"""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
        except requests.RequestException as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise LLMError(f"HTTP 请求失败（重试用尽）: {e}") from e
        if resp.status_code < 400:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            ra = resp.headers.get("Retry-After", "")
            delay = float(ra) if ra.replace(".", "", 1).isdigit() else min(2 ** attempt, 30)
            time.sleep(delay)
            continue
        raise LLMError(f"{url} 返回 {resp.status_code}: {resp.text[:500]}")
    raise LLMError(f"重试用尽: {last_err}")


class LLMProvider(ABC):
    """一个 provider = 一种线格式 + 一个鉴权 + 一组能力开关。"""

    def __init__(self, name: str, model: str, base_url: str,
                 api_key: str | None, caps: Capabilities):
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.caps = caps

    @abstractmethod
    def chat(self, *, system: str, messages: list[Msg], images=(),
             max_tokens: int = 4096, json_schema: dict | None = None) -> LLMResult:
        ...
