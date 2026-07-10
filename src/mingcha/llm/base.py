"""LLM provider 抽象基类 + 统一请求/响应类型 + HTTP 重试。**原生 HTTP，不绑定任何官方 SDK**。

本模块是 llm/ 适配层的地基，定义三样东西供 anthropic.py / openai_compat.py 复用：
  · 异常     —— LLMError（通用）/ LLMRefusal（模型拒答，需特殊对待）；
  · 数据类   —— ImageRef / Msg / Capabilities / LLMResult，跨 provider 中立的请求/响应类型；
  · 公共设施 —— encode_image（图片转 base64）、post_with_retry（带退避的 POST）、LLMProvider（ABC）。

上层从不直接 new provider，而是经 llm.get_provider 工厂按 config 注册表拿到 LLMProvider 实例。
对应设计文档 §6.1 / §6.7。
"""
from __future__ import annotations

import base64
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import requests

from ..config import HTTP_MAX_RETRIES, HTTP_TIMEOUT
from .._log import get_logger

log = get_logger("mingcha.llm")


class LLMError(RuntimeError):
    """本适配层所有可预期失败的统一基类：HTTP 重试用尽、非 2xx 响应、结构化解析失败等。
    上层用它一处兜底，无需区分具体 provider 的异常类型。"""


class LLMRefusal(LLMError):
    """模型主动拒答（Anthropic stop_reason=='refusal' / OpenAI finish_reason=='content_filter'）。
    §6.7：与普通错误区别对待——统一映射为「无法判定，建议人工复核」，且**不重试原 prompt**
    （重试同样内容只会再次被拒）。逐帧判定中单帧拒答被 batch 层捕获并标注，不拖垮整批。"""


@dataclass
class ImageRef:
    """一张待发送图片的中立引用（provider 无关）。各 provider 的 chat() 再据此读文件、
    base64 编码、包装成自家线格式（Anthropic 的 image source / OpenAI 的 data URL）。"""
    path: str
    cacheable: bool = False       # 标记可缓存：支持 prompt caching 的 provider 会在此打缓存断点
    #                               （如 VISUAL_LOCATE 里多次比对中固定不变的参考图前缀，§6.5），其它家忽略。


@dataclass
class Msg:
    """一轮对话消息（纯文本；图片不走这里，单独经 chat() 的 images 通道传入）。"""
    role: str                     # "user" | "assistant"
    text: str = ""


@dataclass
class Capabilities:
    """一家 provider 的能力开关，来自 config.PROVIDERS[...]['caps']。上层据此择路
    （如 json_schema 决定走原生结构化还是内联 schema 兜底，见 structured.py）。"""
    vision: bool = True           # 是否支持图像输入
    json_schema: bool = False     # 是否支持原生结构化输出（Anthropic tool_use / OpenAI response_format）
    prompt_cache: bool = False    # 是否支持 prompt caching（目前仅 claude）
    batch: bool = False           # 是否支持批处理 API；当前逐帧统一走同步并发（batch.py），此位暂未被消费


@dataclass
class LLMResult:
    """一次 chat() 调用的统一返回，屏蔽各家响应结构差异，供 structured.py / __init__.py 消费。"""
    text: str                     # 抽取出的文本内容（结构化调用时即那段 JSON 串）
    usage: dict = field(default_factory=dict)  # token 用量原样透传（字段名各家不同，_log_usage 做兼容）
    model: str = ""               # 响应回报的模型名（provider 填，可能与请求的略有出入）
    raw: dict = field(default_factory=dict)    # 原始响应 JSON，留作调试 / 兜底二次解析


def encode_image(path: str) -> str:
    """把本地图片读成 base64 ascii 串（各 provider 再自行包装成线格式）。"""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def post_with_retry(url, headers, json_body, *, timeout=HTTP_TIMEOUT,
                    max_retries=HTTP_MAX_RETRIES, label="") -> dict:
    """POST 一个 JSON 请求，对**瞬时性失败**自动退避重试，返回解析后的 JSON dict。

    只重试"重试可能有用"的失败：
      · 网络层异常（连接超时 / 断连 / DNS 等 requests.RequestException）；
      · HTTP 429（限流）与 5xx（500/502/503/504 服务端错误）。
    **不重试**其它 4xx（400/401/403 等客户端错误——参数或鉴权问题，重试也白搭），直接抛 LLMError。

    退避：优先采纳响应的 Retry-After 头（仅识别"秒数"数字格式；HTTP 日期格式不解析、退回指数退避），
    否则指数退避 min(2**attempt, 30) 秒封顶。共尝试 max_retries+1 次（默认 5 次 = 1 初次 + 4 重试），
    用尽仍失败则抛 LLMError。label 仅作日志前缀（如 'claude:claude-opus-4-8'），便于并发时区分来源。
    """
    tag = f"[{label}] " if label else ""
    last_err = None
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
        except requests.RequestException as e:
            # —— 网络层异常：仍有重试机会就指数退避，否则包装成 LLMError 抛出 ——
            last_err = e
            dt = time.time() - t0
            if attempt < max_retries:
                delay = min(2 ** attempt, 30)
                log.warning("%s请求异常（第 %d 次，%.1fs）: %s —— %.0fs 后重试",
                            tag, attempt + 1, dt, e, delay)
                time.sleep(delay)
                continue
            log.error("%s请求失败（重试用尽）: %s", tag, e)
            raise LLMError(f"HTTP 请求失败（重试用尽）: {e}") from e
        dt = time.time() - t0
        if resp.status_code < 400:
            # —— 成功（2xx/3xx）：解析 JSON 返回 ——
            log.info("%s%s → %d（%.1fs）", tag, url, resp.status_code, dt)
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            # —— 限流 / 服务端错误且仍有重试机会：Retry-After 是纯数字秒则采纳，否则指数退避 ——
            ra = resp.headers.get("Retry-After", "")
            delay = float(ra) if ra.replace(".", "", 1).isdigit() else min(2 ** attempt, 30)
            log.warning("%s%s → %d（第 %d 次，%.1fs）—— %.0fs 后重试",
                        tag, url, resp.status_code, attempt + 1, dt, delay)
            time.sleep(delay)
            continue
        # —— 不可重试的 4xx，或 429/5xx 但重试已用尽：如实抛错（截断响应体避免刷屏日志）——
        log.error("%s%s → %d: %s", tag, url, resp.status_code, resp.text[:300])
        raise LLMError(f"{url} 返回 {resp.status_code}: {resp.text[:500]}")
    raise LLMError(f"重试用尽: {last_err}")  # 理论不可达：末次迭代各分支必 return/raise，此行仅兜底防御


class LLMProvider(ABC):
    """provider 抽象基类：一个 provider = 一种线格式 + 一套鉴权 + 一组能力开关。
    子类只需实现 chat()，把中立请求翻译成自家线格式、再把响应翻译回 LLMResult。
    实例由 llm.get_provider 工厂按 config 注册表构造，上层不直接 new。"""

    def __init__(self, name: str, model: str, base_url: str,
                 api_key: str | None, caps: Capabilities):
        self.name = name                      # provider 注册名（'claude' / 'openai' / 'glm'）
        self.model = model                    # 具体模型 id（如 'claude-opus-4-8'）
        self.base_url = base_url.rstrip("/")  # API 根地址；去尾斜杠，拼接 /v1/... 时避免双斜杠
        self.api_key = api_key                # 可能为 None（缺 key）；发请求前由门面层校验并报错
        self.caps = caps                      # 能力开关，决定走哪条结构化 / 缓存路径

    @abstractmethod
    def chat(self, *, system: str, messages: list[Msg], images=(),
             max_tokens: int = 4096, json_schema: dict | None = None) -> LLMResult:
        """发一次对话请求并返回统一 LLMResult。子类实现（anthropic.py / openai_compat.py）：
        组线格式 → post_with_retry → 抽取文本 / 用量 → 包成 LLMResult。
        json_schema 非空时应走原生结构化输出（provider 支持的话）。"""
        ...
