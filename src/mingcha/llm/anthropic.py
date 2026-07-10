"""AnthropicProvider —— POST /v1/messages 线格式。结构化输出走 tool_use 强制。
支持 prompt caching（§6.5）与拒答映射（§6.7）。

对应设计文档 §6.2。
"""
from __future__ import annotations

import json

from .base import (
    LLMProvider, LLMResult, LLMRefusal, encode_image, post_with_retry,
)
from .._log import get_logger

_CACHE_CONTROL = {"type": "ephemeral"}

log = get_logger("mingcha.llm")


class AnthropicProvider(LLMProvider):
    def chat(self, *, system, messages, images=(), max_tokens=4096,
             json_schema=None) -> LLMResult:
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        use_cache = getattr(self.caps, "prompt_cache", False)
        content: list[dict] = []
        for img in images:
            block = {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": encode_image(img.path)},
            }
            # §6.5：cacheable 的固定前缀（如参考图）打 cache_control，后续调用只付变化部分
            if use_cache and getattr(img, "cacheable", False):
                block["cache_control"] = _CACHE_CONTROL
            content.append(block)
        instruction = "\n".join(m.text for m in messages if m.role == "user")
        content.append({"type": "text", "text": instruction})

        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            sys_block = {"type": "text", "text": system}
            if use_cache:
                sys_block["cache_control"] = _CACHE_CONTROL  # system 固定 → 缓存
            body["system"] = [sys_block]
        if json_schema:
            # tool_use 强制结构化：schema 作为 input_schema，tool_choice 锁定该工具
            body["tools"] = [{"name": "structured_output",
                              "description": "以结构化 JSON 返回结果",
                              "input_schema": json_schema}]
            body["tool_choice"] = {"type": "tool", "name": "structured_output"}

        log.debug("anthropic 调用 model=%s images=%d schema=%s max_tokens=%d",
                  self.model, len(list(images)), bool(json_schema), max_tokens)
        data = post_with_retry(f"{self.base_url}/v1/messages", headers, body,
                               label=f"{self.name}:{self.model}")

        # §6.7 拒答映射：不重试原 prompt，交上层转「建议人工复核」
        if data.get("stop_reason") == "refusal":
            log.warning("anthropic %s 拒答（stop_reason=refusal）", self.model)
            raise LLMRefusal(f"{self.name}:{self.model} 拒答（stop_reason=refusal）")

        if json_schema:
            for b in data.get("content", []):
                if b.get("type") == "tool_use":
                    return LLMResult(text=json.dumps(b.get("input", {}), ensure_ascii=False),
                                     usage=data.get("usage", {}),
                                     model=data.get("model", self.model), raw=data)
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        return LLMResult(text=text, usage=data.get("usage", {}),
                         model=data.get("model", self.model), raw=data)
