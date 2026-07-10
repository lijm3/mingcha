"""OpenAICompatProvider —— POST /v1/chat/completions（GPT-5.5 / GLM-5.2 / 兼容网关）。
首期写好线格式，SUMMARY 默认走 Claude，不强制测试本 provider。

对应设计文档 §6.2。
"""
from __future__ import annotations

from .base import LLMProvider, LLMResult, LLMRefusal, encode_image, post_with_retry
from .._log import get_logger

log = get_logger("mingcha.llm")


class OpenAICompatProvider(LLMProvider):
    def chat(self, *, system, messages, images=(), max_tokens=4096,
             json_schema=None) -> LLMResult:
        headers = {"Authorization": f"Bearer {self.api_key or ''}",
                   "content-type": "application/json"}
        content: list[dict] = []
        for img in images:
            b64 = encode_image(img.path)
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        instruction = "\n".join(m.text for m in messages if m.role == "user")
        content.append({"type": "text", "text": instruction})

        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": content})

        body: dict = {"model": self.model, "messages": msgs, "max_tokens": max_tokens}
        if json_schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "structured_output",
                                                       "schema": json_schema}}
        log.debug("openai_compat 调用 model=%s images=%d schema=%s max_tokens=%d",
                  self.model, len(images), bool(json_schema), max_tokens)
        data = post_with_retry(f"{self.base_url}/v1/chat/completions", headers, body,
                               label=f"{self.name}:{self.model}")
        choice = data["choices"][0]
        # §6.7 拒答映射：内容过滤 → 不重试，交上层转「建议人工复核」
        if choice.get("finish_reason") == "content_filter":
            log.warning("openai_compat %s 拒答（finish_reason=content_filter）", self.model)
            raise LLMRefusal(f"{self.name}:{self.model} 拒答（finish_reason=content_filter）")
        text = choice["message"]["content"]
        return LLMResult(text=text, usage=data.get("usage", {}),
                         model=data.get("model", self.model), raw=data)
