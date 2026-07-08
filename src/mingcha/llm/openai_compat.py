"""OpenAICompatProvider —— POST /v1/chat/completions（GPT-5.5 / GLM-5.2 / 兼容网关）。
首期写好线格式，SUMMARY 默认走 Claude，不强制测试本 provider。

对应设计文档 §6.2。
"""
from __future__ import annotations

from .base import LLMProvider, LLMResult, LLMRefusal, encode_image, post_with_retry


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
        data = post_with_retry(f"{self.base_url}/v1/chat/completions", headers, body)
        choice = data["choices"][0]
        # §6.7 拒答映射：内容过滤 → 不重试，交上层转「建议人工复核」
        if choice.get("finish_reason") == "content_filter":
            raise LLMRefusal(f"{self.name}:{self.model} 拒答（finish_reason=content_filter）")
        text = choice["message"]["content"]
        return LLMResult(text=text, usage=data.get("usage", {}),
                         model=data.get("model", self.model), raw=data)
