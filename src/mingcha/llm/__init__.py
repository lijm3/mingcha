"""LLM 适配层门面：get_provider(role) 工厂 + vision_structured / text_structured。
上层 analyzer 只依赖这三个，不知背后是 Claude 还是 GPT/GLM。

对应设计文档 §6.1。
"""
from __future__ import annotations

from ..config import PROVIDERS, api_key, resolve_role
from .anthropic import AnthropicProvider
from .base import (
    Capabilities, ImageRef, LLMError, LLMProvider, LLMRefusal, LLMResult, Msg,
)
from .batch import judge_frames_batch
from .openai_compat import OpenAICompatProvider
from .structured import structured

_KIND = {"anthropic": AnthropicProvider, "openai_compat": OpenAICompatProvider}

__all__ = ["get_provider", "vision_structured", "text_structured",
           "judge_frames", "describe",
           "ImageRef", "LLMProvider", "LLMError", "LLMRefusal"]


def get_provider(role: str, override: str | None = None) -> LLMProvider:
    """role ∈ {'vision','classify'}；从 config 注册表解析出具体 provider+model。"""
    prov_name, model = resolve_role(role, override)
    spec = PROVIDERS.get(prov_name)
    if not spec:
        raise ValueError(f"未知 provider: {prov_name}（可选: {list(PROVIDERS)}）")
    caps = Capabilities(**spec["caps"])
    cls = _KIND[spec["kind"]]
    return cls(prov_name, model, spec["base_url"], api_key(prov_name), caps)


def _log_usage(provider: LLMProvider, result: LLMResult) -> None:
    u = result.usage or {}
    it = u.get("input_tokens", u.get("prompt_tokens"))
    ot = u.get("output_tokens", u.get("completion_tokens"))
    print(f"  [llm] {provider.name}:{provider.model}  in={it} out={ot}")


def vision_structured(role, system, images, instruction, schema, *,
                      override=None, cacheable_images=()):
    """门面：选 provider → 组装请求 → 结构化输出 → Pydantic 校验。analyzer 只调这个。
    cacheable_images 中的路径会标记 cacheable=True（支持 caching 的 provider 自动利用，§6.5）。"""
    provider = get_provider(role, override)
    if not provider.api_key:
        raise LLMError(f"provider {provider.name} 缺少 API key："
                       f"请设置环境变量 {PROVIDERS[provider.name]['api_key_env']}")
    cset = {str(c) for c in cacheable_images}
    imgs = []
    for i in images:
        if isinstance(i, ImageRef):
            imgs.append(i)
        else:
            s = str(i)
            imgs.append(ImageRef(s, cacheable=s in cset))
    obj, result = structured(provider, system, imgs, instruction, schema)
    _log_usage(provider, result)
    return obj


def judge_frames(role, system, frames_with_time, schema, *,
                 override=None, instruction=""):
    """逐帧高召回判定（LOCATE 粗扫 / MODERATE）。内部并发，自动降级（§6.6）。
    返回 [(frame_path, t, schema_obj), ...]，顺序同输入。"""
    provider = get_provider(role, override)
    if not provider.api_key:
        raise LLMError(f"provider {provider.name} 缺少 API key："
                       f"请设置环境变量 {PROVIDERS[provider.name]['api_key_env']}")
    results = judge_frames_batch(provider, system, frames_with_time, schema,
                                 instruction=instruction)
    print(f"  [llm] {provider.name}:{provider.model}  judged {len(results)} frames")
    return results


def describe(role, image_path, *, override=None):
    """一次 vision 调用，返回参考图的文字描述（FR-2.3），供 VISUAL_LOCATE 交叉验证。
    失败返回空串（描述只是辅助，不应中断主流程）。"""
    from ..config import DESCRIBE_MAX_TOKENS
    from ..prompts import DESCRIBE_INSTRUCTION, DESCRIBE_SYSTEM
    provider = get_provider(role, override)
    if not provider.api_key:
        return ""
    try:
        r = provider.chat(system=DESCRIBE_SYSTEM, images=[ImageRef(str(image_path))],
                          messages=[Msg("user", DESCRIBE_INSTRUCTION)],
                          max_tokens=DESCRIBE_MAX_TOKENS)
        _log_usage(provider, r)
        return (r.text or "").strip()
    except Exception:  # noqa: BLE001 —— 描述失败不影响后续像素预筛与语义确认
        return ""


def text_structured(role, instruction, schema, *, system="", override=None):
    provider = get_provider(role, override)
    if not provider.api_key:
        raise LLMError(f"provider {provider.name} 缺少 API key："
                       f"请设置环境变量 {PROVIDERS[provider.name]['api_key_env']}")
    obj, result = structured(provider, system, [], instruction, schema)
    _log_usage(provider, result)
    return obj
