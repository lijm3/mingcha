"""LLM 适配层的**门面（facade）**——把「选 provider → 校验密钥 → 结构化输出 → 记用量」
这套固定动作收敛成几个函数，供上层 analyzer 直接调用：

    vision_structured  一次多图 vision 调用 → 结构化对象（SUMMARY 摘要 / VISUAL 语义确认）
    judge_frames       逐帧高召回判定，内部并发（LOCATE 粗扫 / MODERATE 审核）
    text_structured    纯文本结构化，无图（如意图分类的 LLM few-shot 兜底）
    describe           参考图 → 一段文字描述（VISUAL_LOCATE 交叉验证，失败不致命）

analyzer 只依赖上面这几个 + get_provider，**不关心背后是 Claude 还是 GPT / GLM**：
线格式差异藏在 anthropic.py / openai_compat.py，结构化差异藏在 structured.py。
新增一家模型通常只改 config.PROVIDERS / ROLES（数据驱动），无需动本文件。

对应设计文档 §6.1。
"""
from __future__ import annotations

from ..config import PROVIDERS, api_key, resolve_role  # 注册表 + 密钥解析 + 角色→(provider,model)
from .._log import get_logger
from .anthropic import AnthropicProvider              # kind="anthropic"：POST /v1/messages 线格式
from .base import (
    Capabilities, ImageRef, LLMError, LLMProvider, LLMRefusal, LLMResult, Msg,
)
from .batch import judge_frames_batch                 # 逐帧同步并发（ThreadPoolExecutor 限流）
from .openai_compat import OpenAICompatProvider       # kind="openai_compat"：/v1/chat/completions
from .structured import structured                    # 跨 provider 结构化输出 + 校验 + 回灌重试

# kind 字符串 → Provider 实现类的分派表。只有新增「一种线格式」才需在此登记；
# 新增同类模型（换 base_url/model）只改 config.PROVIDERS，不动这里。
_KIND = {"anthropic": AnthropicProvider, "openai_compat": OpenAICompatProvider}

log = get_logger("mingcha.llm")

# 门面对外的公开面：analyzer 应只从这里取用，其余符号均为内部实现细节。
__all__ = ["get_provider", "vision_structured", "text_structured",
           "judge_frames", "describe",
           "ImageRef", "LLMProvider", "LLMError", "LLMRefusal"]


def get_provider(role: str, override: str | None = None) -> LLMProvider:
    """按角色实例化一个 provider。role ∈ {'vision','classify'}。

    三步走：resolve_role 把 (role, override) 解析成 (provider 名, model) → 从 PROVIDERS
    注册表取该家的线格式/base_url/能力位 → 用 _KIND 选实现类并实例化。
    override 形如 'openai:gpt-5.5'（指定家+模型）或 'glm'（只切家、用该角色默认模型）。

    注意：这里**不校验密钥**——api_key(prov_name) 即便返回 None 也照常构造 provider；
    是否持有 key 由各门面函数（vision_structured 等）在真正发请求前检查并报错。
    """
    prov_name, model = resolve_role(role, override)
    spec = PROVIDERS.get(prov_name)
    if not spec:
        raise ValueError(f"未知 provider: {prov_name}（可选: {list(PROVIDERS)}）")
    caps = Capabilities(**spec["caps"])   # 能力位：vision / json_schema / prompt_cache / batch
    cls = _KIND[spec["kind"]]             # 线格式实现类（anthropic / openai_compat）
    log.debug("角色 %s → %s:%s @ %s", role, prov_name, model, spec["base_url"])
    return cls(prov_name, model, spec["base_url"], api_key(prov_name), caps)


def _log_usage(provider: LLMProvider, result: LLMResult) -> None:
    """统一记录一次调用的 token 用量。兼容两套字段名：Anthropic 用
    input_tokens/output_tokens，OpenAI 系用 prompt_tokens/completion_tokens。"""
    u = result.usage or {}
    it = u.get("input_tokens", u.get("prompt_tokens"))
    ot = u.get("output_tokens", u.get("completion_tokens"))
    log.info("用量 %s:%s  in=%s out=%s", provider.name, provider.model, it, ot)


def vision_structured(role, system, images, instruction, schema, *,
                      override=None, cacheable_images=()):
    """一次多图 vision 调用，产出经 Pydantic 校验的结构化对象。analyzer 的主力入口。

    images            图片路径（str/Path）或 ImageRef 的混合列表；裸路径按下方规则包成 ImageRef。
    schema            Pydantic 模型类（见 types.py），既作输出契约又作校验器。
    cacheable_images  其中的路径会被标记 cacheable=True；仅支持 prompt caching 的 provider
                      （caps.prompt_cache，目前只有 claude）会真正利用，其它家忽略此标记（§6.5）。

    流程：选 provider → 校验密钥 → 归一化 images → structured() 发请求+校验+失败回灌 →
    记用量 → 返回 schema 实例。
    """
    import time
    provider = get_provider(role, override)
    if not provider.api_key:
        raise LLMError(f"provider {provider.name} 缺少 API key："
                       f"请设置环境变量 {PROVIDERS[provider.name]['api_key_env']}")
    # 归一化：已是 ImageRef 的原样保留；裸路径按是否落在 cacheable 集合里决定 cacheable 位。
    cset = {str(c) for c in cacheable_images}
    imgs = []
    for i in images:
        if isinstance(i, ImageRef):
            imgs.append(i)
        else:
            s = str(i)
            imgs.append(ImageRef(s, cacheable=s in cset))
    log.info("vision_structured[%s] %s:%s  schema=%s  images=%d",
             role, provider.name, provider.model, schema.__name__, len(imgs))
    t0 = time.time()
    obj, result = structured(provider, system, imgs, instruction, schema)
    log.info("vision_structured 完成 %s（%.1fs）", schema.__name__, time.time() - t0)
    _log_usage(provider, result)
    return obj


def judge_frames(role, system, frames_with_time, schema, *,
                 override=None, instruction=""):
    """逐帧高召回判定（LOCATE 粗扫 / MODERATE 审核）。内部 ThreadPoolExecutor 并发，
    单帧拒答/异常自动降级为占位结果、绝不拖垮整批（§6.6，详见 batch.judge_frames_batch）。

    frames_with_time  [(frame_path, t 秒), ...]。
    返回 [(frame_path, t, schema_obj), ...]，顺序严格同输入（便于按时间轴回填结果）。
    """
    import time
    provider = get_provider(role, override)
    if not provider.api_key:
        raise LLMError(f"provider {provider.name} 缺少 API key："
                       f"请设置环境变量 {PROVIDERS[provider.name]['api_key_env']}")
    n = len(frames_with_time) if hasattr(frames_with_time, "__len__") else "?"
    log.info("judge_frames[%s] %s:%s  逐帧判定 %s 帧（schema=%s）…",
             role, provider.name, provider.model, n, schema.__name__)
    t0 = time.time()
    results = judge_frames_batch(provider, system, frames_with_time, schema,
                                 instruction=instruction)
    log.info("judge_frames 完成 %d 帧（%.1fs）", len(results), time.time() - t0)
    return results


def describe(role, image_path, *, override=None):
    """一次 vision 调用，返回参考图的文字描述（FR-2.3），供 VISUAL_LOCATE 交叉验证。

    与其它门面不同的两点：① 走**低层 provider.chat()** 直接取自由文本（要描述而非结构化）；
    ② **吞掉所有异常**返回空串——描述只是辅助信号，缺了也不该中断后续像素预筛与语义确认。
    缺 key 时同样静默返回 ""（不像其它门面那样抛 LLMError）。
    """
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
    """纯文本结构化（无图），等价于 images=[] 的 vision_structured。
    典型用途是意图分类的 LLM few-shot 兜底；role 视场景选取（如分类传 'classify'，
    走更轻量的模型，见 config.ROLES）。缺 key 抛 LLMError。"""
    provider = get_provider(role, override)
    if not provider.api_key:
        raise LLMError(f"provider {provider.name} 缺少 API key："
                       f"请设置环境变量 {PROVIDERS[provider.name]['api_key_env']}")
    obj, result = structured(provider, system, [], instruction, schema)
    _log_usage(provider, result)
    return obj
