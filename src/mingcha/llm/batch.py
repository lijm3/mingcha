"""逐帧高召回判定 —— §6.6 / §7.3。

设计里 MODERATE 逐帧判定「支持 batch 则走批处理省钱，否则降级同步并发」。真实
Anthropic Batches / OpenAI Batch 是**异步提交 + 轮询**（分钟级时延），且经第三方代理
时兼容性不确定；审核类要求较快反馈（§15-4）。因此这里统一实现**稳健的同步并发**
（ThreadPoolExecutor 限流）作为对所有 provider 都可用的主路径——功能等价、实时性好，
省钱靠 §6.5 的 system prompt caching 与两阶段扫描兜底。真正的离线批处理留作后续优化。

单帧异常不影响整体：拒答（LLMRefusal）→ 标注「建议人工复核」；其它错误 → 标注失败原因。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..config import JUDGE_MAX_WORKERS
from .base import ImageRef, LLMRefusal
from .structured import structured


def judge_frames_batch(provider, system: str, frames_with_time, schema, *,
                       instruction: str = "") -> list[tuple[str, float, object]]:
    """对 [(frame_path, t), ...] 逐帧结构化判定，并发执行、保持输入顺序。
    返回 [(frame_path, t, schema_obj), ...]。见设计文档 §6.6 / §7.3。"""
    items = list(frames_with_time)
    n = len(items)
    if n == 0:
        return []

    def work(idx: int):
        path, t = items[idx]
        try:
            obj, _ = structured(provider, system, [ImageRef(path)], instruction, schema)
            return idx, path, t, obj
        except LLMRefusal:
            obj = schema()
            if hasattr(obj, "note"):
                obj.note = "模型拒答（内容安全过滤），建议人工复核"
            return idx, path, t, obj
        except Exception as e:  # noqa: BLE001 —— 单帧失败不拖垮整批
            obj = schema()
            if hasattr(obj, "note"):
                obj.note = f"判定失败（{type(e).__name__}）"
            return idx, path, t, obj

    results: list = [None] * n
    with ThreadPoolExecutor(max_workers=min(JUDGE_MAX_WORKERS, n)) as ex:
        for idx, path, t, obj in ex.map(work, range(n)):
            results[idx] = (path, t, obj)
    return results
