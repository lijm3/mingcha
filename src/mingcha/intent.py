"""意图分类 —— FR-2 / G4。规则兜底优先（省一次 LLM），模糊/多意图交给 LLM，
LLM 不可用时兜底为 SUMMARY。对应设计文档 §4。
"""
from __future__ import annotations

from .types import Intent, IntentResult

_LOCATE_KW = ("什么时间", "第几秒", "最早", "出现在", "何时", "几分几秒", "时间点", "定位")
_MODERATE_KW = ("有没有", "是否", "检测", "包含", "有无", "存在吗", "审核")
_SUMMARY_KW = ("总结", "讲了什么", "概括", "主要内容", "介绍一下", "说了什么", "讲的是", "内容是")
_PLATE_KW = ("车牌", "车牌号", "牌照", "车号", "识别车牌", "高亮车牌", "标注车牌")


def _rule_based(prompt: str, has_image: bool) -> IntentResult | None:
    if has_image:
        return IntentResult(intents=[Intent.VISUAL_LOCATE],
                            reason="含参考图，强信号 → VISUAL_LOCATE")
    p = prompt or ""
    # PLATE 置于 LOCATE/MODERATE 之前：「识别/标注车牌」不应被 MODERATE「识别/检测」抢走
    if any(k in p for k in _PLATE_KW):
        return IntentResult(intents=[Intent.PLATE], target=p,
                            return_scope="all_ranges", reason="命中车牌关键词")
    if any(k in p for k in _LOCATE_KW):
        return IntentResult(intents=[Intent.LOCATE], target=p,
                            return_scope="earliest", reason="命中定位关键词")
    if any(k in p for k in _MODERATE_KW):
        return IntentResult(intents=[Intent.MODERATE], target=p,
                            return_scope="exists", reason="命中审核关键词")
    if any(k in p for k in _SUMMARY_KW):
        return IntentResult(intents=[Intent.SUMMARY], reason="命中总结关键词")
    return None


def classify(prompt: str, has_image: bool = False, *, classify_model=None) -> IntentResult:
    rule = _rule_based(prompt, has_image)
    if rule is not None:
        return rule
    # 规则未命中 → LLM few-shot 分类（classify 档）。任何异常兜底为 SUMMARY，保证不崩。
    try:
        from . import llm
        from .prompts import CLASSIFY_SYSTEM, classify_user
        return llm.text_structured("classify", classify_user(prompt, has_image),
                                   IntentResult, system=CLASSIFY_SYSTEM,
                                   override=classify_model)
    except Exception as e:  # noqa: BLE001 —— 分类失败不应中断主流程
        return IntentResult(intents=[Intent.SUMMARY],
                            reason=f"分类兜底（规则未命中且 LLM 分类不可用: {type(e).__name__}）→ SUMMARY")
