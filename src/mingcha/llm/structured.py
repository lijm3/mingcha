"""跨 provider 结构化输出：原生 json_schema 优先，否则内联 schema 兜底，
统一 JSON 抽取 + Pydantic 校验 + 失败回灌重试 1 次。替代 SDK 的 parse。

对应设计文档 §6.4。
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from .base import LLMProvider, LLMResult, Msg


def _loads_lenient(blob: str) -> dict:
    """尽力解析 LLM 生成的 JSON，逐级放宽容忍度：
      ① 严格解析；
      ② strict=False —— 容忍字符串值内的裸控制字符（换行/Tab 等，代理手写 JSON 常见）；
      ③ 去掉对象/数组的尾随逗号后再试。
    三级都失败才抛，交由上层回灌重试。"""
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(blob, strict=False)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r",(\s*[}\]])", r"\1", blob)  # 尾逗号：{"a":1,} / [1,]
    return json.loads(cleaned, strict=False)


def _extract_json(text: str) -> dict:
    """从文本抽第一个 JSON 对象，容忍 ```json fence / 前后噪声 / 字符串内花括号。"""
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("响应中未找到 JSON 对象")
    # 字符串感知的括号匹配：跳过字符串内部的 {} 与转义，避免值里含花括号时错位
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return _loads_lenient(text[start:i + 1])
    raise ValueError("JSON 括号不匹配")


def structured(provider: LLMProvider, system: str, images, instruction: str,
               schema: type[BaseModel]) -> tuple[BaseModel, LLMResult]:
    """返回 (校验后的 pydantic 对象, 最后一次 LLMResult)。"""
    js = schema.model_json_schema()
    if getattr(provider.caps, "json_schema", False):
        r = provider.chat(system=system, images=images,
                          messages=[Msg("user", instruction)], json_schema=js)
    else:
        inline = (instruction + "\n\n严格按以下 JSON schema 输出，只输出 JSON，不要多余文字:\n"
                  + json.dumps(js, ensure_ascii=False))
        r = provider.chat(system=system, images=images, messages=[Msg("user", inline)])
    try:
        return schema.model_validate(_extract_json(r.text)), r
    except (ValidationError, ValueError) as e:
        # 失败回灌重试 1 次（对任意 provider 都有效）
        fix = (instruction + f"\n\n上次输出无法解析或不符合 schema（错误：{e}）。"
               "请只输出符合以下 schema 的合法 JSON：\n" + json.dumps(js, ensure_ascii=False))
        r2 = provider.chat(system=system, images=images, messages=[Msg("user", fix)])
        return schema.model_validate(_extract_json(r2.text)), r2
