"""structured._extract_json 健壮性回归测试。

覆盖第三方代理偶发不走 tool_use、改由 text block 手写多行 JSON 时的常见瑕疵：
字符串内裸换行（复现线上 JSONDecodeError: Expecting ',' delimiter）、字符串内花括号、
尾随逗号、```json fence、前导 thinking 文本。见 llm/structured.py。
"""
import pytest

from mingcha.llm.structured import _extract_json


def test_multiline_with_bare_newline_in_string():
    """复现线上报错：多行缩进 JSON 且字符串值内含裸换行。"""
    blob = ('{\n  "topic": "测试",\n  "segments": ["第一段\n仍在同一字符串"],\n'
            '  "key_points": [],\n  "summary": "行一\n行二"\n}')
    d = _extract_json(blob)
    assert d["topic"] == "测试"
    assert d["segments"][0].startswith("第一段")


def test_braces_inside_string_value():
    """字符串值内含花括号，不能让括号匹配错位。"""
    d = _extract_json('{"topic": "公式 {a+b}", "summary": "见 {x}"}')
    assert d["topic"] == "公式 {a+b}"


def test_trailing_comma():
    d = _extract_json('{"topic": "t", "segments": ["a",], "summary": "s",}')
    assert d["segments"] == ["a"]


def test_json_fence():
    d = _extract_json('```json\n{"topic": "t", "summary": "s"}\n```')
    assert d["topic"] == "t"


def test_leading_thinking_text():
    d = _extract_json('让我分析一下。\n\n{"topic": "t", "summary": "s"}')
    assert d["topic"] == "t"


def test_no_json_raises():
    with pytest.raises(ValueError):
        _extract_json("这里完全没有 JSON 对象")
