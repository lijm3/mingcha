"""明察 (MingCha) — 看懂/看准/看住视频的多模型 AI 智能体。

构建在 claude-real-video (crv) 之上：crv 负责取视频/转写/音频等预处理，
明察新增编排层（意图分类 → 管线规划 → 多模型分析 → 带证据组装）。

Python API::

    from mingcha import ask
    answer = ask("video.mp4", "总结这个视频讲了什么")

`ask` 为惰性导入（见下），避免 `import mingcha` 时就拉起 orchestrator 及其依赖链。
"""
from __future__ import annotations

__version__ = "0.1.0"


def ask(source: str, prompt: str, query_image: str | None = None, **kwargs):
    """便捷入口：等价于 Orchestrator().ask(...)。惰性导入以保持顶层轻量。"""
    from .orchestrator import Orchestrator
    return Orchestrator().ask(source, prompt, query_image=query_image, **kwargs)
