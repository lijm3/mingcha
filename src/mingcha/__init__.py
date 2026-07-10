"""明察 (MingCha) — 看懂/看准/看住视频的多模型 AI 智能体。

取自「明察秋毫」。本包**完全自包含**：在自研的视频预处理链路（取视频 → 抽帧 →
去重 → 转写 → 拼图，全程携带时间戳）之上，构建一层编排——**意图分类 → 管线规划
→ 多模型分析 → 带证据（时间戳/截图）组装**。
（媒体 IO 层 media.py 移植并改编自开源工具 crv，现已内化为自有代码，不再 import crv。）

本文件是包的**顶层入口**，刻意保持极薄，只做两件事：
  1. 暴露包版本号 ``__version__``（供 cli.py 的 --version 与启动横幅读取）；
  2. 提供便捷函数 ``ask``，作为 ``Orchestrator().ask(...)`` 的一行式封装。

Python API::

    from mingcha import ask
    answer = ask("video.mp4", "总结这个视频讲了什么")

``ask`` 内部为**惰性导入**（见下），使 ``import mingcha`` 本身不会拉起 orchestrator
及其庞大依赖链（llm / analyzer / preprocess…），保证顶层 import 轻量、启动快。
"""
# 开启后本文件里 `str | None` 这类 PEP 604 联合类型注解会延迟为字符串、不在导入时
# 求值，从而在 Python 3.9 等旧版本也能书写（全项目各模块统一沿用此约定）。
from __future__ import annotations

# 包版本号，唯一真相源。⚠️ pyproject.toml 里另有一份写死的 version，二者需手动同步。
__version__ = "0.1.0"


def ask(source: str, prompt: str, query_image: str | None = None, **kwargs):
    """便捷入口：一行式跑完整条流水线，等价于 ``Orchestrator().ask(...)``。

    参数：
        source      视频来源——本地路径或可下载 URL（B 站 / YouTube 等，交由 yt-dlp）。
        prompt      自然语言问题；意图分类据此决定走哪个/哪些分析器。
        query_image 参考图路径，仅 VISUAL_LOCATE（以图搜）用到，其余意图留空。
        **kwargs    透传给 ``Orchestrator.ask``，常用：
                      out_dir='mingcha-out'          输出目录（覆盖式重写）
                      use_cache=True                 指纹命中则复用上次结果（NFR-6）
                      cookies / cookies_from_browser 取需登录的视频
                      progress / cancel              进度回调与取消钩子（后端 SSE 用）

    返回：``types.Answer``——结构化结果（核心 ``answer`` 文本 + ``evidence`` 证据 +
        ``caveats`` 采样局限声明等），并同时落盘到 ``out_dir/answer.json``。

    两点说明：
      · **惰性导入** Orchestrator：把 import 放进函数体，让 ``import mingcha`` 不必
        牵连 orchestrator 的依赖链，顶层保持轻量（动机同上，模块 docstring 已述）。
      · 这里用的是默认 ``Orchestrator()``，故**无法在此切 provider / 指定模型**；
        需要按角色 override 时，请直接构造 ``Orchestrator(provider='glm', …).ask(...)``。
    """
    from .orchestrator import Orchestrator
    return Orchestrator().ask(source, prompt, query_image=query_image, **kwargs)
