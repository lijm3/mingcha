# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

明察 (MingCha) 是一个**完全自包含**的多模型视频分析 AI 智能体。取自「明察秋毫」，目标是"看懂、看准、看住"视频。核心是在自研的视频预处理链路（取视频 → 抽帧 → 去重 → 转写 → 拼图，全程携带时间戳）之上，构建一个编排层：**意图分类 → 管线规划 → 多模型分析 → 带证据（时间戳/截图）组装**。

四类意图（对应 `analyzer/` 下四个分析器）：
- **SUMMARY 理解**（已实现，端到端可用）— 结构化摘要
- **LOCATE 定位**（已实现，待真跑）— 目标最早出现的精确时间戳，两阶段粗扫→精扫
- **MODERATE 审核**（已实现，待真跑）— 高召回内容审核，命中区间合并
- **VISUAL_LOCATE 以图搜**（已实现，待真跑）— 参考图 + 三级由粗到细（像素预筛→语义确认→精扫）

## 常用命令

```bash
# 安装（完全自包含，无需其它前置包）
pip install -e .
pip install -e ".[whisper]"      # 可选：启用 whisper 语音转写

# 运行测试（全部 mock LLM，不烧 token、不需网络/API key）
pytest
pytest tests/test_timestamps.py            # 单个文件
pytest tests/test_smoke_summary.py::test_summary_smoke   # 单个用例

# 运行 CLI
mingcha ask <video-url-or-path> "总结这个视频讲了什么"
mc ask video.mp4 "总结" --provider glm                 # 全部角色切到 GLM
mc ask video.mp4 "定位" --vision-model openai:gpt-5.5  # 仅画面分析用 GPT
mc ask video.mp4 "以图搜" --image ref.jpg              # VISUAL_LOCATE
mc ask video.mp4 "总结" --no-cache                     # 强制重算，忽略缓存
```

**系统依赖**：`ffmpeg` / `ffprobe` 必须在 PATH 上（非 pip 可装）。`yt-dlp` 和 `Pillow` 已作为 pip 依赖。测试不需要这些，但真实运行需要。

## 架构（数据流）

`Orchestrator.ask()`（`orchestrator.py`）是唯一入口，串起整条流水线，与设计文档 §9 一一对应：

```
输入校验(_validate) → 意图分类(intent) → 管线规划(planner)
  → [缓存指纹命中则直接返回] → 预处理(preprocess) → 分析分发(_dispatch)
  → 组装落盘(assembler.write → answer.json) → 写缓存指纹
```

关键分层与文件职责：

- **`types.py`** — 所有数据结构用 Pydantic 定义（`Intent`/`Plan`/`Evidence`/`Answer` + 各分析器的输出 schema）。这些 schema 同时被 `llm/structured.py` 用作跨 provider 的结构化输出校验契约。改动数据结构从这里开始。

- **`intent.py`** — 意图分类。**规则兜底优先**（关键词匹配，省一次 LLM 调用），规则未命中才走 LLM few-shot；LLM 失败兜底为 SUMMARY，保证永不崩。

- **`planner.py`** — 纯函数，把意图列表映射为 `Plan`（抽帧密度、去重阈值、是否保留音频等）。多意图取"最严格"并集。MODERATE 会关去重、不封顶帧数（分段密采）。

- **`preprocess.py`** — 预处理编排的核心，也是本项目"完全自研、不依赖 crv"的关键。**不调用任何 crv 的 process()**，而是自己编排 8 步：取视频 → 带时间戳抽帧 → 解析时间戳 → 带时间戳去重 → 转写 → 音频 → 写 timestamps.json → 拼图。抽帧/去重/拼图全部自写，时间戳穿透整条链路（这是 G1 地基）。

- **`media.py`** — 媒体 IO 层（取视频/转写/音频/ffprobe 探测）。**移植并改编自开源工具 claude-real-video (crv)**，现已内化为明察自有代码，不再 `import claude_real_video`。风格特点：`subprocess` 静默执行外部命令、**不检查返回码**，靠后续文件存在性检查兜底（调试时留意）。

- **`timestamps.py`** — G1 时间戳地基。解析 ffmpeg `metadata=print` 输出的 `pts_time`，维护"帧→秒"映射，读写 `timestamps.json`。`hms()` 用整数毫秒运算避免浮点边界。

- **`llm/`** — 多模型适配层，**原生 HTTP，无任何官方 SDK 绑定**：
  - `__init__.py` — 门面，上层 analyzer 只依赖 `vision_structured` / `text_structured` / `judge_frames` / `describe`，不知背后是 Claude 还是 GPT/GLM。
  - `base.py` — `LLMProvider` ABC + 统一请求/响应类型 + `post_with_retry`（429/5xx 指数退避）。
  - `anthropic.py` — POST `/v1/messages`，结构化输出走 `tool_use` 强制，支持 prompt caching。
  - `openai_compat.py` — POST `/v1/chat/completions`（GPT/GLM/兼容网关），结构化走 `response_format`。
  - `structured.py` — 跨 provider 结构化输出：原生 json_schema 优先，否则内联 schema 兜底，统一抽 JSON + Pydantic 校验 + 失败回灌重试 1 次。
  - `batch.py` — 逐帧高召回判定的**同步并发**实现（ThreadPoolExecutor 限流）。注意：设计文档提到 batch API，但真实 batch 是异步提交-轮询、时延分钟级，不适合审核实时性，故这里统一用同步并发。

- **`analyzer/`** — 四个分析器，各自消费预处理产物 + 调 `llm` 门面 + 调 `assembler` 组装 `Answer`：
  - `summary.py` — grids 拼图 + 转写 → 一次 vision 调用 → 结构化摘要。
  - `locate.py` — 两阶段：粗扫逐帧取最早命中 → 命中帧 ±WINDOW 秒 `rescan.dense_extract` 密采精扫到秒。
  - `moderate.py` — 逐帧高召回（低阈值少漏）→ 相邻命中点合并为时间区间。
  - `visual.py` — 三级：`similarity.rank` 像素预筛 Top-K → 参考图+候选帧成对语义确认 → 精扫。区分"同一个体"vs"同类外观"。

- **`rescan.py`** — 两阶段精扫。用 `-ss/-to` 快速定位候选区间密采。**注意**：`-ss` 置于 `-i` 前会把时间戳重置到 0，解析出的 `pts_time` 是相对区间起点的，需回加 t0 还原绝对时间。

- **`assembler.py`** — 结果组装。所有 `Answer` 从这里产出（`write`/`not_found`/`from_locate`/`from_moderate`/`from_visual`）。落盘 `answer.json`。

## 关键约定与陷阱

- **G1 时间戳地基不可绕过**：整条链路的核心价值是每一帧都能可靠对回源视频的绝对时间。任何改动抽帧/去重/精扫的代码，都必须保证时间戳穿透。preprocess 不用 crv 的 process() 正是因为它去重后重命名会丢弃时间信息。

- **Windows filtergraph 转义**：ffmpeg 抽帧统一用 `cwd=子目录 + 相对文件名`，绕开 Windows 盘符冒号在 filtergraph 里的转义地狱。新增 ffmpeg 调用请沿用这个手法（见 `preprocess.extract_frames_timed` / `rescan.dense_extract`）。

- **诚实优先（NFR-1）**：找不到目标要走 `assembler.not_found` 如实否定，并且 `caveats` 必填非空声明采样局限。时间戳退化估算（metadata 段数与抽帧数不符）也要在 caveats 标注。测试 `test_locate_not_found` 专门守这条。

- **永不崩**：意图分类失败、单帧判定失败（含模型拒答 `LLMRefusal`）、缓存损坏、provider 配置损坏，都有兜底路径，不中断主流程。新增逻辑请保持这个风格。

- **新增 provider/模型 = 改配置，通常不写代码**：`config.py` 的 `PROVIDERS`（注册表）和 `ROLES`（角色→模型映射）是数据驱动的。也支持项目级 `.crv/providers.{yaml,json}` 覆盖，无需改代码。

- **密钥管理**：`config.api_key()` 优先级为 直填字段 → 环境变量 → 容错（若 `api_key_env` 里填的其实是 key 本身则直接用）。⚠️ 当前 `config.py` 的 `claude`/`openai` provider 把真实 key 直接写进了 `api_key_env` 字段（依赖容错分支生效）——**这些 key 不应提交到版本库**，改动 config.py 时注意不要扩散密钥。密钥永远不从 CLI 参数、也不从 YAML 覆盖文件读取。

- **缓存复用（NFR-6）**：指纹 = (source 含本地文件 mtime+size, prompt, plan, 模型)。命中且 `answer.json` 完好则跳过预处理与分析。`--no-cache` 强制重算。

- **输出目录是覆盖式的**：默认 `mingcha-out/`，重跑会覆盖（`.gitignore` 已排除 `*-out/`、`answer.json` 等运行产物）。

## 现状

- 9 项测试全绿，但**均为 mock provider，未真跑烧 token**。四类分析器逻辑就绪，但第三方代理（camel-hub / denxio）对 Anthropic `tool_use`、OpenAI `response_format` 线格式的兼容性尚未真实验证。
- `OpenAICompatProvider` 与 LLM few-shot 分类：代码就位，未接真实 GPT/GLM 测试。
- 详细进度对照见 `docs/实现进度.md`；需求与详细设计见 `docs/` 下另两个文档。
