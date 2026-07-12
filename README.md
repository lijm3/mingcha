# 明察 (MingCha)

> 一个能“看懂、看准、看住”视频的多模型 AI 智能体。取自「明察秋毫」。

明察是**完全自包含**的独立项目：取视频、抽帧、去重、转写、拼图等预处理全部内化在本仓库（`media.py` + `preprocess.py`，媒体 IO 部分移植并改编自开源工具 [claude-real-video](../claude-real-video)），在其上构建编排层——**意图分类 → 管线规划 → 多模型分析 → 带证据（时间戳/截图）组装**。多模型可插拔（Claude / GPT / GLM，原生 HTTP，无 SDK 绑定）。

## 五类意图

| 意图 | 示例 | 输出 | 状态 |
|---|---|---|---|
| SUMMARY 理解 | “总结这个视频讲了什么” | 结构化摘要 | ✅ 端到端可用 |
| LOCATE 定位 | “琼A 车牌最早出现在什么时间” | 精确时间戳 + 截图 | ✅ 已实现 |
| MODERATE 审核 | “有没有暴力画面” | 高召回判定 + 命中区间 | ✅ 已实现 |
| VISUAL_LOCATE 以图搜 | 附图 + “这个人出现在哪” | 时间戳 + 相似度 | ✅ 已实现 |
| PLATE 车牌/车辆 | “识别视频里的车牌” | 车牌号 + 高亮标注视频 | ✅ 已实现 |

> 每个功能是什么、背后怎么实现的（小白也能看懂）→ **[docs/功能详解-小白版.md](docs/功能详解-小白版.md)**

## 安装

```bash
pip install -e .                 # 完全自包含，无需先装 crv
pip install -e ".[whisper]"      # 可选：启用 whisper 语音转写
```

系统依赖：`ffmpeg` / `ffprobe` 需在 PATH（非 pip 可装）。`yt-dlp` 与 `Pillow` 已作为 pip 依赖自动安装。

## 用法

```bash
export ANTHROPIC_API_KEY=sk-...
mingcha ask <video-url-or-path> "总结这个视频讲了什么"
```

模型选择（密钥永远从环境变量读，不进命令行）：

```bash
mingcha ask video.mp4 "总结" --provider glm                 # 全部角色切到 GLM
mingcha ask video.mp4 "总结" --vision-model openai:gpt-5.5  # 只画面分析用 GPT
```

## 文档

📖 **完整文档中心（推荐从这里开始）：[docs/README.md](docs/README.md)**

- 新手：[快速入手](docs/快速入手.md) · [功能详解-小白版](docs/功能详解-小白版.md) · [使用手册](docs/使用手册.md)
- 设计：[需求文档](docs/视频智能体-需求文档.md) · [详细设计](docs/视频智能体-详细设计文档.md) · [车牌识别设计](docs/车牌识别高亮追踪-详细设计文档.md)
- 部署：[前后端启动指南](docs/前后端分离-启动指南.md) · [前后端详细设计](docs/前后端分离-详细设计文档.md)

> 关键实现：明察**完全自包含、不依赖 crv 运行时**。媒体 IO（取视频/转写/音频/探测）见 `media.py`（移植改编自 crv）；「抽帧 → 时间戳 → 去重 → 拼图」链路见 `preprocess.py` / `timestamps.py`，全程携带时间戳（G1 地基）。
