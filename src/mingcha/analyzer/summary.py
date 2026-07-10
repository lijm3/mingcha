"""SUMMARY 分析器 —— FR-5.1。把关键帧拼图 + 语音转写喂给 vision 档模型，产出结构化摘要，
等价于把当前"人工读 MANIFEST、看图说话"这件事自动化。对应设计文档 §7.1。

输入是 preprocess 已落盘的产物（out_dir 下的 grids/ 拼图、frames/ 关键帧、transcript.txt），
输出是一个 SUMMARY 意图的 Answer。这是四类 analyzer 里最简单的一个——**单次 vision 调用、
无逐帧循环**（LOCATE/MODERATE/VISUAL 才有逐帧判定）。
"""
from __future__ import annotations

import glob
import os

from .. import llm                                   # 只依赖门面 llm.vision_structured，不碰具体 provider
from ..prompts import SUMMARY_SYSTEM, summary_user   # system 提示词 + user 指令拼装（含转写与关注视角）
from ..types import Answer, Plan, SummarySchema      # 输入 Plan / 输出契约 SummarySchema / 统一 Answer


def analyze(out_dir: str, plan: Plan, prompt: str, *, vision_model=None) -> Answer:
    """对预处理产物生成结构化摘要。

    out_dir       preprocess 的输出目录，本函数从中读取拼图 / 关键帧 / 转写。
    plan          管线规划参数；SUMMARY 直接消费已落盘产物，**本函数体未引用 plan**——
                  保留该形参只为与其它三个 analyzer 的 analyze(...) 签名统一（供 orchestrator 一致分发）。
    prompt        用户原始问题，作为摘要的"关注视角"注入（见 summary_user），令摘要优先回应它。
    vision_model  角色级模型 override（如 'openai:gpt-5.5'），透传给门面；None 则用默认 vision 模型。
    """
    # 优先用拼图（contact sheet：一张含多帧、每格带时间戳，信息密度高且省 token）；
    # 没有拼图（如帧数太少未触发拼图）则退化为直接把单帧关键帧列表喂给模型。
    grids = sorted(glob.glob(os.path.join(out_dir, "grids", "grid_*.jpg")))
    frames = sorted(glob.glob(os.path.join(out_dir, "frames", "frame_*.jpg")))
    images = grids or frames

    # 读语音转写（可选）：errors="ignore" 容忍编码杂质；缺文件则留空串，
    # 由 summary_user 侧提示模型"仅依据画面分析"。
    transcript = ""
    tpath = os.path.join(out_dir, "transcript.txt")
    if os.path.exists(tpath):
        transcript = open(tpath, encoding="utf-8", errors="ignore").read()

    # 单次 vision 调用：门面内部据 provider 能力走原生 json_schema 或内联兜底，并做 Pydantic 校验。
    result: SummarySchema = llm.vision_structured(
        role="vision", system=SUMMARY_SYSTEM, images=images,
        instruction=summary_user(transcript, why=prompt),
        schema=SummarySchema, override=vision_model)

    # 组装人类可读正文：优先完整 summary，模型未给则降级用 topic，仍无则空串。
    body = (result.summary or result.topic or "").strip()
    if result.segments:
        body += "\n\n分段脉络：\n" + "\n".join(f"- {s}" for s in result.segments)
    if result.key_points:
        body += "\n\n关键点：\n" + "\n".join(f"- {k}" for k in result.key_points)
    # 三段式结构化字段（topic/segments/key_points）既拼进 body，又单独透传给 Answer：
    # 前端 SummaryCard 可分区渲染（§9.7），拿不到结构化时优雅降级为整段 body 文本。
    # confidence 固定 0.8：SUMMARY 是理解类任务，没有 LOCATE/MODERATE 那种逐帧"命中"证据，
    # 故给一个启发式的中高置信度，而非从判定结果推算。
    return Answer(intent="SUMMARY", answer=body.strip(), confidence=0.8,
                  topic=result.topic or "", segments=list(result.segments),
                  key_points=list(result.key_points), artifacts_dir=out_dir)
