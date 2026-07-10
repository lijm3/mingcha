"""结果组装 —— FR-6。把分析结果写成 answer.json；提供 not_found / not_implemented
及 from_locate / from_moderate / from_visual。对应设计文档 §8。
"""
from __future__ import annotations

import os

from . import timestamps
from .types import Answer, Evidence


def write(answer: Answer, out_dir: str) -> Answer:
    """把 Answer 落盘为 answer.json（= Answer.model_dump_json），返回该 Answer。"""
    if not answer.artifacts_dir:
        answer.artifacts_dir = os.path.abspath(out_dir)
    path = os.path.join(out_dir, "answer.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(answer.model_dump_json(indent=2))
    return answer


def not_found(out_dir: str, target: str | None, caveats: str,
              intent: str = "LOCATE") -> Answer:
    """FR-6.2 诚实否定：找不到目标时如实回答，并声明采样局限（NFR-1）。"""
    return Answer(intent=intent, target=target,
                  answer=f"在抽取的关键帧中未发现{('：' + target) if target else '目标'}。",
                  confidence=0.0,
                  caveats=caveats or "受采样密度限制，未抽到的帧中可能存在。",
                  artifacts_dir=os.path.abspath(out_dir))


def not_implemented(intent: str, out_dir: str) -> Answer:
    """骨架留桩意图的合法 Answer，保证 CLI 不崩溃。"""
    return Answer(intent=intent,
                  answer=f"意图 {intent} 尚未实现（首期仅 SUMMARY 端到端可用）。",
                  confidence=0.0,
                  caveats="该意图为骨架占位，见设计文档实现路线 P2 / P2.5 / P3。",
                  artifacts_dir=os.path.abspath(out_dir))


def from_locate(target: str, frame: str, t: float, confidence: float, note: str,
                out_dir: str, caveats: str) -> Answer:
    """LOCATE 命中：目标最早出现的帧 + 精确时间戳。"""
    hms = timestamps.hms(t)
    ev = Evidence(frame=os.path.basename(frame), t=round(t, 3), hms=hms,
                  confidence=round(confidence, 3), note=note)
    return Answer(intent="LOCATE", target=target,
                  answer=f"目标「{target}」最早出现在 {hms}（约第 {t:.1f} 秒）。",
                  evidence=[ev], confidence=round(confidence, 3),
                  caveats=caveats, artifacts_dir=os.path.abspath(out_dir))


def from_moderate(target: str, ranges: list[tuple[float, float]],
                  evidences: list[Evidence], out_dir: str, caveats: str,
                  review_note: str = "") -> Answer:
    """MODERATE 审核：命中片段合并为时间区间；高召回，建议人工复核。"""
    if not ranges:
        return Answer(intent="MODERATE", target=target,
                      answer=f"在抽取的关键帧中未发现「{target}」相关内容。",
                      confidence=0.0,
                      caveats=caveats or "受采样密度限制，未抽到的帧中可能存在；如需从严请人工复核。",
                      artifacts_dir=os.path.abspath(out_dir))
    spans = "；".join(f"{timestamps.hms(a)}~{timestamps.hms(b)}" for a, b in ranges)
    ans = (f"检测到疑似「{target}」内容，命中时间区间：{spans}。"
           f"（自动审核为高召回结果，请以人工复核为准。）")
    if review_note:
        ans += f" {review_note}"
    conf = max((e.confidence for e in evidences), default=0.0)
    return Answer(intent="MODERATE", target=target, answer=ans,
                  evidence=evidences, confidence=round(conf, 3),
                  caveats=caveats or "高召回自动初筛，可能有误报；最终判定需人工复核。",
                  artifacts_dir=os.path.abspath(out_dir))


def from_visual(query_image: str, desc: str, verdict: str, frame: str, t: float,
                similarity: float, confidence: float, note: str,
                out_dir: str, caveats: str) -> Answer:
    """VISUAL_LOCATE 命中：区分 same/similar，相似≠同一时降级措辞与 confidence（R-5）。"""
    hms = timestamps.hms(t)
    ev = Evidence(frame=os.path.basename(frame), t=round(t, 3), hms=hms,
                  confidence=round(confidence, 3), similarity=round(similarity, 3),
                  verdict=verdict, note=note)
    if verdict == "same":
        head = f"参考图中的对象最早出现在 {hms}（约第 {t:.1f} 秒）。"
    else:  # similar
        head = (f"在 {hms}（约第 {t:.1f} 秒）找到与参考图**外观相似**的对象，"
                f"但不能确定为同一个体（相似≠同一）。")
    if desc:
        head += f"（参考图：{desc}）"
    return Answer(intent="VISUAL_LOCATE",
                  target=desc or None, query_image=os.path.abspath(query_image),
                  answer=head, evidence=[ev], confidence=round(confidence, 3),
                  caveats=caveats, artifacts_dir=os.path.abspath(out_dir))
