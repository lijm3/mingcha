"""结果组装 —— FR-6。把分析结果写成 answer.json；提供 not_found / not_implemented
及 from_locate / from_moderate / from_visual。对应设计文档 §8。
"""
from __future__ import annotations

import os

from . import timestamps
from .types import Answer, Evidence, PlateTrack


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


def from_plate(tracks: list[PlateTrack], annotated_video: str | None,
               out_dir: str, caveats: str) -> Answer:
    """PLATE 组装（FR-5.5.6 / §8）：每条有效 track 一条 Evidence（代表帧 + bbox + 最终车牌号），
    全部 track 落 plate_tracks，标注视频路径落 annotated_video。

    无有效 track → 诚实否定（NFR-1），并区分「一辆没检出」与「检出但全模糊」两种情形，
    caveats 必填非空。有效 = 有车牌号且置信度 > 0。"""
    abs_out = os.path.abspath(out_dir)
    if any(getattr(t, "kind", "plate") == "vehicle" for t in tracks):
        return _from_vehicle(tracks, annotated_video, out_dir, caveats, abs_out)
    valid = [t for t in tracks if t.plate_text and t.confidence > 0]
    if not valid:
        if tracks:  # 检出了车牌区域，但都读不出 → 不硬猜（§0 诚实边界）
            ans = f"检出 {len(tracks)} 处车牌区域，但均过于模糊，无法可靠识别车牌号。"
            cav = caveats or "检出车牌区域但清晰度不足；建议提供更清晰的视频或人工复核。"
        else:
            ans = "在抽取的关键帧中未检出车牌。"
            cav = caveats or "受采样密度限制，短暂出现或远处的车牌可能漏检。"
        return Answer(intent="PLATE", answer=ans, confidence=0.0, caveats=cav,
                      plate_tracks=tracks, annotated_video=annotated_video,
                      artifacts_dir=abs_out)

    # 命中：按首次出现时间排序；每条 track 选最清晰帧（ocr_confidence 最高）作代表帧
    valid.sort(key=lambda t: t.first_t)
    evidence: list[Evidence] = []
    parts: list[str] = []
    for tr in valid:
        rep = max(tr.detections, key=lambda d: d.ocr_confidence, default=None)
        evidence.append(Evidence(
            frame=os.path.basename(rep.frame) if rep else "",
            t=round(tr.first_t, 3), hms=timestamps.hms(tr.first_t),
            confidence=round(tr.confidence, 3), bbox=rep.bbox if rep else None,
            track_id=tr.track_id, plate_text=tr.plate_text, plate_color=tr.plate_color,
            note=f"{tr.method}·{tr.n_frames}帧" + (f"·{tr.caveats}" if tr.caveats else "")))
        color = f"，{tr.plate_color}" if tr.plate_color else ""
        parts.append(f"{tr.plate_text}（{timestamps.hms(tr.first_t)}–{timestamps.hms(tr.last_t)}"
                     f"{color}，置信 {tr.confidence:.2f}）")
    ans = f"识别到 {len(valid)} 辆车的车牌：" + "；".join(parts) + "。"
    return Answer(intent="PLATE", answer=ans, evidence=evidence,
                  confidence=round(max(t.confidence for t in valid), 3), caveats=caveats,
                  plate_tracks=tracks, annotated_video=annotated_video, artifacts_dir=abs_out)


def _from_vehicle(tracks: list[PlateTrack], annotated_video: str | None,
                  out_dir: str, caveats: str, abs_out: str) -> Answer:
    """车辆追踪模式组装：展示所有稳定车辆轨迹（框选跟随），识别出车牌的额外标注车牌号。
    稳定 = 多帧（能画出跟随）或已识别出车牌；单帧车辆多为误检，不展示。"""
    valid = [t for t in tracks if t.n_frames >= 2 or (t.plate_text and t.confidence > 0)]
    if not valid:
        return Answer(intent="PLATE", answer="未稳定追踪到车辆。", confidence=0.0,
                      caveats=caveats or "未检出可稳定追踪的车辆（画面过小 / 片段过短）。",
                      plate_tracks=tracks, annotated_video=annotated_video, artifacts_dir=abs_out)
    valid.sort(key=lambda t: t.first_t)
    with_plate = [t for t in valid if t.plate_text and t.confidence > 0]
    evidence: list[Evidence] = []
    for tr in valid:
        rep = max(tr.detections, key=lambda d: d.ocr_confidence, default=None)
        evidence.append(Evidence(
            frame=os.path.basename(rep.frame) if rep else "",
            t=round(tr.first_t, 3), hms=timestamps.hms(tr.first_t),
            confidence=round(tr.confidence, 3), bbox=rep.bbox if rep else None,
            track_id=tr.track_id, plate_text=tr.plate_text or None,
            plate_color=tr.plate_color, note=tr.label))
    n, m = len(valid), len(with_plate)
    if m:
        plates = "；".join(f"{t.plate_text}（{timestamps.hms(t.first_t)}–{timestamps.hms(t.last_t)}）"
                          for t in with_plate)
        ans = f"追踪到 {n} 辆车并框选跟随，其中 {m} 辆识别出车牌：{plates}。"
    else:
        ans = f"追踪到 {n} 辆车并框选跟随；受清晰度限制未能读出车牌号。"
    return Answer(intent="PLATE", answer=ans, evidence=evidence,
                  confidence=round(max((t.confidence for t in valid), default=0.0), 3),
                  caveats=caveats, plate_tracks=tracks, annotated_video=annotated_video,
                  artifacts_dir=abs_out)


def plate_unavailable(out_dir: str, err: Exception) -> Answer:
    """未装 [plate] extra（或引擎不可用）时的可读降级 Answer（永不崩）。
    与 preprocess 里 have('whisper') 缺失时的降级同款语义。"""
    missing = getattr(err, "name", None) or str(err) or "未知依赖"
    return Answer(
        intent="PLATE",
        answer=("车牌识别需要额外依赖（HyperLPR3 / OpenCV / onnxruntime）。"
                f'请安装：pip install -e ".[plate]"（缺少：{missing}）。'),
        confidence=0.0,
        caveats="PLATE 依赖未安装，已跳过车牌分析；其它意图不受影响。",
        artifacts_dir=os.path.abspath(out_dir))
