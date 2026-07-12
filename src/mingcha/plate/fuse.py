"""多帧时序融合 —— FR-5.5.4 / §6.5–§6.6，对付车牌模糊的主力（P5.2 核心）。

纯 Python、零重依赖、确定性可单测。两层：
  ① 多帧字符级置信度加权投票（vote）——车在动，同一车牌经过几十帧、每帧模糊处不同，
     按字符位对齐逐位投票，得到比任何单帧都准的车牌号（追踪白送的红利，几乎零成本）。
  ② 中国车牌语法规则约束（apply_rules）——结构强、字符集受限，用规则纠正 OCR 高频混淆
     （省份 / 地区字母位 / 序号位），几乎零成本救回大量错误。
"""
from __future__ import annotations

from collections import Counter

from .. import config
from ..types import PlateTrack

# 民用 31 省市自治区简称（省份位合法性校验）
PROVINCES = set("京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新")
# 地区字母位（第 2 位）必为字母：OCR 读成数字时按形近纠回字母
_DIGIT_TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}


def vote(track: PlateTrack) -> tuple[str, float, list[float]]:
    """多帧字符级置信度加权投票。返回 (车牌号, 整牌置信度, 逐位可信度)。

    ① 按车牌长度众数对齐（普通 7 位 / 新能源 8 位）——长度异常帧剔除，避免「7 位 vs 8 位」错位；
    ② 每个字符位按各帧 char_confidence 加权累计得票，取最高票字符；
       缺逐字符置信度时退回该帧整体 ocr_confidence（再退 1.0）作权重；
    ③ 整牌置信度取最弱位（min pos_conf）——木桶效应：一位没把握，整牌就不该高置信。
    """
    dets = [d for d in track.detections if d.text]
    if not dets:
        return "", 0.0, []
    target_len = Counter(len(d.text) for d in dets).most_common(1)[0][0]
    aligned = [d for d in dets if len(d.text) == target_len]

    final: list[str] = []
    pos_confs: list[float] = []
    for i in range(target_len):
        tally: dict[str, float] = {}
        total = 0.0
        for d in aligned:
            ch = d.text[i]
            w = (d.char_confidences[i] if i < len(d.char_confidences)
                 else (d.ocr_confidence or 1.0))
            w = max(float(w), 1e-6)
            tally[ch] = tally.get(ch, 0.0) + w
            total += w
        best = max(tally, key=tally.get)
        final.append(best)
        pos_confs.append(tally[best] / total if total > 0 else 0.0)
    conf = min(pos_confs) if pos_confs else 0.0
    return "".join(final), conf, pos_confs


def apply_rules(text: str) -> tuple[str, bool]:
    """中国车牌语法规则约束纠错。返回 (纠正后文本, 是否有改动)。

    - 地区字母位（第 2 位）必为字母：数字按形近纠回字母（0→O、1→I、8→B、6→G、5→S、2→Z）；
    - 序号位（第 3 位起）：国标 GA36 不使用字母 I / O（与 1 / 0 混淆），强制 I→1、O→0。
    省份位（第 1 位）无法从数字/字母恢复汉字，其合法性由 fuse_tracks 标注存疑，不在此强改。
    长度非 7/8（异常）时原样返回、不纠。
    """
    if len(text) not in (7, 8):
        return text, False
    chars = list(text)
    changed = False
    if chars[1].isdigit() and chars[1] in _DIGIT_TO_ALPHA:
        chars[1] = _DIGIT_TO_ALPHA[chars[1]]
        changed = True
    for i in range(2, len(chars)):
        if chars[i] == "I":
            chars[i] = "1"
            changed = True
        elif chars[i] == "O":
            chars[i] = "0"
            changed = True
    return "".join(chars), changed


def is_valid_plate(text: str) -> bool:
    """粗校验中国车牌结构：省份汉字(1) + 地区字母 A–Z(1) + 5~6 位字母数字（普通 7 / 新能源 8）。
    关键：除第 1 位外不得再出现汉字——滤掉俯拍小目标上『京E苏211111』这类误识别噪声。"""
    if len(text) not in (7, 8):
        return False
    if text[0] not in PROVINCES:
        return False
    if not ("A" <= text[1] <= "Z"):
        return False
    return all(("A" <= c <= "Z") or c.isdigit() for c in text[2:])


def fuse_tracks(tracks: list[PlateTrack]) -> list[PlateTrack]:
    """对每条 track 跑 投票 → 规则约束定案，按样本量/省份合法性调整置信度与 caveats。
    原地写回 plate_text / confidence / method / plate_color / n_frames，返回同一列表。"""
    for tr in tracks:
        if not tr.n_frames:
            tr.n_frames = len(tr.detections)
        text, conf, _ = vote(tr)
        if not text:
            tr.plate_text = ""
            tr.confidence = 0.0
            tr.caveats = _add(tr.caveats, "无有效 OCR 读数")
            continue
        corrected, changed = apply_rules(text)
        tr.plate_text = corrected
        tr.method = "rule_fixed" if changed else "vote"
        tr.confidence = round(conf, 3)
        # 车牌颜色多数投票
        colors = [d.plate_color for d in tr.detections if d.plate_color]
        if colors:
            tr.plate_color = Counter(colors).most_common(1)[0][0]
        # 样本不足 → 压低置信（投票无力，§6.5 边界）
        if tr.n_frames < config.PLATE_TRACK_MIN_FRAMES:
            tr.confidence = round(tr.confidence * 0.6, 3)
            tr.caveats = _add(tr.caveats, f"样本不足（仅 {tr.n_frames} 帧）")
        # 结构合法性校验：非法（非省份首字 / 除首位混入汉字 / 长度异常）→ 大幅降置信 + 标注。
        # 滤掉俯拍小目标的『京E苏211111』这类噪声读数，避免以高置信误导（NFR-1 诚实优先）。
        if not is_valid_plate(corrected):
            tr.confidence = round(tr.confidence * 0.35, 3)
            tr.caveats = _add(tr.caveats, "车牌结构非法，疑似误识别")
    return tracks


def _add(existing: str, note: str) -> str:
    return f"{existing}；{note}" if existing else note
