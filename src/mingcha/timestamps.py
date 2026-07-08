"""G1 时间戳地基：解析 ffmpeg metadata=print 的 pts_time，维护「帧→秒」映射，
读写 timestamps.json。对应设计文档 §3.3 / FR-4。

明察对 crv 零侵入 —— crv 的抽帧不携带时间戳，故这里由明察自己解析。
"""
from __future__ import annotations

import json
import os
import re

from .types import FrameStamp

_PTS_RE = re.compile(r"pts_time:([-\d.]+)")


def parse_meta(meta_path: str) -> list[float]:
    """按出现顺序返回每个被选中帧的 pts_time（秒）；索引 i 对应 raw_{i+1:05d}.jpg。
    metadata=print 每帧输出一行 `frame:N  pts:X  pts_time:Y`（后跟该帧 metadata）。
    解析失败返回空列表。"""
    times: list[float] = []
    if not meta_path or not os.path.exists(meta_path):
        return times
    with open(meta_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("frame:"):
                m = _PTS_RE.search(s)
                if m:
                    try:
                        times.append(float(m.group(1)))
                    except ValueError:
                        pass
    return times


def raw_stamp_map(meta_path: str) -> dict[str, float]:
    """{'raw_00001.jpg': 12.5, ...}，键顺序即时间顺序。"""
    return {f"raw_{i + 1:05d}.jpg": t for i, t in enumerate(parse_meta(meta_path))}


def hms(t: float) -> str:
    """12.5 -> '00:00:12.500'（用整数毫秒运算，避免浮点边界）。"""
    if t < 0:
        t = 0.0
    total_ms = int(round(t * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write(out_dir: str, kept_stamps: dict[str, float]) -> str:
    """写 timestamps.json = [{frame, t, hms}, ...]，按 t 升序。返回路径。"""
    data = [
        {"frame": frame, "t": round(t, 3), "hms": hms(t)}
        for frame, t in sorted(kept_stamps.items(), key=lambda kv: kv[1])
    ]
    path = os.path.join(out_dir, "timestamps.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return path


def load(out_dir: str) -> list[FrameStamp]:
    """读回 timestamps.json，供 analyzer 使用。"""
    path = os.path.join(out_dir, "timestamps.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [FrameStamp(**d) for d in data]
