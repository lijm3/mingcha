"""VISUAL_LOCATE 预筛 —— §6.6 级1。复用 crv 的 16×16 RGB 像素思路做轻量相似度。

注：crv 的 sig/pct_diff 是 dedup_frames 内部嵌套函数不可 import；这里复用其思路自写
（与 preprocess.dedup_timed 同款逻辑）。CLIP 作为可选升级（R-5 / [clip] extra）。
"""
from __future__ import annotations

import os


def _sig(path: str, size: int = 16):
    """把图片降采样到 size×size RGB 像素列表（等亮度换色也可区分）。"""
    from PIL import Image
    return list(Image.open(path).convert("RGB").resize((size, size)).getdata())


def _pct_diff(a, b, tol: int = 25) -> float:
    """两个签名的像素差异百分比（任一通道超过 tol 即算变化）。"""
    if not a or len(a) != len(b):
        return 100.0
    changed = sum(max(abs(x[0] - y[0]), abs(x[1] - y[1]), abs(x[2] - y[2])) > tol
                  for x, y in zip(a, b))
    return 100.0 * changed / len(a)


def rank(query_image: str, frame_paths: list[str]) -> list[tuple[str, float]]:
    """对 query_image 与每张 frame 算像素相似度（相似度 = 100 - pct_diff），
    按相似度降序返回 [(frame_path, score), ...]，供取 Top-K。见设计文档 §6.6。

    读图失败的帧记 0 分排在末尾，不中断整体排序。"""
    q = _sig(query_image)
    scored: list[tuple[str, float]] = []
    for fp in frame_paths:
        try:
            score = 100.0 - _pct_diff(q, _sig(fp))
        except (OSError, ValueError):
            score = 0.0
        scored.append((fp, round(score, 2)))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored
