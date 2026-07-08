"""两阶段精扫 —— §6.2 / §7.2。对候选区间用 ffmpeg 密集重抽并落 pts_time。

复用 preprocess.extract_frames_timed 的手法（cwd=子目录 + 相对文件名，绕开 Windows
盘符冒号在 filtergraph 里的转义）：`-ss t0 -to t1` 快速定位候选区间，`fps=1/step`
密采，`metadata=print` 落每帧 pts_time。因 `-ss` 置于 `-i` 前会把时间戳重置到 0，
解析出的 pts_time 是相对区间起点的，回加 t0 还原为源视频绝对时间。
"""
from __future__ import annotations

import glob
import os
import subprocess

from . import timestamps


def dense_extract(source_video: str, out_dir: str, t0: float, t1: float,
                  step: float = 0.3, scale_width: int = 1280) -> list[tuple[str, float]]:
    """对 [t0, t1] 密集重抽，返回 [(frame_abs_path, t_绝对秒), ...] 按时间升序。

    R-2：精扫用更高 scale_width（默认 1280）以利车牌/文字类二次识别。
    区间为空或抽帧失败时返回 []（调用方回退到粗扫命中）。见设计文档 §7.2。"""
    t0 = max(0.0, float(t0))
    t1 = float(t1)
    if t1 <= t0:
        return []

    rescan_dir = os.path.join(out_dir, "rescan")
    os.makedirs(rescan_dir, exist_ok=True)
    # 清理上一次精扫的残留（重跑覆盖式），避免旧帧混入
    for old in glob.glob(os.path.join(rescan_dir, "*")):
        try:
            os.remove(old)
        except OSError:
            pass

    vf = (f"fps=1/{step},"
          f"metadata=print:file=rescan_meta.txt,"
          f"scale={scale_width}:-1")
    subprocess.run(
        ["ffmpeg", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", os.path.abspath(source_video),
         "-vf", vf, "-fps_mode", "vfr", "rescan_%05d.jpg",
         "-hide_banner", "-loglevel", "error"],
        cwd=rescan_dir, capture_output=True, text=True)

    frames = sorted(glob.glob(os.path.join(rescan_dir, "rescan_*.jpg")))
    if not frames:
        return []

    # 相对区间起点的 pts_time；回加 t0 还原绝对时间
    rel = timestamps.parse_meta(os.path.join(rescan_dir, "rescan_meta.txt"))
    out: list[tuple[str, float]] = []
    for i, fp in enumerate(frames):
        # 段序与写盘序在单趟 vfr 下一致；段数不符则退化为 t0 + index*step 近似
        t_abs = (t0 + rel[i]) if i < len(rel) else (t0 + i * step)
        out.append((os.path.abspath(fp), round(t_abs, 3)))
    return out
