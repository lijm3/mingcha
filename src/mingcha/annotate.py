"""高亮回写 —— FR-5.5.5 / §7，「跟随移动」的关键。产出 annotated.mp4：每辆车一个高亮框 +
车牌号标签，框随车移动、标签固定为该 track 的最终定案文本（不用每帧抖动的原始 OCR，避免闪烁）。

复用明察既有手法：抽帧/编码走 `cwd=子目录 + 相对文件名` 绕开 Windows filtergraph 盘符冒号
转义（同 preprocess / rescan）；中文标签用 `PIL.ImageDraw`（cv2.putText 画不了汉字，同
make_grids_timed）。本模块只需 Pillow（核心依赖）+ ffmpeg，不进 [plate] 惰性探测，但 PIL 仍
在函数内 import 以保持顶层轻量（`import mingcha.annotate` 不拉起重依赖）。

坐标系还原（关键坑，§7）：检测 bbox 在 PLATE_FRAME_WIDTH（1280）抽帧坐标系，回写在原始
分辨率——按 `原宽 / frame_width` 缩放，否则框错位。插值只在 track 的检测区间内，不外推。
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess

from . import media, timestamps
from .types import BBox, PlateTrack

_WRITE_FPS = 12.0          # 回写帧率：检测帧间插值补齐到此密度，框平滑跟随（权衡平滑度与体积）
_BOX_COLORS = {"蓝": (60, 130, 255), "黄": (255, 200, 0), "绿(新能源)": (0, 200, 120),
               "双层黄": (255, 200, 0), "黑": (90, 90, 90), "白": (210, 210, 210)}
_DEFAULT_COLOR = (0, 220, 0)


def render(source_video: str, out_dir: str, tracks: list[PlateTrack], *,
           frame_width: int = 1280, max_seconds: float = 600.0,
           write_fps: float = _WRITE_FPS) -> str | None:
    """把每条有效 track 的 bbox + 最终车牌号画回视频，输出 annotated.mp4（保留原音轨）。
    无有效 track / 抽帧失败 → 返回 None（不产视频，上层走诚实否定）。"""
    # 可回写的轨迹：有标签（车牌号 / 车辆N）且多帧（单帧画不出跟随）。车辆模式无牌也回写。
    valid = [t for t in tracks if (getattr(t, "label", "") or t.plate_text) and t.n_frames >= 2]
    if not valid:
        return None

    work = os.path.join(out_dir, "annotate")
    os.makedirs(work, exist_ok=True)
    for old in glob.glob(os.path.join(work, "*")):
        try:
            os.remove(old)
        except OSError:
            pass

    # ① 抽回写帧（原分辨率、不缩放；带 pts_time 对齐 track 时间）；超长仅回写前 max_seconds 秒
    dur = min(float(media.duration(source_video) or 0), max_seconds)
    _extract_full_frames(source_video, work, write_fps, dur)
    frames = sorted(glob.glob(os.path.join(work, "af_*.jpg")))
    if not frames:
        return None
    rel = timestamps.parse_meta(os.path.join(work, "annotate_meta.txt"))

    # ② 原始尺寸 → 坐标还原比例（检测在 frame_width 抽帧系）
    from PIL import Image, ImageDraw
    with Image.open(frames[0]) as im0:
        orig_w = im0.width
    scale = orig_w / float(frame_width) if frame_width else 1.0
    font = _load_font()

    # ③ 逐帧绘制（框随插值移动，标签用定案文本）
    for i, fp in enumerate(frames):
        t = rel[i] if i < len(rel) else i / write_fps
        with Image.open(fp) as im:
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)
            for tr in valid:
                if not (tr.first_t <= t <= tr.last_t):
                    continue
                bb = _interp_bbox(tr.detections, t)
                if bb is None:
                    continue
                x, y, w, h = _scale_bbox(bb, scale)
                color = _BOX_COLORS.get(tr.plate_color or "", _DEFAULT_COLOR)
                draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
                _draw_label(draw, x, y, getattr(tr, "label", "") or tr.plate_text or "车", color, font)
            im.save(fp, quality=88)

    # ④ 编码 + 原音轨
    return _encode(source_video, work, out_dir, write_fps)


# ---- 纯计算：插值 + 坐标还原（可脱离 ffmpeg 单测）----
def _interp_bbox(dets, t: float) -> BBox | None:
    """在检测序列里对时间 t 线性插值 bbox；区间外返回 None（首/末检测帧外不外推，§7）。"""
    pts = sorted(dets, key=lambda d: d.t)
    if not pts or t < pts[0].t or t > pts[-1].t:
        return None
    prev = pts[0]
    for d in pts:
        if d.t == t:
            return d.bbox
        if d.t > t:
            span = d.t - prev.t
            if span <= 0:
                return prev.bbox
            r = (t - prev.t) / span
            return BBox(
                x=int(prev.bbox.x + (d.bbox.x - prev.bbox.x) * r),
                y=int(prev.bbox.y + (d.bbox.y - prev.bbox.y) * r),
                w=int(prev.bbox.w + (d.bbox.w - prev.bbox.w) * r),
                h=int(prev.bbox.h + (d.bbox.h - prev.bbox.h) * r))
        prev = d
    return pts[-1].bbox


def _scale_bbox(bb: BBox, scale: float) -> tuple[int, int, int, int]:
    """检测抽帧坐标系 → 原始分辨率坐标系（按 原宽/frame_width 缩放）。"""
    return int(bb.x * scale), int(bb.y * scale), int(bb.w * scale), int(bb.h * scale)


# ---- ffmpeg / PIL 细节 ----
def _extract_full_frames(video: str, work: str, fps: float, dur: float) -> None:
    vf = f"fps={fps},metadata=print:file=annotate_meta.txt"
    cmd = ["ffmpeg", "-i", os.path.abspath(video)]
    if dur > 0:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vf", vf, "-fps_mode", "vfr", "af_%05d.jpg", "-hide_banner", "-loglevel", "error"]
    subprocess.run(cmd, cwd=work, capture_output=True, text=True)


def _encode(source_video: str, work: str, out_dir: str, fps: float) -> str | None:
    """标注帧序列 → annotated.mp4（+ 原音轨，若有；无音轨直接用无声视频）。"""
    out_path = os.path.join(out_dir, "annotated.mp4")
    tmp = os.path.join(work, "_video.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", f"{fps}", "-i", "af_%05d.jpg",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "_video.mp4",
         "-hide_banner", "-loglevel", "error"],
        cwd=work, capture_output=True, text=True)
    if not os.path.exists(tmp):
        return None
    if media.has_audio(source_video):
        subprocess.run(
            ["ffmpeg", "-y", "-i", os.path.abspath(tmp), "-i", os.path.abspath(source_video),
             "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest",
             os.path.abspath(out_path), "-hide_banner", "-loglevel", "error"],
            capture_output=True, text=True)
    if not os.path.exists(out_path):
        shutil.copy(tmp, out_path)          # 无音轨 / 合并失败 → 退回无声标注视频
    return out_path if os.path.exists(out_path) else None


def _draw_label(draw, x: int, y: int, text: str, color, font) -> None:
    """框上方画车牌号标签（贴顶时改画框内），中文用 PIL 字体。"""
    ty = y - 22 if y >= 22 else y + 2
    w = 8 + int(len(text) * 16)
    draw.rectangle([x, ty, x + w, ty + 20], fill=color)
    draw.text((x + 4, ty + 2), text, fill=(0, 0, 0), font=font)


def _load_font():
    """探测常见中文字体；缺失退回默认位图字体（汉字可能显示为方框，§16 风险项）。"""
    from PIL import ImageFont
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/PingFang.ttc"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, 18)
            except OSError:
                continue
    return ImageFont.load_default()
