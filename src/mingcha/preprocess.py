"""预处理编排（明察对 crv 零侵入的关键）。

明察不调用 crv 的 process()（它去重后重命名会丢弃时间信息，事后无法可靠对回）。
这里自己编排：
  ① 取视频       —— 复用 crv.fetch_video
  ② 带时间戳抽帧  —— 自写 extract_frames_timed（ffmpeg select + metadata=print）
  ③ 解析时间戳    —— timestamps.raw_stamp_map（+ 段数校验兜底）
  ④ 带时间戳去重  —— 自写 dedup_timed（复用 crv 的 16×16 RGB 像素思路，因其为
                     dedup_frames 内部嵌套函数不可 import；时间戳穿过去重+抽稀+重命名）
  ⑤ 转写        —— 复用 crv.existing_subtitles / transcribe
  ⑥ 音频(可选)   —— 复用 crv.extract_full_audio
  ⑦ 写时间戳     —— timestamps.write -> timestamps.json
  ⑧ 拼图        —— 自写 make_grids_timed（标签用 hms 而非文件名）

对应设计文档 §3 / §6.1（按「完全不改 crv」决策重写为明察自包含）。
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass, field

from . import timestamps
from .config import FRAME_SCALE_WIDTH, MAX_FRAMES_HARD
from .media import (
    duration, existing_subtitles, extract_full_audio, fetch_video,
    fps, has_audio, have, transcribe,
)
from .types import Plan


@dataclass
class PreprocessResult:
    out_dir: str
    video: str
    duration: int
    frames_dir: str
    frame_count: int
    extracted: int
    transcript_path: str | None
    transcript_note: str
    audio_path: str | None
    timestamps_path: str | None
    grids: list[str] = field(default_factory=list)
    caveats: str = ""


# ---- ② 带时间戳抽帧 ----
def extract_frames_timed(video: str, frames_dir: str, scene: float,
                         fps_floor: float) -> tuple[int, str]:
    """一趟 ffmpeg select（同 crv 表达式）后接 metadata=print 落 pts_time。
    用 cwd=frames_dir + 相对文件名，绕开 Windows 盘符冒号在 filtergraph 里的转义地狱。
    返回 (raw 帧数, frames_meta.txt 路径)。"""
    os.makedirs(frames_dir, exist_ok=True)
    every_n = max(1, round(fps(video) * fps_floor))
    vf = (f"select='gt(scene,{scene})+not(mod(n,{every_n}))',"
          f"metadata=print:file=frames_meta.txt,"
          f"scale={FRAME_SCALE_WIDTH}:-1")
    subprocess.run(
        ["ffmpeg", "-i", os.path.abspath(video), "-vf", vf,
         "-fps_mode", "vfr", "raw_%05d.jpg",
         "-hide_banner", "-loglevel", "error"],
        cwd=frames_dir, capture_output=True, text=True)
    meta_path = os.path.join(frames_dir, "frames_meta.txt")
    n_raw = len(glob.glob(os.path.join(frames_dir, "raw_*.jpg")))
    return n_raw, meta_path


# ---- ④ 带时间戳去重（复用 crv 的像素思路，时间戳穿透）----
def dedup_timed(frames_dir: str, raw_stamps: dict[str, float], threshold: float = 8,
                window: int = 4, max_frames: int | None = 150,
                dropped_dir: str | None = None) -> tuple[int, list[dict], dict[str, float]]:
    """对 raw_*.jpg 按真实像素差滑窗去重 + 均匀抽稀 + 重命名 frame_NNN.jpg，
    全程携带时间戳。返回 (保留数, per-frame records, {frame_NNN.jpg: t})。

    sig/pct_diff 直接照搬 crv dedup_frames 的思路（那里是内部嵌套函数，无法 import）。"""
    from PIL import Image

    frames = sorted(glob.glob(os.path.join(frames_dir, "raw_*.jpg")))

    def sig(path: str, size: int = 16):
        # RGB（非灰度）：等亮度换色（红→绿）不能在比较器里看起来相同
        return list(Image.open(path).convert("RGB").resize((size, size)).getdata())

    def pct_diff(a, b, tol: int = 25) -> float:
        changed = sum(max(abs(x[0] - y[0]), abs(x[1] - y[1]), abs(x[2] - y[2])) > tol
                      for x, y in zip(a, b))
        return 100.0 * changed / len(a)

    kept: list[str] = []
    recent: list = []            # 最近 window 张保留帧的签名
    records: list[dict] = []
    for f in frames:
        h = sig(f)
        dist = min((pct_diff(h, k) for k in recent), default=None)
        name = os.path.basename(f)
        if dist is None or dist > threshold:
            kept.append(f)
            recent.append(h)
            if len(recent) > window:
                recent.pop(0)
            records.append({"name": name, "dist": dist, "kept": True})
        else:
            if dropped_dir:
                os.makedirs(dropped_dir, exist_ok=True)
                shutil.move(f, os.path.join(dropped_dir, name))
            else:
                os.remove(f)
            records.append({"name": name, "dist": dist, "kept": False})

    # 均匀抽稀（护栏：即使 max_frames=None，也不超过 MAX_FRAMES_HARD）
    cap = max_frames if max_frames else MAX_FRAMES_HARD
    if len(kept) > cap:
        step = len(kept) / cap
        keep_idx = {int(i * step) for i in range(cap)}
        for i, f in enumerate(list(kept)):
            if i not in keep_idx:
                kept.remove(f)
                os.remove(f)
                nm = os.path.basename(f)
                for rec in records:
                    if rec["name"] == nm:
                        rec["kept"] = False
                        rec["capped"] = True

    # 重命名 raw_* -> tmp_NNN -> frame_NNN，同步搬运时间戳
    kept_stamps: dict[str, float] = {}
    for i, f in enumerate(sorted(kept), 1):
        raw_name = os.path.basename(f)
        new_name = f"frame_{i:03d}.jpg"
        if raw_name in raw_stamps:
            kept_stamps[new_name] = raw_stamps[raw_name]
        os.rename(f, os.path.join(frames_dir, f"tmp_{i:03d}.jpg"))
    for f in sorted(os.listdir(frames_dir)):
        if f.startswith("tmp_"):
            os.rename(os.path.join(frames_dir, f),
                      os.path.join(frames_dir, "frame_" + f[4:]))
    return len(kept), records, kept_stamps


# ---- ⑧ 拼图（标签用时间戳）----
def make_grids_timed(frames_dir: str, out_dir: str, label_map: dict[str, str] | None = None,
                     cols: int = 3, rows: int = 3, cell_width: int = 480) -> list[str]:
    """把 frame_*.jpg 按序拼成 3×3 contact sheet；格子标签用 hms（供 vision 直接引用时间）。"""
    from PIL import Image, ImageDraw

    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frames:
        return []
    grids_dir = os.path.join(out_dir, "grids")
    os.makedirs(grids_dir, exist_ok=True)
    per = cols * rows
    label_h = 22
    sheets: list[str] = []
    for gi in range(0, len(frames), per):
        batch = frames[gi:gi + per]
        first = Image.open(batch[0])
        cw = cell_width
        ch = int(first.height * cw / first.width) + label_h
        sheet = Image.new("RGB", (cols * cw, rows * ch), "black")
        draw = ImageDraw.Draw(sheet)
        for i, f in enumerate(batch):
            im = Image.open(f).resize((cw, ch - label_h))
            x, y = (i % cols) * cw, (i // cols) * ch
            sheet.paste(im, (x, y + label_h))
            name = os.path.basename(f)
            label = (label_map or {}).get(name, name)
            draw.text((x + 6, y + 4), label, fill="white")
        dest = os.path.join(grids_dir, f"grid_{gi // per + 1:02d}.jpg")
        sheet.save(dest, quality=85)
        sheets.append(dest)
    return sheets


# ---- 编排 ----
def run(source: str, out_dir: str, plan: Plan, *, cookies: str | None = None,
        cookies_from_browser: str | None = None, want_grids: bool = True,
        progress=None, cancel=None) -> PreprocessResult:
    """预处理主编排。progress(state, fraction, note) 可选回调，把内部各步细粒度进度
    上报给上层（后端 SSE）；为 None 时静默（CLI 行为不变）。cancel() 返回 True 则中止。"""
    def emit(state: str, frac: float, note: str) -> None:
        if progress:
            progress(state, frac, note)

    def check_cancel() -> None:
        if cancel:
            cancel()

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    # 覆盖式语义：清掉上一次的 frames/ 与 grids/，避免旧 frame_*.jpg 残留导致
    # 去重阶段 tmp_NNN→frame_NNN 重命名在 Windows 上撞已存在文件（FileExistsError），
    # 也避免旧帧/旧拼图混入本次分析。
    for stale in (frames_dir, os.path.join(out_dir, "grids")):
        shutil.rmtree(stale, ignore_errors=True)

    # ① 取视频（复用 crv）
    check_cancel()
    emit("downloading", 0.12, "取视频中…")
    video = fetch_video(source, out_dir, cookies=cookies,
                        cookies_from_browser=cookies_from_browser)
    dur = duration(video)
    emit("extracting", 0.2, f"取视频完成（时长约 {dur}s），开始抽帧…")

    # ② 带时间戳抽帧
    check_cancel()
    n_raw, meta_path = extract_frames_timed(video, frames_dir, plan.scene, plan.fps_floor)
    emit("extracting", 0.4, f"抽帧完成：{n_raw} 帧，去重中…")

    # ③ 解析时间戳 + 段数校验兜底（NFR-1 诚实）
    raw_stamps = timestamps.raw_stamp_map(meta_path)
    caveats = ""
    if len(raw_stamps) != n_raw:
        n_meta = len(raw_stamps)
        raw_stamps = {f"raw_{i + 1:05d}.jpg": i * plan.fps_floor for i in range(n_raw)}
        caveats = (f"时间戳退化估算：ffmpeg metadata 段数({n_meta})与抽帧数({n_raw})不符，"
                   f"改用 index×fps_floor 近似，时间点可能有偏差。")

    # ④ 带时间戳去重
    check_cancel()
    kept, _records, kept_stamps = dedup_timed(
        frames_dir, raw_stamps, threshold=plan.dedup_threshold, max_frames=plan.max_frames)
    emit("extracting", 0.5, f"去重后保留 {kept} 帧（自 {n_raw}）")

    # ⑤ 转写（复用 crv 三级回退：sidecar/内嵌字幕 → whisper）
    check_cancel()
    transcript = existing_subtitles(source, video, out_dir)
    if transcript:
        note = f"{transcript} (from the video's own subtitles)"
    elif not have("whisper"):
        note = "(none — no existing subtitles; install whisper to transcribe)"
    elif not has_audio(video):
        note = "(none — this video has no subtitles and no audio track)"
    else:
        emit("transcribing", 0.55, "whisper 转写中（较慢，请耐心等待）…")
        transcript = transcribe(video, out_dir, "auto")
        note = (f"{transcript} (transcribed by whisper)" if transcript
                else "(none — transcription failed)")

    # ⑥ 音频（可选）
    audio_path = extract_full_audio(video, out_dir) if plan.keep_audio else None

    # ⑦ 写时间戳
    ts_path = timestamps.write(out_dir, kept_stamps) if kept_stamps else None

    # ⑧ 拼图（SUMMARY 分析吃 grids）
    grids: list[str] = []
    if want_grids:
        label_map = None
        if plan.grid_label_time and kept_stamps:
            label_map = {f: timestamps.hms(t) for f, t in kept_stamps.items()}
        grids = make_grids_timed(frames_dir, out_dir, label_map)
    emit("extracting", 0.65, f"预处理完成：{kept} 帧 | 拼图 {len(grids)}")

    return PreprocessResult(
        out_dir=out_dir, video=video, duration=dur, frames_dir=frames_dir,
        frame_count=kept, extracted=n_raw, transcript_path=transcript,
        transcript_note=note, audio_path=audio_path, timestamps_path=ts_path,
        grids=grids, caveats=caveats)
