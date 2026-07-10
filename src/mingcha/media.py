"""媒体 IO 层：取视频、转写、音频、探测 —— 明察完全自包含，不依赖 crv。

实现移植并改编自 claude-real-video (crv) 的 core.py（上游开源工具）：把取视频、
转写、音频这些经过实战检验、边界很多的“脏活”内化进明察，从此单包独立分发、
不再 import claude_real_video。抽帧/去重/拼图见 preprocess.py（本就是明察自写）。

这些函数用 subprocess capture_output 静默执行外部命令（ffmpeg/ffprobe/yt-dlp/whisper），
不检查返回码 —— 靠后续文件存在性检查兜底（延续 crv 的风格，调试时留意）。
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def have(tool: str) -> bool:
    """外部工具是否在 PATH 上。"""
    return shutil.which(tool) is not None


# ---- 取视频 ----
def fetch_video(src: str, out_dir: str, cookies: str | None = None,
                cookies_from_browser: str | None = None) -> str:
    """URL 走 yt-dlp（cookies 三级回退：无 → 浏览器 → 文件），本地文件直接复制到
    source.mp4。cookies 仅限本人授权访问的内容使用。"""
    dest = os.path.join(out_dir, "source.mp4")
    if src.startswith(("http://", "https://")):
        if not have("yt-dlp"):
            raise RuntimeError("未找到 yt-dlp，请安装：pip install yt-dlp")
        base = ["yt-dlp", src, "-o", dest, "--merge-output-format", "mp4",
                "--no-warnings", "-q"]
        _run(base)
        if not os.path.exists(dest) and cookies_from_browser:
            _run(base + ["--cookies-from-browser", cookies_from_browser])
        if not os.path.exists(dest) and cookies:
            _run(base + ["--cookies", cookies])
        if not os.path.exists(dest):
            # yt-dlp 可能写成了别的扩展名
            hits = sorted(glob.glob(os.path.join(out_dir, "source.*")))
            if hits:
                dest = hits[0]
        if not os.path.exists(dest):
            raise RuntimeError("下载失败（私有视频？试试 --cookies your_cookies.txt）")
    else:
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        # 本地源恰好已在输出目录且就是 source.mp4（如后端把上传直接存到任务目录）时，
        # 源=目标会触发 SameFileError；此时无需复制，直接用它。
        if os.path.abspath(src) == os.path.abspath(dest) or (
                os.path.exists(dest) and os.path.samefile(src, dest)):
            return src
        shutil.copy(src, dest)
    return dest


# ---- 探测（ffprobe）----
def duration(video: str) -> int:
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "default=nw=1:nk=1", video])
    try:
        return int(float(r.stdout.strip()))
    except (ValueError, AttributeError):
        return 0


def fps(video: str) -> float:
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=avg_frame_rate", "-of", "default=nw=1:nk=1", video])
    try:
        num, den = r.stdout.strip().split("/")
        return float(num) / float(den) if float(den) else 25.0
    except (ValueError, ZeroDivisionError, AttributeError):
        return 25.0


def has_audio(video: str) -> bool:
    r = _run(["ffprobe", "-v", "error", "-select_streams", "a",
              "-show_entries", "stream=codec_type", "-of", "csv=p=0", video])
    return bool(r.stdout.strip())


def _has_subtitle_stream(video: str) -> bool:
    r = _run(["ffprobe", "-v", "error", "-select_streams", "s",
              "-show_entries", "stream=index", "-of", "csv=p=0", video])
    return bool(r.stdout.strip())


# ---- 转写：字幕 sidecar → 内嵌字幕轨 → whisper（三级回退）----
def _subs_to_text(sub_path: str, out_txt: str) -> str | None:
    """把 .srt/.vtt 转成纯文本（去序号、时间码、样式标签）。成功返回 out_txt。"""
    try:
        raw = open(sub_path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return None
    lines: list[str] = []
    for ln in raw.splitlines():
        s = ln.strip().lstrip("﻿").strip()  # 去 BOM
        if not s or s.startswith("WEBVTT") or s.isdigit() or "-->" in s:
            continue
        s = re.sub(r"<[^>]+>", "", s)  # 去 vtt 内联标签，如 <v ->
        if s:
            lines.append(s)
    text = "\n".join(lines).strip()
    if not text:
        return None
    open(out_txt, "w", encoding="utf-8").write(text + "\n")
    return out_txt


def existing_subtitles(src: str, video: str, out_dir: str) -> str | None:
    """优先用视频已自带的字幕，不重新转写。(1) 本地源文件旁的 .srt/.vtt sidecar，
    (2) 内嵌字幕轨。返回 transcript 路径或 None。有字幕就比 whisper 更快更准。"""
    dst = os.path.join(out_dir, "transcript.txt")
    if not src.startswith(("http://", "https://")):
        base = os.path.splitext(src)[0]
        for ext in (".srt", ".vtt"):
            cand = base + ext
            if os.path.exists(cand) and _subs_to_text(cand, dst):
                return dst
    if _has_subtitle_stream(video):
        raw = os.path.join(out_dir, "_embedded.srt")
        _run(["ffmpeg", "-y", "-i", video, "-map", "0:s:0", raw,
              "-hide_banner", "-loglevel", "error"])
        if os.path.exists(raw):
            ok = _subs_to_text(raw, dst)
            try:
                os.remove(raw)
            except OSError:
                pass
            if ok:
                return dst
    return None


def transcribe(video: str, out_dir: str, lang: str | None, model: str = "base") -> str | None:
    """可选：抽音频 + 跑 whisper（需 whisper CLI 已安装）。"""
    if not have("whisper"):
        return None
    wav = os.path.join(out_dir, "audio.wav")
    _run(["ffmpeg", "-i", video, "-vn", "-ar", "16000", "-ac", "1", wav,
          "-hide_banner", "-loglevel", "error"])
    if not os.path.exists(wav):
        return None
    cmd = ["whisper", wav, "--model", model, "--output_format", "txt", "--output_dir", out_dir]
    if lang and lang != "auto":
        cmd += ["--language", lang]
    _run(cmd)
    src = os.path.join(out_dir, "audio.txt")
    dst = os.path.join(out_dir, "transcript.txt")
    if os.path.exists(src):
        os.replace(src, dst)
        return dst
    return None


# ---- 完整音轨 ----
def extract_full_audio(video: str, out_dir: str) -> str | None:
    """保存完整原声（音乐+人声+音效），供能听音频的模型使用。先尝试无损 stream copy，
    失败再转 AAC。"""
    if not has_audio(video):
        return None
    dst = os.path.join(out_dir, "audio.m4a")
    _run(["ffmpeg", "-y", "-i", video, "-vn", "-c:a", "copy", dst,
          "-hide_banner", "-loglevel", "error"])
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    _run(["ffmpeg", "-y", "-i", video, "-vn", "-c:a", "aac", "-b:a", "192k", dst,
          "-hide_banner", "-loglevel", "error"])
    return dst if os.path.exists(dst) and os.path.getsize(dst) > 0 else None
