"""PLATE 编排级冒烟测试 —— 意图/规划/降级/诚实否定/mock 全管线/回写坐标/抽帧改造。

全 mock：不真调 CV 模型、不烧 token、不需 ffmpeg/GPU/[plate] extra（仿 test_smoke_summary）。
"""
import json
import os

import mingcha.media as media
import mingcha.plate as plate
import mingcha.preprocess as preprocess
from mingcha import annotate
from mingcha.preprocess import PreprocessResult
from mingcha.types import BBox, Intent, PlateDetection, PlateTrack


def _make_source(tmp_path):
    src = str(tmp_path / "x.mp4")
    open(src, "wb").close()
    return src


def _fake_pre(out):
    def run(source, out_dir, plan, **kw):
        return PreprocessResult(
            out_dir=out, video="x.mp4", duration=10, frames_dir=out, frame_count=5,
            extracted=5, transcript_path=None, transcript_note="(none)",
            audio_path=None, timestamps_path=None, grids=[])
    return run


# ---- 意图分类 + 管线规划 ----
def test_plate_intent_and_plan():
    from mingcha.intent import classify
    from mingcha.planner import plan
    ir = classify("识别车牌并高亮跟随")
    assert ir.intents == [Intent.PLATE]
    pl = plan(ir.intents)
    assert pl.dedup_threshold == 0          # 关去重（连续帧是追踪前提）
    assert pl.grid_label_time is False       # 不走拼图 VLM 范式
    assert pl.frame_width == 1280            # 车牌小目标高分辨率
    assert pl.keep_audio and pl.scene == 0.0


def test_plate_keyword_not_stolen_by_moderate():
    # 「识别车牌」含 MODERATE 关键词「识别」，但 PLATE 优先级更高，不被抢走
    from mingcha.intent import classify
    assert classify("识别车牌").intents == [Intent.PLATE]


# ---- 未装 [plate] extra → 可读降级（永不崩）----
def test_plate_unavailable_degrades(tmp_path, monkeypatch):
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    src = _make_source(tmp_path)
    monkeypatch.setattr(preprocess, "run", _fake_pre(out))

    def boom(*a, **k):
        raise ImportError("No module named 'hyperlpr3'", name="hyperlpr3")
    monkeypatch.setattr(plate, "run_pipeline", boom)

    from mingcha.orchestrator import Orchestrator
    ans = Orchestrator().ask(src, "识别车牌", out_dir=out)
    assert ans.intent == "PLATE"
    assert ans.confidence == 0.0
    assert "hyperlpr3" in ans.answer and "plate" in ans.answer  # 指明缺哪个 + 安装办法
    assert ans.caveats                                          # 降级也带说明


# ---- mock 全管线：走完 from_plate，answer.json 合法含 plate_tracks ----
def test_plate_pipeline_mock(tmp_path, monkeypatch):
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    src = _make_source(tmp_path)
    monkeypatch.setattr(preprocess, "run", _fake_pre(out))

    def fake_run(out_dir, plan, source_video, **kw):
        tr = PlateTrack(
            track_id=1, plate_text="京A12345", confidence=0.92, plate_color="蓝",
            first_t=3.0, last_t=7.0, n_frames=20, method="vote",
            detections=[PlateDetection(t=3.0, frame="frame_003.jpg",
                                       bbox=BBox(x=100, y=200, w=120, h=40),
                                       text="京A12345", ocr_confidence=0.92, plate_color="蓝")])
        return [tr], os.path.join(out_dir, "annotated.mp4")
    monkeypatch.setattr(plate, "run_pipeline", fake_run)

    from mingcha.orchestrator import Orchestrator
    ans = Orchestrator().ask(src, "识别车牌", out_dir=out)
    assert ans.intent == "PLATE"
    assert "京A12345" in ans.answer
    assert ans.evidence and ans.evidence[0].plate_text == "京A12345"
    assert ans.evidence[0].bbox is not None and ans.evidence[0].track_id == 1

    d = json.load(open(os.path.join(out, "answer.json"), encoding="utf-8"))
    assert d["plate_tracks"] and d["plate_tracks"][0]["plate_text"] == "京A12345"
    assert d["annotated_video"].endswith("annotated.mp4")


# ---- 诚实否定：检出车牌区域但全模糊 → 否定 + caveats 非空（NFR-1）----
def test_plate_honest_negative_all_blurry(tmp_path, monkeypatch):
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    src = _make_source(tmp_path)
    monkeypatch.setattr(preprocess, "run", _fake_pre(out))

    def fake_run(out_dir, plan, source_video, **kw):
        tr = PlateTrack(track_id=1, plate_text="", confidence=0.0, first_t=1.0, last_t=2.0,
                        n_frames=2, detections=[PlateDetection(
                            t=1.0, frame="frame_001.jpg", bbox=BBox(x=0, y=0, w=10, h=5))])
        return [tr], None
    monkeypatch.setattr(plate, "run_pipeline", fake_run)

    from mingcha.orchestrator import Orchestrator
    ans = Orchestrator().ask(src, "识别车牌", out_dir=out)
    assert ans.intent == "PLATE"
    assert ans.confidence == 0.0
    assert not ans.evidence
    assert ans.caveats and "模糊" in ans.answer


# ---- 回写：插值 + 坐标系还原（纯计算，不需 ffmpeg）----
def test_annotate_interp_and_scale():
    dets = [PlateDetection(t=0.0, frame="a", bbox=BBox(x=0, y=0, w=10, h=10)),
            PlateDetection(t=2.0, frame="b", bbox=BBox(x=20, y=0, w=10, h=10))]
    mid = annotate._interp_bbox(dets, 1.0)
    assert mid.x == 10                                    # 线性插值中点
    assert annotate._interp_bbox(dets, 3.0) is None       # 区间外不外推
    # 检测在 1280 抽帧系 → 1920 原分辨率，scale=1.5
    assert annotate._scale_bbox(BBox(x=100, y=200, w=120, h=40), 1920 / 1280) == (150, 300, 180, 60)


# ---- 回写：mock ffmpeg，走通抽帧→绘制→编码，产出 annotated.mp4 ----
def test_annotate_render_builds_video(tmp_path, monkeypatch):
    from PIL import Image
    out = str(tmp_path / "out")
    work = os.path.join(out, "annotate")
    os.makedirs(work, exist_ok=True)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "af_%05d.jpg" in cmd and "-vf" in cmd:          # 抽帧 → 伪造两张原分辨率帧
            for n in (1, 2):
                Image.new("RGB", (1920, 1080), "black").save(os.path.join(work, f"af_{n:05d}.jpg"))
            with open(os.path.join(work, "annotate_meta.txt"), "w", encoding="utf-8") as fh:
                fh.write("frame:0 pts:0 pts_time:0\nframe:1 pts:1 pts_time:0.5\n")
        elif "_video.mp4" in cmd:                          # 编码 → 伪造无声视频
            open(os.path.join(work, "_video.mp4"), "wb").close()

        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr(annotate.subprocess, "run", fake_run)
    monkeypatch.setattr(media, "duration", lambda v: 1)
    monkeypatch.setattr(media, "has_audio", lambda v: False)

    tr = PlateTrack(track_id=1, plate_text="京A12345", confidence=0.9, first_t=0.0, last_t=0.6,
                    n_frames=2, detections=[
                        PlateDetection(t=0.0, frame="f", bbox=BBox(x=100, y=100, w=120, h=40)),
                        PlateDetection(t=0.5, frame="f", bbox=BBox(x=140, y=100, w=120, h=40))])
    res = annotate.render("video.mp4", out, [tr], frame_width=1280, max_seconds=600)
    assert res and res.endswith("annotated.mp4") and os.path.exists(res)
    assert any("af_%05d.jpg" in c for c in calls)          # 走了 cwd+相对名 抽帧手法


def test_annotate_no_valid_tracks_returns_none(tmp_path):
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    blurry = PlateTrack(track_id=1, plate_text="", confidence=0.0, detections=[])
    assert annotate.render("v.mp4", out, [blurry], frame_width=1280) is None


# ---- 抽帧改造：scene<=0（PLATE）走纯均匀抽帧、高分辨率 ----
def test_extract_frames_scene_zero_uniform(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd

        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr(preprocess.subprocess, "run", fake_run)
    monkeypatch.setattr(preprocess, "fps", lambda v: 30.0)
    preprocess.extract_frames_timed("v.mp4", str(tmp_path / "frames"),
                                    scene=0.0, fps_floor=0.2, scale_width=1280)
    cmd = captured["cmd"]
    vf = cmd[cmd.index("-vf") + 1]
    assert "gt(scene" not in vf                # PLATE 关场景选择
    assert "not(mod(n,6))" in vf               # 30fps × 0.2 → every_n=6（5fps 均匀）
    assert "scale=1280:-1" in vf               # 车牌高分辨率
