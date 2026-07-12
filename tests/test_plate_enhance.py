"""PLATE 疑难兜底单测 —— §6.7/§6.8（P5.4）：VLM 采纳策略 + 触发条件 + 裁剪调用。

纯 mock：不真调 VLM、不烧 token、不需 [plate] extra。
"""
import os

from mingcha import plate
from mingcha.plate import vlm_fallback
from mingcha.types import BBox, PlateDetection, PlateTrack, PlateVLMSchema


def _tr(text="京A1234S", conf=0.4, tid=1):
    return PlateTrack(track_id=tid, plate_text=text, confidence=conf,
                      first_t=0.0, last_t=1.0, n_frames=5)


# ---- 采纳策略（enhance_uncertain + _adopt_vlm）----
def test_vlm_disagree_takes_vlm_and_marks(monkeypatch):
    monkeypatch.setattr(vlm_fallback, "correct",
                        lambda *a, **k: PlateVLMSchema(plate_text="京A1234B", confidence=0.85))
    (out,) = plate.enhance_uncertain([_tr(conf=0.4)], "frames", "out")
    assert out.plate_text == "京A1234B"          # 不一致 → 取 VLM
    assert out.method == "vlm_corrected"
    assert out.confidence == 0.85 and "VLM 纠正" in out.caveats


def test_vlm_agree_boosts_confidence(monkeypatch):
    monkeypatch.setattr(vlm_fallback, "correct",
                        lambda *a, **k: PlateVLMSchema(plate_text="京A1234S", confidence=0.9))
    (out,) = plate.enhance_uncertain([_tr(conf=0.4)], "frames", "out")
    assert out.plate_text == "京A1234S"           # 一致 → 保留文本
    assert out.confidence == 0.9                  # 提升置信
    assert out.method != "vlm_corrected"


def test_vlm_not_triggered_when_confident(monkeypatch):
    calls = {"n": 0}

    def spy(*a, **k):
        calls["n"] += 1
        return None
    monkeypatch.setattr(vlm_fallback, "correct", spy)
    plate.enhance_uncertain([_tr(conf=0.95)], "frames", "out")   # ≥ PLATE_VLM_THRESH
    assert calls["n"] == 0                         # 高置信不烧 token


def test_vlm_failure_keeps_vote(monkeypatch):
    monkeypatch.setattr(vlm_fallback, "correct", lambda *a, **k: None)   # VLM 拒答/异常
    (out,) = plate.enhance_uncertain([_tr(conf=0.4)], "frames", "out")
    assert out.plate_text == "京A1234S" and out.confidence == 0.4        # 保留投票结果


# ---- vlm_fallback.correct：真跑 PIL 裁剪 + mock llm 门面 ----
def test_correct_crops_and_calls_llm(tmp_path, monkeypatch):
    from PIL import Image
    from mingcha import llm

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    Image.new("RGB", (1280, 720), "gray").save(str(frames_dir / "frame_001.jpg"))

    tr = PlateTrack(track_id=1, plate_text="京A1234S", confidence=0.4, first_t=0.0, last_t=1.0,
                    n_frames=1, detections=[PlateDetection(
                        t=0.0, frame="frame_001.jpg",
                        bbox=BBox(x=100, y=100, w=220, h=70), ocr_confidence=0.4)])

    captured = {}

    def fake_vs(role, system, images, instruction, schema, **kw):
        captured["images"] = list(images)
        return PlateVLMSchema(plate_text="京A1234B", confidence=0.8)
    monkeypatch.setattr(llm, "vision_structured", fake_vs)

    res = vlm_fallback.correct(tr, str(frames_dir), out_dir=str(tmp_path))
    assert res is not None and res.plate_text == "京A1234B"
    assert captured["images"]                                   # 裁剪图路径已传给 VLM
    assert (tmp_path / "plates").exists()                       # 裁剪落盘 plates/


def test_correct_no_detections_returns_none(monkeypatch):
    # 无可裁剪帧 → 不调用 VLM，返回 None（不烧 token）
    from mingcha import llm
    monkeypatch.setattr(llm, "vision_structured",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应调用 VLM")))
    assert vlm_fallback.correct(_tr(), "no_frames_dir", out_dir="out") is None
