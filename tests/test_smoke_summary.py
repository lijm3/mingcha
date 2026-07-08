"""SUMMARY / LOCATE / MODERATE 冒烟测试：mock LLM + mock 预处理，验证编排产出合法
answer.json。不依赖 ffmpeg / 网络 / API key，不烧 token。
"""
import json
import os

import mingcha.llm as llm
import mingcha.preprocess as preprocess
import mingcha.rescan as rescan
from mingcha import timestamps
from mingcha.preprocess import PreprocessResult
from mingcha.types import HitSchema, ModerateSchema, SummarySchema


def _make_source(tmp_path):
    """造一个真实存在的源文件（满足 FR-1.5 输入校验；预处理被 mock，不会真解析它）。"""
    src = str(tmp_path / "x.mp4")
    open(src, "wb").close()
    return src


def _fake_pre(out):
    """返回一个 mock 预处理器：不依赖 ffmpeg，产出到 out 目录。"""
    def run(source, out_dir, plan, **kw):
        return PreprocessResult(
            out_dir=out, video="x.mp4", duration=10, frames_dir=out, frame_count=3,
            extracted=3, transcript_path=None, transcript_note="(none)",
            audio_path=None, timestamps_path=None, grids=[])
    return run


def test_summary_smoke(tmp_path, monkeypatch):
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    src = _make_source(tmp_path)

    monkeypatch.setattr(
        llm, "vision_structured",
        lambda *a, **k: SummarySchema(topic="测试主题", summary="这是一段测试摘要。",
                                      segments=["段一"], key_points=["要点一"]))
    monkeypatch.setattr(preprocess, "run", _fake_pre(out))

    from mingcha.orchestrator import Orchestrator
    ans = Orchestrator().ask(src, "总结这个视频讲了什么", out_dir=out)

    assert ans.intent == "SUMMARY"
    assert ans.answer
    d = json.load(open(os.path.join(out, "answer.json"), encoding="utf-8"))
    assert d["intent"] == "SUMMARY"
    assert d["answer"]
    assert d["artifacts_dir"]
    for key in ("intent", "answer", "evidence", "confidence", "caveats", "artifacts_dir"):
        assert key in d


def _seed_frames(out, n=3):
    """造 frames 目录 + timestamps.json（供 LOCATE/MODERATE 的 timestamps.load 读取）。"""
    frames = os.path.join(out, "frames")
    os.makedirs(frames, exist_ok=True)
    stamps = {}
    for i in range(1, n + 1):
        name = f"frame_{i:03d}.jpg"
        open(os.path.join(frames, name), "wb").close()  # judge 被 mock，不会真读图
        stamps[name] = float(i)
    timestamps.write(out, stamps)


def test_locate_smoke(tmp_path, monkeypatch):
    """'最早出现' → 规则命中 LOCATE → 两阶段定位，产出带证据的合法 Answer。"""
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    src = _make_source(tmp_path)
    monkeypatch.setattr(preprocess, "run", _fake_pre(out))
    _seed_frames(out, 3)

    def fake_judge(role, system, items, schema, **kw):
        res = []
        for path, t in items:
            hit = abs(t - 2.0) < 0.01
            res.append((path, t, HitSchema(present=hit, confidence=0.9 if hit else 0.1,
                                           note="命中" if hit else "")))
        return res
    monkeypatch.setattr(llm, "judge_frames", fake_judge)
    monkeypatch.setattr(rescan, "dense_extract", lambda *a, **k: [])

    from mingcha.orchestrator import Orchestrator
    ans = Orchestrator().ask(src, "红色的车最早出现在什么时间", out_dir=out)
    assert ans.intent == "LOCATE"
    assert ans.evidence and ans.evidence[0].hms == "00:00:02.000"
    assert ans.confidence >= 0.5


def test_locate_not_found(tmp_path, monkeypatch):
    """粗扫全未命中 → 诚实否定（FR-6.2）+ 采样局限非空（NFR-1）。"""
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    src = _make_source(tmp_path)
    monkeypatch.setattr(preprocess, "run", _fake_pre(out))
    _seed_frames(out, 3)
    monkeypatch.setattr(
        llm, "judge_frames",
        lambda role, system, items, schema, **kw: [
            (p, t, HitSchema(present=False, confidence=0.0)) for p, t in items])

    from mingcha.orchestrator import Orchestrator
    ans = Orchestrator().ask(src, "紫色飞机最早出现在什么时间", out_dir=out)
    assert ans.intent == "LOCATE"
    assert not ans.evidence
    assert ans.caveats  # 否定结论 caveats 必填非空


def test_moderate_smoke(tmp_path, monkeypatch):
    """'有没有' → 规则命中 MODERATE → 高召回 + 区间合并。"""
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    src = _make_source(tmp_path)
    monkeypatch.setattr(preprocess, "run", _fake_pre(out))
    _seed_frames(out, 4)

    def fake_judge(role, system, items, schema, **kw):
        # 第 2、3 帧命中（相邻，应合并为一个区间）
        return [(p, t, ModerateSchema(present=(t in (2.0, 3.0)),
                                      confidence=0.8 if t in (2.0, 3.0) else 0.0))
                for p, t in items]
    monkeypatch.setattr(llm, "judge_frames", fake_judge)

    from mingcha.orchestrator import Orchestrator
    ans = Orchestrator().ask(src, "有没有暴力内容", out_dir=out)
    assert ans.intent == "MODERATE"
    assert len(ans.evidence) == 2
    assert "00:00:02.000~00:00:03.000" in ans.answer  # 相邻命中合并为一个区间


def test_cache_reuse(tmp_path, monkeypatch):
    """NFR-6：相同 (source,prompt) 第二次命中缓存，不再触发预处理与分析。"""
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)
    src = _make_source(tmp_path)

    calls = {"pre": 0, "vis": 0}

    def counting_pre(source, out_dir, plan, **kw):
        calls["pre"] += 1
        return PreprocessResult(
            out_dir=out, video="x.mp4", duration=10, frames_dir=out, frame_count=3,
            extracted=3, transcript_path=None, transcript_note="(none)",
            audio_path=None, timestamps_path=None, grids=[])

    def counting_vis(*a, **k):
        calls["vis"] += 1
        return SummarySchema(topic="t", summary="s")

    monkeypatch.setattr(preprocess, "run", counting_pre)
    monkeypatch.setattr(llm, "vision_structured", counting_vis)

    from mingcha.orchestrator import Orchestrator
    a1 = Orchestrator().ask(src, "总结这个视频", out_dir=out)
    a2 = Orchestrator().ask(src, "总结这个视频", out_dir=out)
    assert a1.answer == a2.answer
    assert calls["pre"] == 1 and calls["vis"] == 1  # 第二次全部走缓存

    # use_cache=False 强制重算
    Orchestrator().ask(src, "总结这个视频", out_dir=out, use_cache=False)
    assert calls["pre"] == 2 and calls["vis"] == 2
