"""PLATE 多帧融合单测 —— §6.5 投票 + §6.6 规则约束（P5.2 核心，§14 重点）。

纯函数、确定性、零重依赖：不真调 CV 模型、不烧 token、不需 ffmpeg/GPU/[plate] extra。
"""
from mingcha.plate import fuse
from mingcha.types import BBox, PlateDetection, PlateTrack


def _det(text, t=0.0, confs=None, conf=0.9, color=None):
    return PlateDetection(t=t, frame=f"frame_{int(t):03d}.jpg",
                          bbox=BBox(x=0, y=0, w=10, h=5), text=text,
                          char_confidences=confs if confs is not None else [conf] * len(text),
                          ocr_confidence=conf, plate_color=color)


def _track(texts, tid=1, confs_list=None, colors=None):
    dets = [_det(t, t=float(i),
                 confs=confs_list[i] if confs_list else None,
                 color=colors[i] if colors else None)
            for i, t in enumerate(texts)]
    return PlateTrack(track_id=tid, plate_text="", first_t=0.0,
                      last_t=float(len(texts) - 1), n_frames=len(texts), detections=dets)


# ---- 投票 ----
def test_vote_majority_beats_single_frame_noise():
    # 位 2：两帧读「1」、一帧误读「I」→ 投票取多数「1」，定案盖过任何单帧噪声
    tr = _track(["京A1234S", "京AI234S", "京A1234S"])
    text, conf, pos = fuse.vote(tr)
    assert text == "京A1234S"
    assert 0.0 <= conf <= 1.0 and len(pos) == 7


def test_vote_confidence_weighted():
    # 高置信的少数票可翻盘多数低置信票（加权投票，非简单计数）
    tr = _track(["京A1234A", "京A1234A", "京A1234B"],
                confs_list=[[0.3] * 7, [0.3] * 7, [0.9] * 7])
    text, _, _ = fuse.vote(tr)
    assert text[-1] == "B"          # 末位：2×0.3(A) < 1×0.9(B)


def test_vote_length_mode_ignores_outlier():
    # 两帧 7 位 + 一帧误检 8 位 → 按 7 位众数对齐，异常帧剔除、不错位投票
    tr = _track(["京A1234S", "京A1234S", "京A1234SX"])
    text, _, _ = fuse.vote(tr)
    assert text == "京A1234S" and len(text) == 7


# ---- 规则约束 ----
def test_rule_letter_position_digit_to_alpha():
    fixed, changed = fuse.apply_rules("京8A1234")   # 地区字母位读成数字 8 → 纠为 B
    assert changed and fixed[1] == "B"


def test_rule_serial_no_letter_IO():
    fixed, changed = fuse.apply_rules("京AI2O45")   # 序号位不许 I/O：I→1、O→0
    assert fixed == "京A12045" and changed


def test_rule_keeps_valid_plate():
    fixed, changed = fuse.apply_rules("京A1234S")
    assert fixed == "京A1234S" and not changed


def test_rule_ignores_abnormal_length():
    fixed, changed = fuse.apply_rules("京A12")       # 长度异常不纠
    assert fixed == "京A12" and not changed


# ---- 端到端融合 ----
def test_fuse_tracks_defines_plate_and_method():
    tr = _track(["京AI234S", "京A1234S", "京A1234S"], colors=["蓝", "蓝", "黄"])
    (out,) = fuse.fuse_tracks([tr])
    assert out.plate_text == "京A1234S"
    assert out.confidence > 0.0
    assert out.method in ("vote", "rule_fixed")
    assert out.plate_color == "蓝"                   # 颜色多数投票


def test_fuse_low_samples_penalized():
    (few,) = fuse.fuse_tracks([_track(["京A1234S", "京A1234S"])])          # 2 帧 < MIN(3)
    (many,) = fuse.fuse_tracks([_track(["京A1234S"] * 5, tid=2)])
    assert "样本不足" in few.caveats
    assert many.confidence >= few.confidence


def test_valid_plate_structure():
    assert fuse.is_valid_plate("京A1234S")            # 合法普通 7 位
    assert fuse.is_valid_plate("京AD12345")           # 合法新能源 8 位
    assert not fuse.is_valid_plate("京E苏211111")     # 第 3 位汉字 → 非法（误识别噪声）
    assert not fuse.is_valid_plate("甲A1234S")        # 首字非省份简称
    assert not fuse.is_valid_plate("京11234S")        # 第 2 位数字（应为字母）→ 非法
    assert not fuse.is_valid_plate("京A123")          # 长度异常


def test_fuse_illegal_structure_penalized():
    # 俯拍小目标常见的非法读数 → 大幅降置信 + 标「疑似误识别」，不以高置信误导
    (out,) = fuse.fuse_tracks([_track(["京E苏1111"] * 4)])   # 7 位但第 3 位汉字
    assert "疑似误识别" in out.caveats
    assert out.confidence < 0.5


def test_fuse_empty_reads_is_honest():
    tr = _track(["", "", ""])                              # 全无读数
    (out,) = fuse.fuse_tracks([tr])
    assert out.plate_text == "" and out.confidence == 0.0 and out.caveats
