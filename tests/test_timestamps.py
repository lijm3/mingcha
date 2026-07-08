"""G1 时间戳地基单测：hms 格式、metadata 解析、raw 映射、write/load 往返。"""
import re

from mingcha import timestamps


def test_hms_format():
    assert timestamps.hms(0) == "00:00:00.000"
    assert timestamps.hms(12.5) == "00:00:12.500"
    assert timestamps.hms(3661.25) == "01:01:01.250"
    assert re.match(r"^\d\d:\d\d:\d\d\.\d\d\d$", timestamps.hms(0.999))
    assert timestamps.hms(-5) == "00:00:00.000"       # 负数夹到 0


def test_parse_meta(tmp_path):
    meta = tmp_path / "frames_meta.txt"
    meta.write_text(
        "frame:0    pts:0      pts_time:0\n"
        "lavfi.scene_score=0.100000\n"
        "frame:1    pts:10240  pts_time:1.5\n"
        "frame:2    pts:20480  pts_time:3.0\n",
        encoding="utf-8")
    assert timestamps.parse_meta(str(meta)) == [0.0, 1.5, 3.0]
    assert timestamps.raw_stamp_map(str(meta)) == {
        "raw_00001.jpg": 0.0, "raw_00002.jpg": 1.5, "raw_00003.jpg": 3.0}


def test_parse_meta_missing(tmp_path):
    # 文件不存在 → 空列表（触发上层退化估算）
    assert timestamps.parse_meta(str(tmp_path / "nope.txt")) == []


def test_write_load_roundtrip(tmp_path):
    out = str(tmp_path)
    timestamps.write(out, {"frame_002.jpg": 5.0, "frame_001.jpg": 1.0})
    stamps = timestamps.load(out)
    # 按 t 升序落盘
    assert [s.frame for s in stamps] == ["frame_001.jpg", "frame_002.jpg"]
    assert [s.t for s in stamps] == [1.0, 5.0]
    assert stamps[0].hms == "00:00:01.000"
