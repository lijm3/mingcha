"""车牌识别 CV 子包 —— FR-5.5 / docs/车牌识别高亮追踪-详细设计文档.md。

门面对上层 analyzer/plate.py 只暴露 detect_and_track / ocr_frames / fuse_tracks /
run_pipeline，不泄露背后是 HyperLPR3 还是 YOLO（与四个现有 analyzer 只依赖 llm 门面同构）。
换检测/OCR 引擎 = 改本子包内部，不动 analyzer。

重依赖（hyperlpr3/opencv/onnxruntime/numpy）隔离在本子包、惰性 import：顶层只 import fuse
（纯 Python），不 import 任何重依赖；探测发生在 run_pipeline 的 _ensure_deps()。未装 [plate]
extra → ImportError（带 .name），由 analyzer.plate 兜底为诚实降级（永不崩，§10/§11）。
"""
from __future__ import annotations

import os

from .. import config
from . import fuse as _fuse


def _ensure_deps() -> None:
    """探测所需 extra；缺任一依赖抛 ImportError（带 .name），由上层降级。
    车辆模式需 ultralytics（+ 框内找车牌时的 hyperlpr3）；车牌模式需 [plate]。"""
    import cv2          # noqa: F401 —— 两种模式都要（读图/裁剪）
    import numpy        # noqa: F401
    if config.PLATE_TRACK_MODE == "vehicle":
        import ultralytics  # noqa: F401
        if config.PLATE_IN_VEHICLE:
            import hyperlpr3  # noqa: F401 —— 车辆框内二次找车牌
    else:
        import hyperlpr3    # noqa: F401


def detect_and_track(frames_with_time, *, gpu: bool = True, det_thresh=None):
    """逐帧检测 + 跨帧追踪 → list[PlateTrack]（未定案，交后续处理）。
    检测器按 config.PLATE_TRACK_MODE 选：vehicle=YOLO 车辆（大目标，跟随稳）/ plate=HyperLPR3 车牌。
    frames_with_time: [(frame_path, t), ...] 连续帧（按时间升序）。单帧检测崩 → 跳过、追踪照常。"""
    from .._log import get_logger
    from . import track as _track
    if config.PLATE_TRACK_MODE == "vehicle":
        from . import vehicle as _detector
        default_thresh, target_name = config.PLATE_VEHICLE_CONF, "车辆"
    else:
        from . import detect as _detector
        default_thresh, target_name = config.PLATE_DET_THRESH, "车牌"
    log = get_logger("mingcha.plate")
    det_thresh = default_thresh if det_thresh is None else det_thresh
    dets_by_frame = []
    frames_with_box = total_boxes = 0
    for path, t in frames_with_time:
        try:
            boxes = _detector.detect(path, gpu=gpu, det_thresh=det_thresh)
        except ImportError:
            raise                      # 依赖缺失 → 冒泡到 run_pipeline，走降级
        except Exception:              # noqa: BLE001 —— 坏帧/解码错，跳过不断链
            boxes = []
        if boxes:
            frames_with_box += 1
            total_boxes += len(boxes)
        dets_by_frame.append((t, os.path.basename(path), boxes))
    tracks = _track.associate(dets_by_frame)
    multi = sum(1 for tr in tracks if tr.n_frames > 1)
    # 诊断：区分「检测没检到」(有框帧数很低) vs「检到了没关联上」(轨迹多但都单帧)。
    log.info("PLATE 诊断[%s] | 抽帧 %d，有目标框的帧 %d，检测框合计 %d；"
             "追踪出 %d 条轨迹（多帧 %d 条，单帧 %d 条）",
             target_name, len(frames_with_time), frames_with_box, total_boxes,
             len(tracks), multi, len(tracks) - multi)
    return tracks


def ocr_frames(tracks, frames_dir, *, gpu: bool = True):
    """对每条 track 的每个检测框做二次精识别，回填 text/char_confidences/ocr_confidence/color。
    frames_dir: 帧所在目录（PlateDetection.frame 是文件名）。单框失败静默跳过（永不崩）。"""
    from . import ocr as _ocr
    for tr in tracks:
        for d in tr.detections:
            fp = os.path.join(frames_dir, os.path.basename(d.frame))
            if not os.path.exists(fp):
                continue
            text, char_confs, conf, color = _ocr.read(fp, d.bbox, gpu=gpu)
            if text:                   # 二次识别成功 → 覆盖初检读数（通常更准）
                d.text = text
                d.char_confidences = char_confs
                d.ocr_confidence = conf
                if color:
                    d.plate_color = color
    return tracks


def fuse_tracks(tracks):
    """多帧投票 + 车牌规则约束定案（转调 fuse 纯函数，§6.5–§6.6）。"""
    return _fuse.fuse_tracks(tracks)


def run_pipeline(out_dir, plan, source_video, *, vision_model=None, progress=None):
    """PLATE 主管线：密集连续帧（预处理已产出）→ 检测追踪 → OCR → 融合 → 疑难兜底 → 高亮回写。
    返回 (tracks, annotated_video)。progress(state, frac, note) 可选，上报细粒度进度（后端 SSE）。
    缺 [plate] extra → _ensure_deps 抛 ImportError，由 analyzer.plate 兜底降级（永不崩）。"""
    from .. import timestamps

    _ensure_deps()

    def emit(frac, note):
        if progress:
            progress("analyzing", frac, note)

    stamps = timestamps.load(out_dir)
    frames_dir = os.path.join(out_dir, "frames")
    frames_with_time = [(os.path.join(frames_dir, s.frame), s.t) for s in stamps
                        if os.path.exists(os.path.join(frames_dir, s.frame))]
    if not frames_with_time:
        return [], None

    gpu = config.PLATE_USE_GPU
    vehicle_mode = config.PLATE_TRACK_MODE == "vehicle"

    emit(0.78, "车辆检测与追踪…" if vehicle_mode else "车牌检测与追踪…")
    tracks = detect_and_track(frames_with_time, gpu=gpu)

    if vehicle_mode:
        for tr in tracks:
            tr.kind = "vehicle"
        if config.PLATE_IN_VEHICLE:          # 在每辆车框内二次找车牌（找到才标，找不到只框选跟随）
            emit(0.84, "车辆框内识别车牌…")
            find_plates_in_vehicles(tracks, frames_dir, gpu=gpu)
            emit(0.88, "多帧融合定案…")
            tracks = fuse_tracks(tracks)
            emit(0.90, "疑难车牌兜底纠错…")
            tracks = enhance_uncertain(tracks, frames_dir, out_dir, vision_model=vision_model)
    else:
        emit(0.84, f"逐帧 OCR（{len(tracks)} 条轨迹）…")
        tracks = ocr_frames(tracks, frames_dir, gpu=gpu)
        emit(0.88, "多帧融合定案…")
        tracks = fuse_tracks(tracks)
        emit(0.90, "疑难车牌兜底纠错…")
        tracks = enhance_uncertain(tracks, frames_dir, out_dir, vision_model=vision_model)

    _assign_labels(tracks)

    # 回写：车辆模式即使无牌也回写（框跟随车辆）；两种模式都需「多帧轨迹」才画得出跟随。
    annotated = None
    if any((t.label or t.plate_text) and t.n_frames >= 2 for t in tracks):
        emit(0.93, "高亮回写视频…")
        from .._log import get_logger
        from ..annotate import render
        annotated = render(source_video, out_dir, tracks,
                           frame_width=plan.frame_width,
                           max_seconds=config.PLATE_ANNOTATE_MAX_SEC)
        get_logger("mingcha.plate").info("PLATE 回写 | annotated=%s", annotated or "未生成")
    return tracks, annotated


def enhance_uncertain(tracks, frames_dir, out_dir, *, vision_model=None):
    """低置信 track 兜底纠错（§6.7 超分 + §6.8 VLM）。仅对 plate_text 非空且
    confidence < PLATE_VLM_THRESH 的少数疑难 track 触发 VLM（唯一烧 token 环节）；超分/透视矫正
    服务于给 VLM 更清晰的裁剪图（见 vlm_fallback._crop_best）。VLM 不可用/拒答/缺 key →
    保留投票结果（永不崩）。"""
    from . import vlm_fallback
    for tr in tracks:
        if not tr.plate_text or tr.confidence >= config.PLATE_VLM_THRESH:
            continue
        res = vlm_fallback.correct(tr, frames_dir, out_dir=out_dir, vision_model=vision_model)
        if res and res.plate_text:
            _adopt_vlm(tr, res)
    return tracks


def _adopt_vlm(tr, res):
    """VLM 采纳策略（§6.8）：与投票一致 → 提置信；不一致 → 取 VLM（有规则推理）并标注纠正。"""
    if res.plate_text == tr.plate_text:
        tr.confidence = round(min(1.0, max(tr.confidence, res.confidence)), 3)
    else:
        tr.plate_text = res.plate_text
        tr.confidence = round(res.confidence, 3)
        tr.method = "vlm_corrected"
        tr.caveats = f"{tr.caveats}；VLM 纠正" if tr.caveats else "VLM 纠正"
        if res.plate_color:
            tr.plate_color = res.plate_color
    return tr


# ---- 车辆模式专用 ----
def find_plates_in_vehicles(tracks, frames_dir, *, gpu: bool = True):
    """车辆模式：在每条车辆轨迹各检测帧的**车辆框内**用 HyperLPR3 二次找车牌。
    找到则把车牌读数填进该 detection（供 fuse 投票定案）；找不到不影响车辆轨迹（仍框选跟随）。"""
    for tr in tracks:
        for d in tr.detections:
            fp = os.path.join(frames_dir, os.path.basename(d.frame))
            if not os.path.exists(fp):
                continue
            found = _detect_plate_in_box(fp, d.bbox, gpu=gpu)
            if found:
                text, conf, color = found
                d.text = text
                d.char_confidences = [conf] * len(text)
                d.ocr_confidence = conf         # 有牌帧：置信改为车牌 OCR 置信
                d.plate_color = color
    return tracks


def _detect_plate_in_box(frame_path, bbox, *, gpu: bool = True):
    """在给定车辆 bbox 内用 HyperLPR3 找车牌。返回 (text, conf, color) 或 None（不抛，永不崩）。"""
    import cv2
    from .engine import get_catcher
    img = cv2.imread(frame_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    x1, y1 = max(0, bbox.x), max(0, bbox.y)
    x2, y2 = min(w, bbox.x + bbox.w), min(h, bbox.y + bbox.h)
    if x2 <= x1 or y2 <= y1:
        return None
    try:
        results = get_catcher(gpu)(img[y1:y2, x1:x2]) or []
    except Exception:  # noqa: BLE001
        return None
    if not results:
        return None
    best = max(results, key=lambda r: r[1] if len(r) > 1 else 0.0)
    text = str(best[0] or "")
    if not text:
        return None
    conf = float(best[1]) if len(best) > 1 else 0.0
    from .detect import _TYPE_COLOR
    color = _TYPE_COLOR.get(int(best[2])) if len(best) > 2 and best[2] is not None else None
    return text, conf, color


def _assign_labels(tracks):
    """给每条轨迹定展示/回写标签。有可信车牌 → 车牌号；车辆模式无牌 → 「车辆N」+ 用车辆检测置信。"""
    for tr in tracks:
        if tr.plate_text and tr.confidence > 0:
            tr.label = tr.plate_text
        elif tr.kind == "vehicle":
            tr.label = f"车辆{tr.track_id}"
            confs = [d.ocr_confidence for d in tr.detections if d.ocr_confidence]
            if confs:                          # 无牌车辆：置信回退为车辆检测置信均值
                tr.confidence = round(sum(confs) / len(confs), 3)
        else:
            tr.label = tr.plate_text
    return tracks
