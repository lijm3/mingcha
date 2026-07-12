"""VLM 车牌兜底纠错 —— FR-5.5 / §6.8，明察相对纯 CV 方案的差异化（P5.4）。

对多帧投票后仍低置信的少数疑难 track：取其最清晰的几张车牌裁剪图（可选超分/透视矫正），
连同候选读数交给 VLM，按中国车牌规则给最可能读数 + 逐位依据。复用 llm.vision_structured 门面
（与 visual._pair_judge 同款调用，支持 prompt caching 省钱）。

这是**唯一会烧 token 的环节**，仅对少数 track、每 track 2–3 张小图触发，成本可控。裁剪用 PIL
（核心依赖，不需 cv2），使 VLM 兜底在无 [plate] extra 时也能跑（只要有 key）。拒答/异常 →
返回 None，保留投票结果（永不崩，同 visual）。
"""
from __future__ import annotations

import os


def correct(track, frames_dir, *, out_dir=None, vision_model=None, top_k: int = 3):
    """对低置信 track 用 VLM 兜底纠错。返回 PlateVLMSchema | None（无图 / 失败 → None）。"""
    from .. import llm
    from ..prompts import PLATE_VLM_INSTRUCTION, plate_vlm_system
    from ..types import PlateVLMSchema

    crops = _crop_best(track, frames_dir, out_dir, top_k)
    if not crops:
        return None
    try:
        return llm.vision_structured(
            "vision", plate_vlm_system(track.plate_text or None),
            images=crops, instruction=PLATE_VLM_INSTRUCTION,
            schema=PlateVLMSchema, override=vision_model, cacheable_images=crops[:1])
    except Exception:  # noqa: BLE001 —— VLM 拒答/异常/缺 key → 保留投票结果（永不崩）
        return None


def _crop_best(track, frames_dir, out_dir, k: int) -> list[str]:
    """取 track 中 ocr_confidence 最高的 k 帧，裁剪车牌小图（可选超分/透视矫正）存盘，返回路径。"""
    from PIL import Image
    from . import preprocess_img, superres

    dets = sorted([d for d in track.detections if d.bbox],
                  key=lambda d: d.ocr_confidence, reverse=True)[:k]
    if not dets:
        return []
    dest_dir = os.path.join(out_dir or frames_dir, "plates")
    os.makedirs(dest_dir, exist_ok=True)
    paths: list[str] = []
    for i, d in enumerate(dets):
        fp = os.path.join(frames_dir, os.path.basename(d.frame))
        if not os.path.exists(fp):
            continue
        try:
            with Image.open(fp) as im:
                im = im.convert("RGB")
                box = (max(0, d.bbox.x), max(0, d.bbox.y),
                       min(im.width, d.bbox.x + d.bbox.w),
                       min(im.height, d.bbox.y + d.bbox.h))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                crop = im.crop(box)
                crop = preprocess_img.rectify(crop, getattr(d, "corners", None))  # 斜视角矫正
                crop = preprocess_img.enhance(crop)                               # 对比度/锐化
                crop = superres.enhance(crop)                                     # 低置信 → 超分
                out_path = os.path.join(dest_dir, f"track_{track.track_id:03d}_{i}.jpg")
                crop.save(out_path, quality=92)
                paths.append(out_path)
        except Exception:  # noqa: BLE001 —— 单图裁剪失败跳过，不拖垮兜底
            continue
    return paths
