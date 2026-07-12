"""车牌超分辨率 —— FR-5.5 / §6.7，低置信按需触发（P5.4）。

引擎 Real-ESRGAN（或车牌专用超分），体积大、依赖重，故**不进 [plate] 主 extra**，另拆
[plate-superres]（默认不装）。惰性 import：未装 / 权重未预置 / 失败 → 原样返回（降级、不抛，
局限由上层在 caveats 声明）。仅对低置信 track 的少数清晰帧触发，不全量跑（超分慢，CPU 尤甚）。
"""
from __future__ import annotations

_UPSAMPLER = None


def enhance(image, *, scale: int = 4):
    """超分放大车牌裁剪小图（PIL.Image → PIL.Image）。

    未装 [plate-superres] / 权重未预置 / 失败 → 原样返回（降级，绝不虚假超分或抛异常）。
    """
    try:
        import numpy as np
        from PIL import Image
    except Exception:  # noqa: BLE001
        return image
    up = _get_upsampler(scale)
    if up is None:
        return image
    try:
        out, _ = up.enhance(np.array(image.convert("RGB")), outscale=scale)
        return Image.fromarray(out)
    except Exception:  # noqa: BLE001 —— 超分失败不影响后续 VLM 兜底
        return image


def _get_upsampler(scale: int):
    """构造 Real-ESRGAN upsampler 单例；未装 / 权重未预置 → None（降级为原图）。

    真跑接入（[plate-superres] + 权重就位后取消注释、按官方配置）：
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, scale=scale)
        return RealESRGANer(scale=scale, model_path="weights/RealESRGAN_x4plus.pth", model=model)
    """
    global _UPSAMPLER
    if _UPSAMPLER is not None:
        return _UPSAMPLER
    try:
        import realesrgan  # noqa: F401 —— 仅探测是否装了超分栈
    except Exception:  # noqa: BLE001
        return None
    # 装了 realesrgan 但权重路径依部署环境，未配置时返回 None → 降级（不虚假超分）。
    return None
