"""OCR 前图像处理 —— FR-5.5 / §6.9，便宜且有效（P5.4）。

透视矫正（用检测四角点把拍歪的车牌摆正为矩形，对斜视角提升显著）+ 轻量增强（CLAHE 对比度
均衡 + 锐化）。全部作用于**车牌裁剪小图**、不动整帧，成本低。cv2 惰性 import（属 [plate] extra）；
缺依赖 / 无四角点 / 失败 → 原样返回（降级，不抛）。输入输出均为 PIL.Image。
"""
from __future__ import annotations


def rectify(image, corners):
    """透视矫正：把四角点 corners=[(x,y)×4] 圈定的斜视角车牌摆正为正矩形。
    corners 缺失 / 非 4 点 / cv2 不可用 / 失败 → 原样返回（降级）。"""
    if not corners or len(corners) != 4:
        return image
    try:
        import cv2
        import numpy as np
        from PIL import Image
        arr = np.array(image.convert("RGB"))
        src = np.array(corners, dtype="float32")
        w = int(max(_dist(corners[0], corners[1]), _dist(corners[2], corners[3])))
        h = int(max(_dist(corners[0], corners[3]), _dist(corners[1], corners[2])))
        if w <= 1 or h <= 1:
            return image
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
        warped = cv2.warpPerspective(arr, cv2.getPerspectiveTransform(src, dst), (w, h))
        return Image.fromarray(warped)
    except Exception:  # noqa: BLE001
        return image


def enhance(image):
    """轻量增强：LAB 空间 CLAHE 对比度均衡 + 非锐化掩模锐化。cv2 不可用 / 失败 → 原样返回。"""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        arr = np.array(image.convert("RGB"))
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        arr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        blur = cv2.GaussianBlur(arr, (0, 0), 3)
        arr = cv2.addWeighted(arr, 1.5, blur, -0.5, 0)
        return Image.fromarray(arr)
    except Exception:  # noqa: BLE001
        return image


def _dist(p, q) -> float:
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
