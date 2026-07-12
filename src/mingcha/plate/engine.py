"""HyperLPR3 引擎单例 —— detect 与 ocr 共享同一个 LicensePlateCatcher（检测+识别一体）。

为什么必须共享（否则 WinError 32）：HyperLPR3 首次运行会把模型权重下载并解压到
~/.hyperlpr3/（如 20230229.zip）。若 detect 与 ocr 各建一个实例，会**两次**触发下载/解压
同一个 zip——在 Windows 上第二次访问正被占用的 zip 即 PermissionError [WinError 32]
「另一个程序正在使用此文件」。共享单例后权重只下载/解压一次，既消除该冲突，也避免重复加载
模型（省时省显存；HyperLPR3 本是检测+识别一体，一个实例足矣）。
"""
from __future__ import annotations

import threading

_CATCHER = None
_LOCK = threading.Lock()


def get_catcher(gpu: bool = True):
    """返回全局唯一的 HyperLPR3 LicensePlateCatcher（检测+识别一体）。

    首次调用时加载模型（并可能下载权重）；双检锁确保多线程下也只初始化一次。gpu 仅作意图
    声明——实际是否走 GPU 由 onnxruntime 的 provider 决定（装 onnxruntime-gpu 且 CUDA 可用则
    自动启用），LicensePlateCatcher 构造本身不接该参数。缺 [plate] → 惰性 import 抛 ImportError。
    """
    global _CATCHER
    if _CATCHER is not None:
        return _CATCHER
    with _LOCK:
        if _CATCHER is None:
            import hyperlpr3 as lpr3
            from .. import config
            # 小/远车牌（监控俯拍等）用 HIGH 档召回明显更好；LOW 更快。可 config 切换。
            level = (lpr3.DETECT_LEVEL_HIGH
                     if str(getattr(config, "PLATE_DETECT_LEVEL", "high")).lower() == "high"
                     else lpr3.DETECT_LEVEL_LOW)
            _CATCHER = lpr3.LicensePlateCatcher(
                inference=lpr3.INFER_ONNX_RUNTIME, detect_level=level)
    return _CATCHER
