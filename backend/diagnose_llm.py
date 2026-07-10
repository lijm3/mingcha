"""LLM 端点诊断：用你的真实 key 发一次最小的 vision 请求并计时。

用法（在项目根目录）：
    # claude（默认 vision provider）
    MINGCHA_LOG_LEVEL=DEBUG python backend/diagnose_llm.py claude sk-你的key

    # 你的 glm 端点
    MINGCHA_LOG_LEVEL=DEBUG python backend/diagnose_llm.py glm sk-你的glmkey

会打印每一步耗时。若卡在 "发起请求…" 很久，说明是模型端点/代理在真实生成时不返回，
不是明察的问题——换个端点或 key。
"""
import sys
import time

from mingcha import config
from mingcha.types import VisualHitSchema
import mingcha.llm as llm
from mingcha.llm.base import ImageRef


def main():
    if len(sys.argv) < 3:
        print("用法: python backend/diagnose_llm.py <provider claude|openai|glm> <api_key>")
        return 1
    provider, key = sys.argv[1], sys.argv[2]

    # 用请求级注入把 key 塞进去（和后端界面填 key 走同一条路）
    config.set_runtime_keys({provider: key})

    # 造一张 1x1 的临时图，避免依赖真实帧
    import base64, tempfile, os
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    tmp = os.path.join(tempfile.gettempdir(), "mc_diag.png")
    with open(tmp, "wb") as f:
        f.write(png_1x1)

    print(f"\n=== 诊断 provider={provider} ===")
    t0 = time.time()
    prov = llm.get_provider("vision", provider)
    print(f"provider: {prov.name}:{prov.model} @ {prov.base_url}")
    print(f"api_key: {'已注入 ' + key[:6] + '…' if prov.api_key else '缺失!'}")
    print("\n发起一次最小 vision 请求（1x1 图 + 一句话，强制结构化）…")
    try:
        obj = llm.vision_structured(
            "vision", "你是测试助手。", [ImageRef(tmp)],
            "只回一个 JSON：{\"verdict\":\"no\",\"similarity\":0,\"confidence\":0,\"note\":\"ok\"}",
            VisualHitSchema, override=provider)
        print(f"\n✅ 成功！耗时 {time.time()-t0:.1f}s  返回: {obj}")
    except Exception as e:
        print(f"\n❌ 失败（耗时 {time.time()-t0:.1f}s）: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
