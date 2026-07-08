"""明察 CLI —— `mingcha ask` / `mc ask`。对应设计文档 §10。"""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .orchestrator import Orchestrator


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="mingcha", description="明察 —— 看懂/看准/看住视频的多模型 AI 智能体")
    parser.add_argument("--version", action="version", version=f"mingcha {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    ask = sub.add_parser("ask", help="对视频提问")
    ask.add_argument("source", help="视频 URL 或本地路径")
    ask.add_argument("prompt", nargs="?", default="", help="自然语言要求")
    ask.add_argument("--image", default=None, help="参考图（VISUAL_LOCATE 以图搜视频）")
    ask.add_argument("-o", "--out", default="mingcha-out", help="输出目录（覆盖式）")
    ask.add_argument("--provider", default=None, help="一次性切换所有角色到某 provider（如 glm）")
    ask.add_argument("--vision-model", default=None, help="画面分析模型，如 openai:gpt-5.5")
    ask.add_argument("--classify-model", default=None, help="意图分类模型")
    ask.add_argument("--cookies", default=None, help="Netscape cookies 文件（本人授权内容）")
    ask.add_argument("--cookies-from-browser", default=None, help="从浏览器取 cookies")
    ask.add_argument("--no-cache", action="store_true",
                     help="强制重算，忽略 (source,prompt,plan) 缓存复用（NFR-6）")

    args = parser.parse_args(argv)
    if args.cmd != "ask":
        parser.print_help()
        return 1
    if not args.prompt and not args.image:
        print("请提供一句话要求，或用 --image 附参考图。", file=sys.stderr)
        return 2

    orch = Orchestrator(provider=args.provider, vision_model=args.vision_model,
                        classify_model=args.classify_model)
    # 上云告知（C-4）
    print(f"明察 v{__version__} —— 关键帧将发送到所选模型服务商进行分析。")
    try:
        ans = orch.ask(args.source, args.prompt, query_image=args.image, out_dir=args.out,
                       cookies=args.cookies, cookies_from_browser=args.cookies_from_browser,
                       use_cache=not args.no_cache)
    except Exception as e:  # noqa: BLE001
        print(f"\n✗ 失败: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print(f"意图: {ans.intent}")
    print(f"答案:\n{ans.answer}")
    if ans.evidence:
        print("\n证据:")
        for e in ans.evidence:
            print(f"  - {e.frame} @ {e.hms}  (conf={e.confidence})  {e.note}")
    if ans.caveats:
        print(f"\n局限: {ans.caveats}")
    print(f"\nanswer.json: {os.path.join(ans.artifacts_dir, 'answer.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
