"""主编排 —— Orchestrator.ask()：校验 → 分类 → 规划 → 预处理 → 分析 → 组装。
对应设计文档 §9。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

from . import assembler, intent as intent_mod, planner, preprocess
from .types import Answer, Intent

_CACHE_FILE = ".mingcha_cache.json"


class Orchestrator:
    def __init__(self, *, provider: str | None = None, vision_model: str | None = None,
                 classify_model: str | None = None):
        # provider: 一次性把所有角色切到某家（如 'glm'）；分角色 override 优先级更高。
        self.vision_model = vision_model or provider
        self.classify_model = classify_model or provider

    def ask(self, source: str, prompt: str, query_image: str | None = None, *,
            out_dir: str = "mingcha-out", cookies: str | None = None,
            cookies_from_browser: str | None = None, use_cache: bool = True) -> Answer:
        # ⓪ 输入校验（FR-1.5）：源不可达 / 图片不可读 → 明确报错，不静默
        self._validate(source, query_image)

        # ① 分类（FR-2）
        ir = intent_mod.classify(prompt, has_image=query_image is not None,
                                 classify_model=self.classify_model)
        print(f"意图: {[i.value for i in ir.intents]}  target={ir.target or '-'}  ({ir.reason})")

        # ② 规划（FR-3）
        pl = planner.plan(ir.intents)

        # NFR-6：(source, prompt, plan, 模型) 指纹命中且产物完好 → 跳过预处理与分析
        key = self._cache_key(source, prompt, query_image, pl)
        if use_cache:
            cached = self._load_cached(out_dir, key)
            if cached is not None:
                print("命中缓存：复用上次结果（--no-cache 可强制重算）。")
                return cached

        # ③ 预处理：取视频/转写/音频 + 明察自写的时间戳链路
        pre = preprocess.run(source, out_dir, pl, cookies=cookies,
                             cookies_from_browser=cookies_from_browser)
        print(f"预处理: {pre.frame_count} 帧（去重自 {pre.extracted}）| 拼图 {len(pre.grids)} | "
              f"转写 {pre.transcript_note}")
        if pre.caveats:
            print(f"  ⚠ {pre.caveats}")

        # FR-1.3：把参考图复制进产物目录，answer 里回显其归档路径
        query_copy = None
        if query_image:
            query_copy = os.path.join(pre.out_dir, "query_image.jpg")
            shutil.copy(query_image, query_copy)

        # ④ 分析（按意图分发）
        ans = self._dispatch(ir, pl, pre, prompt, query_copy or query_image)
        if pre.caveats and not ans.caveats:
            ans.caveats = pre.caveats

        # ⑤ 组装 + 落盘 answer.json（FR-6）+ 写缓存指纹
        result = assembler.write(ans, pre.out_dir)
        self._save_cache(pre.out_dir, key)
        return result

    # ---- NFR-6 缓存复用 ----
    @staticmethod
    def _cache_key(source, prompt, query_image, plan) -> str:
        """对影响结果的输入取指纹：源（本地文件含 mtime+size）、prompt、参考图、plan、模型。"""
        parts = [str(source), prompt or ""]
        for p in (source if not str(source).startswith(("http://", "https://")) else None,
                  query_image):
            if p and os.path.exists(p):
                st = os.stat(p)
                parts.append(f"{p}:{int(st.st_mtime)}:{st.st_size}")
        parts.append(plan.model_dump_json())
        h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        return h[:16]

    def _load_cached(self, out_dir: str, key: str) -> Answer | None:
        meta = os.path.join(out_dir, _CACHE_FILE)
        ans_path = os.path.join(out_dir, "answer.json")
        if not (os.path.exists(meta) and os.path.exists(ans_path)):
            return None
        try:
            if json.load(open(meta, encoding="utf-8")).get("key") != key:
                return None
            return Answer.model_validate_json(open(ans_path, encoding="utf-8").read())
        except Exception:  # noqa: BLE001 —— 缓存损坏就当未命中，重算
            return None

    @staticmethod
    def _save_cache(out_dir: str, key: str) -> None:
        try:
            with open(os.path.join(out_dir, _CACHE_FILE), "w", encoding="utf-8") as fh:
                json.dump({"key": key}, fh)
        except OSError:
            pass

    @staticmethod
    def _validate(source: str, query_image: str | None) -> None:
        """FR-1.5 主动输入校验：本地源不存在 / 参考图不可读 → 抛可读错误。
        URL 的可达性交由 media.fetch_video 的下载失败兜底（那里已有明确报错）。"""
        if not source or not str(source).strip():
            raise ValueError("未提供视频源（source 为空）。")
        if not str(source).startswith(("http://", "https://")):
            if not os.path.exists(source):
                raise FileNotFoundError(f"视频源不存在: {source}")
        if query_image is not None:
            if not os.path.exists(query_image):
                raise FileNotFoundError(f"参考图不存在: {query_image}")
            try:
                from PIL import Image
                with Image.open(query_image) as im:
                    im.verify()
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"参考图无法读取或不是有效图片: {query_image}（{e}）") from e

    def _dispatch(self, ir, pl, pre, prompt, query_image) -> Answer:
        """按意图分发；多意图（如 SUMMARY+MODERATE）分别分析后合并为一个 Answer（§8）。"""
        from .analyzer import locate, moderate, summary, visual

        intents = list(dict.fromkeys(ir.intents))  # 去重保序
        target = ir.target or prompt
        vm = self.vision_model

        answers: list[Answer] = []
        for it in intents:
            if it == Intent.VISUAL_LOCATE and query_image:
                answers.append(visual.analyze(pre.out_dir, pl, query_image, pre.video,
                                              vision_model=vm))
            elif it == Intent.LOCATE:
                answers.append(locate.analyze(pre.out_dir, pl, target, pre.video,
                                             vision_model=vm))
            elif it == Intent.MODERATE:
                answers.append(moderate.analyze(pre.out_dir, pl, target, vision_model=vm))
            elif it == Intent.SUMMARY:
                answers.append(summary.analyze(pre.out_dir, pl, prompt, vision_model=vm))

        if not answers:  # 例如仅 VISUAL_LOCATE 却无参考图 → 退化为 SUMMARY
            answers.append(summary.analyze(pre.out_dir, pl, prompt, vision_model=vm))

        return answers[0] if len(answers) == 1 else self._merge_answers(answers)

    @staticmethod
    def _merge_answers(answers: list[Answer]) -> Answer:
        """把多个子 Answer 合并：intent 写组合名，answer 分段拼接，evidence/caveats 并集。"""
        body = "\n\n".join(f"【{a.intent}】\n{a.answer}" for a in answers)
        evidence = [e for a in answers for e in a.evidence]
        caveats = "  ".join(dict.fromkeys(a.caveats for a in answers if a.caveats))
        return Answer(
            intent="+".join(a.intent for a in answers),
            target=next((a.target for a in answers if a.target), None),
            query_image=next((a.query_image for a in answers if a.query_image), None),
            answer=body, evidence=evidence,
            confidence=max((a.confidence for a in answers), default=0.0),
            caveats=caveats, artifacts_dir=answers[0].artifacts_dir)
