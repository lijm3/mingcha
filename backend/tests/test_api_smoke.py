"""后端端到端冒烟测试：mock 掉进程池 worker（不真跑内核/不烧 token），
验证 建任务 → SSE 进度 → 取结果 的完整生命周期，以及 token 鉴权 / 路径穿越防护。
"""
import json
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # 把任务产物根目录指到临时目录，避免污染真实 ./data
    monkeypatch.setenv("MINGCHA_DATA_DIR", str(tmp_path))
    import importlib
    from app import settings as st
    importlib.reload(st)
    from app import task_manager as tm
    importlib.reload(tm)

    # 关键：把 TaskManager 的进程池执行换成同线程内联执行的假 worker，
    # 直接把事件塞进该任务的 mp Queue，drain 协程照常消费。
    def fake_start(self, rec, payload):
        q = rec.event_queue
        q.put({"type": "state", "state": "extracting", "progress": 0.4,
               "stage_note": "去重后 3 帧"})
        # 落一个假的 answer.json，供 /answer 读取
        (rec.out_dir / "answer.json").write_text(json.dumps({
            "intent": "SUMMARY", "answer": "这是一段测试摘要。",
            "topic": "测试主题", "segments": ["段一"], "key_points": ["要点一"],
            "evidence": [], "confidence": 0.8, "caveats": "",
        }, ensure_ascii=False), encoding="utf-8")
        q.put({"type": "done", "intent": "SUMMARY", "caveats": ""})
        self._loop.create_task(self._drain(rec))

    monkeypatch.setattr(tm.TaskManager, "start", fake_start)

    from app import main as main_mod
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def test_create_url_task_and_get_answer(client):
    r = client.post("/api/tasks", json={"url": "https://example.com/v.mp4",
                                        "prompt": "总结"})
    assert r.status_code == 202
    body = r.json()
    tid, token = body["task_id"], body["task_token"]

    # 轮询状态直到 done（fake worker 很快）
    for _ in range(50):
        s = client.get(f"/api/tasks/{tid}?token={token}").json()
        if s["state"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert s["state"] == "done", s

    a = client.get(f"/api/tasks/{tid}/answer?token={token}")
    assert a.status_code == 200
    ans = a.json()
    assert ans["intent"] == "SUMMARY"
    assert ans["summary_detail"]["topic"] == "测试主题"


def test_token_required(client):
    r = client.post("/api/tasks", json={"url": "https://x/v.mp4", "prompt": "总结"})
    tid = r.json()["task_id"]
    assert client.get(f"/api/tasks/{tid}").status_code == 422  # 缺 token
    assert client.get(f"/api/tasks/{tid}?token=wrong").status_code == 403


def test_artifact_path_traversal_blocked(client):
    r = client.post("/api/tasks", json={"url": "https://x/v.mp4", "prompt": "总结"})
    tid, token = r.json()["task_id"], r.json()["task_token"]
    for _ in range(50):
        s = client.get(f"/api/tasks/{tid}?token={token}").json()
        if s["state"] == "done":
            break
        time.sleep(0.05)
    # 越权路径应被拒（穿越到上级目录）
    bad = client.get(f"/artifacts/{tid}/../../../etc/passwd?token={token}")
    assert bad.status_code in (403, 404)
    # 正常产物可取
    ok = client.get(f"/artifacts/{tid}/answer.json?token={token}")
    assert ok.status_code == 200
