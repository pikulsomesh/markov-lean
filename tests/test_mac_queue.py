import io
import json
import signal
import subprocess
import threading
import time
from pathlib import Path

import yaml

from markov_lean import mac_queue as mq


def make(tmp_path: Path, stages: list[dict]) -> mq.Queue:
    cfg = tmp_path / "queue.yaml"
    cfg.write_text(yaml.safe_dump({"dir": "q", "stages": stages}))
    return mq.Queue.load(cfg, root=tmp_path)


def run(q, procs=lambda: [], **kw):
    out = io.StringIO()
    r = mq.Runner(q, out=out, procs=procs, sleep=lambda s: None, **kw)
    return r.run(), out.getvalue(), q.read_state()["stages"]


def test_order_done_and_skip_on_rerun(tmp_path):
    q = make(tmp_path, [
        {"name": "a", "run": "echo a >> order.txt; touch a.done", "done_when": ["a.done"]},
        {"name": "b", "run": "echo b >> order.txt", "after": ["a"]},
    ])
    rc, out, st = run(q)
    assert rc == 0 and [st[n]["state"] for n in "ab"] == ["done", "done"]
    assert (tmp_path / "order.txt").read_text().split() == ["a", "b"]
    assert "attempt 1" in (tmp_path / "q/logs/01_a.log").read_text()
    rc, _, st = run(q)                      # a rerun runs nothing again
    assert rc == 0 and (tmp_path / "order.txt").read_text().split() == ["a", "b"]
    assert st["a"]["attempts"] == 1


def test_failure_does_not_block_independent_stages(tmp_path):
    q = make(tmp_path, [
        {"name": "w", "wait_for": ["nothing-like-this"]},
        {"name": "bad", "run": "exit 3", "on_fail": "touch cleaned", "after": ["w"]},
        {"name": "dep", "run": "true", "after": ["bad"]},
        {"name": "next", "run": "true", "after": ["w"]},
    ])
    rc, out, st = run(q)
    assert rc == 1
    assert st["w"]["state"] == "done"
    assert st["bad"]["state"] == "failed" and st["bad"]["exit_code"] == 3
    assert (tmp_path / "cleaned").exists()
    assert st["dep"]["state"] == "blocked"
    assert st["next"]["state"] == "done"
    # fix the stage and start again: only the failed and blocked stages run
    q.stages[1].run = "true"
    rc, _, st = run(q)
    assert rc == 0 and st["bad"]["attempts"] == 2 and st["dep"]["state"] == "done"
    assert st["next"]["attempts"] == 1


def test_incomplete_when_checks_fail(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "s/x.jsonl").write_text("{}\n{}\n")
    q = make(tmp_path, [
        {"name": "search", "run": "true", "done_when": [{"glob": "s/*.jsonl", "files": 1, "rows": 3}]},
    ])
    rc, _, st = run(q)
    assert rc == 1 and st["search"]["state"] == "incomplete" and "rows [2]" in st["search"]["reason"]
    (tmp_path / "s/x.jsonl").write_text("{}\n{}\n{}\n")
    rc, _, st = run(q)
    assert rc == 0 and st["search"]["state"] == "done"


def test_wait_stage_waits_for_the_process(tmp_path):
    q = make(tmp_path, [{"name": "w", "wait_for": ["search_eval", "--benchmark putnam"], "poll": 0},
                        {"name": "a", "run": "true", "after": ["w"]}])
    calls = {"n": 0}

    def procs():
        calls["n"] += 1
        busy = [(4242, "python -m markov_lean.search_eval --benchmark putnam --yes")]
        return busy if calls["n"] <= 3 else [(1, "launchd")]
    rc, _, st = run(q, procs=procs)
    assert rc == 0 and calls["n"] == 5 and st["w"]["state"] == "done" and st["a"]["state"] == "done"


def test_wait_gives_up_after_max_hours(tmp_path):
    q = make(tmp_path, [{"name": "w", "wait_for": ["x"], "max_hours": 1, "poll": 0},
                        {"name": "a", "run": "true", "after": ["w"]}])
    t = iter(range(0, 10**7, 1000))
    rc, _, st = run(q, procs=lambda: [(9, "x")], clock=lambda: next(t))
    assert st["w"]["state"] == "failed" and st["a"]["state"] == "blocked"


def test_disk_check(tmp_path, monkeypatch):
    monkeypatch.setattr(mq, "free_gb", lambda p: 5.0)
    q = make(tmp_path, [{"name": "big", "run": "touch ran", "min_disk_gb": 15}])
    rc, _, st = run(q)
    assert st["big"]["state"] == "failed" and "needs 15 GB" in st["big"]["reason"]
    assert not (tmp_path / "ran").exists()


def test_stale_running_stage_is_rerun(tmp_path):
    q = make(tmp_path, [{"name": "a", "run": "touch ran"}])
    q.write_state({"queue": {"pid": 1}, "stages": {"a": {"state": "running", "attempts": 1, "child_pid": 999999}}})
    rc, out, st = run(q)
    assert rc == 0 and st["a"]["state"] == "done" and st["a"]["attempts"] == 2 and (tmp_path / "ran").exists()


def test_second_copy_only_reports(tmp_path):
    q = make(tmp_path, [{"name": "a", "run": "true"}])
    lk = mq.Lock(q.dir / "queue.lock")
    assert lk.take()
    try:
        rc, out, st = run(q)
        assert rc == 0 and "already running" in out and st["a"]["state"] == "pending"
    finally:
        lk.drop()


def test_stop_interrupts_and_marks_for_resume(tmp_path):
    q = make(tmp_path, [{"name": "slow", "run": "trap 'echo got-int; exit 130' INT; touch started; "
                                                 "while true; do sleep 0.1; done"},
                        {"name": "after", "run": "touch later"}])
    r = mq.Runner(q, out=io.StringIO(), procs=lambda: [])
    th = threading.Thread(target=lambda: setattr(r, "rc", r._run()))
    th.start()
    for _ in range(100):
        if (tmp_path / "started").exists():
            break
        time.sleep(0.05)
    r._signal(signal.SIGTERM, None)
    th.join(30)
    st = q.read_state()["stages"]
    assert r.rc == 130 and st["slow"]["state"] == "interrupted" and st["after"]["state"] == "pending"
    assert "got-int" in (tmp_path / "q/logs/01_slow.log").read_text()


def test_status_and_marks(tmp_path):
    q = make(tmp_path, [{"name": "a", "run": "exit 1", "status": "echo progress-line"},
                        {"name": "b", "run": "true"}])
    run(q)
    text = mq.status_text(q)
    assert "Queue not running" in text and "failed" in text and "progress-line" in text
    assert "exited with 1" in text
    mq.set_state(q, "a", "skipped", "by hand")
    assert "skipped" in mq.status_text(q)


def test_secrets_are_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    q = make(tmp_path, [{"name": "a", "run": "echo \"[${ANTHROPIC_API_KEY:-}][$X]\" > env.txt", "env": {"X": "1"}}])
    run(q)
    assert (tmp_path / "env.txt").read_text().strip() == "[][1]"


def test_repo_queue_config_loads():
    q = mq.Queue.load(Path(__file__).resolve().parents[1] / "configs/mac_queue.yaml",
                      root=Path(__file__).resolve().parents[1])
    names = [s.name for s in q.stages]
    assert names == ["putnam_wait", "sphere_packing", "pnt", "carleson", "minif2f_b64"]
    assert all(s.after == ["putnam_wait"] for s in q.stages[1:])
    root = Path(__file__).resolve().parents[1]
    for s in q.stages[1:4]:
        cfg = s.run.split()[-1]
        assert (root / cfg).exists() and s.on_fail and s.min_disk_gb >= 15
    assert (root / q.stages[4].env["CONFIG"]).exists()
    assert subprocess.run(["bash", "-n", str(root / "scripts/mac_queue.sh")]).returncode == 0
