"""The memory guard that shares the Mac Mini with its GitHub runners (markov_lean/mac_guard.py).

The machine readings are faked; the guarded job is a real child process (a small Python
script), so the stop, the restart and the escalation run for real, in a few seconds.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import textwrap
import time
from pathlib import Path

import pytest

from markov_lean import mac_guard as mg

pytestmark = pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX only")

IDLE = mg.Snapshot(available_gb=12.0, swap_used_gb=0.5, pressure=1, runner_busy=False)
BUSY = mg.Snapshot(available_gb=12.0, swap_used_gb=0.5, pressure=1, runner_busy=True)

CHILD = textwrap.dedent("""
    import os, signal, sys, time
    from pathlib import Path
    d = Path(sys.argv[1])
    n = len(list(d.glob("start_*"))) + 1
    (d / f"start_{n}").write_text(" ".join(sys.argv[2:]))
    finish_on = int(os.environ.get("FINISH_ON", "2"))
    if os.environ.get("STUBBORN"):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if os.environ.get("FAIL"):
        sys.exit(3)
    if n >= finish_on:
        sys.exit(0)
    try:
        time.sleep(120)
    except KeyboardInterrupt:
        (d / f"stopped_{n}").write_text("ctrl-c")
        sys.exit(1)
""")


class Probes:
    """Readings in order; after a reading's count runs out, the next one; the last repeats."""

    def __init__(self, *plan: tuple[mg.Snapshot, int]) -> None:
        self.plan = [list(p) for p in plan]
        self.calls = 0

    def __call__(self) -> mg.Snapshot:
        self.calls += 1
        while len(self.plan) > 1 and self.plan[0][1] <= 0:
            self.plan.pop(0)
        self.plan[0][1] -= 1
        return self.plan[0][0]


@pytest.fixture
def job(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    work = tmp_path / "work"
    work.mkdir()
    return [sys.executable, str(script), str(work)], work


def _guard(cmd, probes, tmp_path, **kw):
    lines: list[str] = []
    kw.setdefault("limits", mg.Limits(cooldown=0.0))
    g = mg.Guard(cmd, poll=0.05, grace=5.0, probe_fn=probes, log=lines.append,
                 state_path=tmp_path / "state.json", lock_path=tmp_path / "guard.lock", **kw)
    return g, lines


def test_parsers_read_macos_and_linux_output():
    vm = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
          "Pages free:                               65536.\nPages active:                 300000.\n"
          "Pages inactive:                           131072.\nPages speculative:                 65536.\n")
    assert mg.parse_vm_stat(vm) == pytest.approx(4.0)   # (65536 + 131072 + 65536) pages of 16 KB
    assert mg.parse_vm_stat("nonsense") is None
    assert mg.parse_swapusage("total = 2048.00M  used = 1536.00M  free = 512.00M  (encrypted)") == pytest.approx(1.5)
    assert mg.parse_swapusage("total = 0.00M  used = 0.00M  free = 0.00M") == 0.0
    avail, swap = mg.parse_meminfo("MemTotal: 16000000 kB\nMemAvailable: 8388608 kB\n"
                                   "SwapTotal: 2097152 kB\nSwapFree: 1048576 kB\n")
    assert avail == pytest.approx(8.0) and swap == pytest.approx(1.0)
    ps = "/usr/bin/foo\n/Users/s/actions-runner/bin/Runner.Listener run\n"
    assert not mg.runner_job_running(ps)   # the always-on listener is not a job
    assert mg.runner_job_running(ps + "/Users/s/actions-runner/bin/Runner.Worker spawnclient 1 2\n")
    assert isinstance(mg.probe(yield_to_runner=False), mg.Snapshot)   # the real reading runs anywhere


def test_policy_stops_and_waits_for_the_right_reasons():
    lim = mg.Limits(min_free_gb=1.0, resume_free_gb=10.0, max_swap_growth_gb=1.5)
    assert mg.stop_reason(IDLE, lim, 0.5) is None
    assert "runner" in mg.stop_reason(BUSY, lim, 0.5)
    assert "pressure" in mg.stop_reason(mg.Snapshot(12.0, 0.5, 2, False), lim, 0.5)
    assert "available" in mg.stop_reason(mg.Snapshot(0.8, 0.5, 1, False), lim, 0.5)
    assert "swap" in mg.stop_reason(mg.Snapshot(5.0, 2.5, 1, False), lim, 0.5)
    assert mg.stop_reason(mg.Snapshot(5.0, 2.5, 1, False), lim, 2.0) is None   # old swap is not growth
    assert mg.stop_reason(mg.Snapshot(None, None, None, False), lim, None) is None   # unknown readings
    assert mg.wait_reason(IDLE, lim) is None
    assert "runner" in mg.wait_reason(BUSY, lim)
    assert "waiting for 10.0" in mg.wait_reason(mg.Snapshot(6.0, 0.5, 1, False), lim)   # room to resume


def test_a_runner_job_stops_the_job_and_it_resumes_after(job, tmp_path):
    cmd, work = job
    probes = Probes((IDLE, 3), (BUSY, 5), (IDLE, 10_000))
    g, lines = _guard(cmd, probes, tmp_path, hours=1.0, hours_flag="--max-hours")
    assert g.run() == 0
    assert (work / "stopped_1").read_text() == "ctrl-c"      # stopped like Ctrl-C, finished work kept
    assert (work / "start_2").exists() and not (work / "start_3").exists()
    assert "--max-hours" in (work / "start_2").read_text()   # each start gets the hours left
    st = json.loads((tmp_path / "state.json").read_text())
    assert st["status"] == "done" and st["starts"] == 2 and st["yields"] == 1
    assert any("runner job started" in l for l in lines)


def test_it_waits_for_a_runner_job_and_memory_before_the_first_start(job, tmp_path):
    cmd, work = job
    low = mg.Snapshot(available_gb=4.0, swap_used_gb=0.5, pressure=1, runner_busy=False)
    probes = Probes((BUSY, 4), (low, 4), (IDLE, 10_000))
    g, lines = _guard(cmd, probes, tmp_path, limits=mg.Limits(cooldown=0.0, resume_free_gb=10.0))
    os.environ["FINISH_ON"] = "1"
    try:
        assert g.run() == 0
    finally:
        del os.environ["FINISH_ON"]
    assert probes.calls >= 9 and (work / "start_1").exists()
    assert any("runner job is running" in l for l in lines) and any("waiting for 10.0" in l for l in lines)


def test_cooldown_holds_the_restart(job, tmp_path):
    cmd, work = job
    probes = Probes((IDLE, 2), (BUSY, 2), (IDLE, 10_000))
    g, _ = _guard(cmd, probes, tmp_path, limits=mg.Limits(cooldown=0.6))
    t = time.monotonic()
    assert g.run() == 0
    assert time.monotonic() - t >= 0.6 and (work / "start_2").exists()


@pytest.mark.parametrize("bad", [mg.Snapshot(12.0, 0.5, 2, False), mg.Snapshot(0.5, 0.5, 1, False),
                                 mg.Snapshot(12.0, 3.0, 1, False)], ids=["pressure", "low memory", "swap"])
def test_memory_trouble_stops_the_job(job, tmp_path, bad):
    cmd, work = job
    g, lines = _guard(cmd, Probes((IDLE, 2), (bad, 1), (IDLE, 10_000)), tmp_path)
    assert g.run() == 0
    assert (work / "stopped_1").exists() and (work / "start_2").exists()
    assert g.state.yields == 1


def test_a_job_that_ignores_ctrl_c_is_terminated_then_killed(job, tmp_path, monkeypatch):
    cmd, work = job
    monkeypatch.setenv("STUBBORN", "1")
    g, lines = _guard(cmd, Probes((IDLE, 2), (BUSY, 10_000)), tmp_path, hours=0.0005)   # ~2 s
    g.grace = g.term_wait = 0.3
    t = time.monotonic()
    rc = g.run()
    assert rc == 0 and time.monotonic() - t < 60
    assert any("SIGTERM" in l for l in lines) and any("killing it" in l for l in lines)
    assert g.state.status == "deadline" and g.child.poll() is not None


def test_repeated_failures_give_up_with_the_exit_code(job, tmp_path, monkeypatch):
    cmd, work = job
    monkeypatch.setenv("FAIL", "1")
    g, lines = _guard(cmd, Probes((IDLE, 10_000)), tmp_path, max_failures=3)
    assert g.run() == 3
    assert len(list(work.glob("start_*"))) == 3 and g.state.status == "gave_up"


def test_guarded_jobs_take_turns(job, tmp_path):
    cmd, work = job
    lock = tmp_path / "guard.lock"
    other = mg.Guard(["true"], lock_path=lock, log=lambda m: None)
    assert other._take_lock()                 # another guarded job holds the lock
    g, lines = _guard(cmd, Probes((IDLE, 10_000)), tmp_path, hours=0.0003)
    assert g.run() == 0 and not (work / "start_1").exists()
    assert g.state.status == "deadline" and any("another guarded job" in l for l in lines)
    other._drop_lock()
    os.environ["FINISH_ON"] = "1"
    try:
        g2, _ = _guard(cmd, Probes((IDLE, 10_000)), tmp_path)
        assert g2.run() == 0 and (work / "start_1").exists()
    finally:
        del os.environ["FINISH_ON"]


def test_ctrl_c_on_the_guard_stops_the_job(job, tmp_path):
    cmd, work = job

    class Interrupting(Probes):
        def __call__(self):
            snap = super().__call__()
            if self.calls == 4:
                raise KeyboardInterrupt
            return snap

    g, _ = _guard(cmd, Interrupting((IDLE, 10_000)), tmp_path)
    assert g.run() == 130
    assert (work / "stopped_1").exists() and g.child.poll() is not None
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "interrupted"


def test_cli_status_and_command(tmp_path, job, capsys):
    cmd, work = job
    state = tmp_path / "s.json"
    os.environ["FINISH_ON"] = "1"
    try:
        rc = mg.main(["--state", str(state), "--lock", str(tmp_path / "l"), "--poll", "0.05",
                      "--min-free-gb", "0", "--resume-free-gb", "0", "--max-swap-growth-gb", "1000",
                      "--no-runner-yield", "--", *cmd])
    finally:
        del os.environ["FINISH_ON"]
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
    assert rc == 0 and (work / "start_1").exists()
    capsys.readouterr()
    assert mg.main(["--status", "--state", str(state)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("done") and "starts 1" in out
