"""Run a long, resumable job on the Mac Mini and give its memory back when something else needs it.

``python -m markov_lean.mac_guard [options] -- <command> [args...]``

The Mac Mini (16 GB) also hosts self-hosted GitHub Actions runners. The search
evaluation holds about 9 GB (the 4-bit prover, its KV cache and Lean with Mathlib) and
the local judge about 7 GB, so a runner job starting next to either could push macOS
into swap or kill something. This guard runs the job as a child process and, every few
seconds, looks at:

* **runner jobs**: a ``Runner.Worker`` process is running (the GitHub runner starts one
  per job; its always-on ``Runner.Listener`` does not count);
* **memory pressure**: ``kern.memorystatus_vm_pressure_level`` (1 normal, 2 warn,
  4 critical);
* **available memory**: free + inactive + speculative pages from ``vm_stat``;
* **swap**: how much ``vm.swapusage`` has grown since the job started.

Guarded jobs also take turns: a lock file (``--lock``) lets one run at a time, so the
search and the local judge, which do not fit in 16 GB together, never start side by side.

When a runner job starts, pressure reaches *warn*, available memory falls under
``--min-free-gb`` or swap grows by more than ``--max-swap-growth-gb``, it stops the job
cleanly (SIGINT, as Ctrl-C would, then SIGTERM and SIGKILL if it hangs), which frees all
of its memory; pausing a process with SIGSTOP would not. Both jobs it is meant for are
resumable at the level of one problem or one proof, so a stop loses at most the item in
flight. Once no runner job has run for ``--cooldown`` seconds, pressure is normal and at
least ``--resume-free-gb`` is available, it starts the same command again, which resumes.

The guard ends when the job exits 0 on its own (finished, or its own time limit), when
``--hours`` have passed, after ``--max-failures`` failed starts in a row, or on Ctrl-C
(which it passes on). It changes no system setting, sends nothing over the network and
never touches the runner. Its state is in ``--state`` (JSON) for ``--status``.

On Linux (CI, or anywhere else) it reads ``/proc/meminfo`` and has no pressure level.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

GB = 1024 ** 3
RUNNER_JOB = "Runner.Worker"
DEFAULT_LOCK = Path.home() / ".cache" / "markov-lean" / "guard.lock"

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


# ---------------------------------------------------------------------- readings


@dataclass
class Snapshot:
    available_gb: float | None      # memory the system can hand out without swapping
    swap_used_gb: float | None
    pressure: int | None            # 1 normal, 2 warn, 4 critical (macOS only)
    runner_busy: bool

    def line(self) -> str:
        def f(x: float | None) -> str:
            return "?" if x is None else f"{x:.1f}"
        p = {None: "n/a", 1: "normal", 2: "warn", 4: "critical"}.get(self.pressure, str(self.pressure))
        return (f"available {f(self.available_gb)} GB, swap {f(self.swap_used_gb)} GB, pressure {p}, "
                f"runner job {'running' if self.runner_busy else 'none'}")


def _run(cmd: Sequence[str]) -> str | None:
    try:
        return subprocess.run(list(cmd), capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def parse_vm_stat(text: str) -> float | None:
    """Available GB from ``vm_stat``: free + inactive + speculative pages."""
    m = re.search(r"page size of (\d+) bytes", text)
    if not m:
        return None
    page = int(m.group(1))
    pages = {}
    for key in ("free", "inactive", "speculative"):
        mm = re.search(rf"Pages {key}:\s+(\d+)", text)
        if mm:
            pages[key] = int(mm.group(1))
    if "free" not in pages:
        return None
    return sum(pages.values()) * page / GB


def parse_swapusage(text: str) -> float | None:
    """Used GB from ``sysctl -n vm.swapusage`` (``total = 2048.00M  used = 1024.50M ...``)."""
    m = re.search(r"used\s*=\s*([\d.]+)([KMGT])", text)
    if not m:
        return None
    scale = {"K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}[m.group(2)]
    return float(m.group(1)) * scale


def parse_meminfo(text: str) -> tuple[float | None, float | None]:
    """(available GB, swap used GB) from Linux ``/proc/meminfo``."""
    vals = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            vals[m.group(1)] = int(m.group(2)) * 1024 / GB
    avail = vals.get("MemAvailable")
    swap = vals["SwapTotal"] - vals["SwapFree"] if "SwapTotal" in vals and "SwapFree" in vals else None
    return avail, swap


def runner_job_running(ps_text: str) -> bool:
    """A GitHub Actions job is running: some process's command names ``Runner.Worker``."""
    return any(RUNNER_JOB in line for line in ps_text.splitlines())


def probe(yield_to_runner: bool = True) -> Snapshot:
    """Read the machine now (macOS, else Linux)."""
    if sys.platform == "darwin":
        avail = parse_vm_stat(_run(["vm_stat"]) or "")
        swap = parse_swapusage(_run(["sysctl", "-n", "vm.swapusage"]) or "")
        level = (_run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]) or "").strip()
        pressure = int(level) if level.isdigit() else None
    else:
        try:
            avail, swap = parse_meminfo(Path("/proc/meminfo").read_text())
        except OSError:
            avail, swap = None, None
        pressure = None
    busy = yield_to_runner and runner_job_running(_run(["ps", "-axo", "command"]) or "")
    return Snapshot(avail, swap, pressure, busy)


# ---------------------------------------------------------------------- policy


@dataclass
class Limits:
    min_free_gb: float = 1.0          # stop the job below this much available memory
    resume_free_gb: float = 10.0      # start (again) only with at least this much available
    max_swap_growth_gb: float = 1.5   # stop the job if swap grew this much since it started
    max_pressure: int = 1             # stop above this level (2 is warn)
    cooldown: float = 120.0           # seconds with no runner job (and room) before starting again


def stop_reason(s: Snapshot, lim: Limits, swap_at_start: float | None) -> str | None:
    """Why the running job must stop now, or None."""
    if s.runner_busy:
        return "a GitHub runner job started"
    if s.pressure is not None and s.pressure > lim.max_pressure:
        return f"memory pressure level {s.pressure}"
    if s.available_gb is not None and s.available_gb < lim.min_free_gb:
        return f"only {s.available_gb:.1f} GB available"
    if (s.swap_used_gb is not None and swap_at_start is not None
            and s.swap_used_gb - swap_at_start > lim.max_swap_growth_gb):
        return f"swap grew by {s.swap_used_gb - swap_at_start:.1f} GB"
    return None


def wait_reason(s: Snapshot, lim: Limits) -> str | None:
    """Why the job can't start yet, or None."""
    if s.runner_busy:
        return "a GitHub runner job is running"
    if s.pressure is not None and s.pressure > 1:
        return f"memory pressure level {s.pressure}"
    if s.available_gb is not None and s.available_gb < lim.resume_free_gb:
        return f"{s.available_gb:.1f} GB available, waiting for {lim.resume_free_gb:.1f}"
    return None


# ---------------------------------------------------------------------- the guard


@dataclass
class State:
    status: str = "starting"          # waiting, running, stopping, done, gave_up, deadline, interrupted
    reason: str = ""
    starts: int = 0
    yields: int = 0
    failures: int = 0
    started_at: str = ""
    updated_at: str = ""
    last: dict = field(default_factory=dict)


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


class Guard:
    def __init__(self, command: Sequence[str], limits: Limits | None = None, hours: float | None = None,
                 hours_flag: str | None = None, poll: float = 10.0, grace: float = 120.0,
                 max_failures: int = 3, min_good_run: float = 600.0, on_pause: str | None = None,
                 state_path: str | Path | None = None, lock_path: str | Path | None = None, probe_fn: Callable[[], Snapshot] | None = None,
                 log: Callable[[str], None] | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.command = list(command)
        self.lim = limits or Limits()
        self.hours, self.hours_flag = hours, hours_flag
        self.poll, self.grace = poll, grace
        self.term_wait = 30.0             # seconds after SIGTERM before SIGKILL
        self.max_failures, self.min_good_run = max_failures, min_good_run
        self.on_pause = on_pause
        self.state_path = Path(state_path) if state_path else None
        self.lock_path = Path(lock_path) if lock_path else None
        self._lock = None
        self.probe = probe_fn or probe
        self.log = log or (lambda m: print(f"{time.strftime('%H:%M:%S')} guard: {m}", flush=True))
        self.clock, self.sleep = clock, sleep
        self.state = State(started_at=_now())
        self.child: subprocess.Popen | None = None
        self._interrupted = False

    # -- bookkeeping
    def _save(self, status: str, reason: str = "", snap: Snapshot | None = None) -> None:
        self.state.status, self.state.reason, self.state.updated_at = status, reason, _now()
        if snap is not None:
            self.state.last = asdict(snap)
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self.state), indent=1) + "\n")
            tmp.replace(self.state_path)

    def _remaining_hours(self, t0: float) -> float | None:
        if self.hours is None:
            return None
        return self.hours - (self.clock() - t0) / 3600

    # -- one guarded job at a time
    def _take_lock(self) -> bool:
        if self.lock_path is None or fcntl is None:
            return True
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.lock_path, "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        f.seek(0)
        f.truncate()
        f.write(f"{os.getpid()} {shlex.join(self.command)}\n")
        f.flush()
        self._lock = f
        return True

    def _drop_lock(self) -> None:
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    # -- the child
    def _start(self, remaining: float | None) -> None:
        cmd = list(self.command)
        if self.hours_flag and remaining is not None:
            cmd += [self.hours_flag, f"{max(remaining, 0.0):.3f}"]
        self.state.starts += 1
        self.child = subprocess.Popen(cmd, start_new_session=True)

    def _signal(self, sig: int) -> None:
        if self.child is None or self.child.poll() is not None:
            return
        try:
            os.killpg(self.child.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _wait(self, seconds: float) -> int | None:
        end = self.clock() + seconds
        while self.child is not None and self.child.poll() is None and self.clock() < end:
            self.sleep(min(0.5, self.poll))
        return None if self.child is None else self.child.poll()

    def stop_child(self) -> int | None:
        """Ctrl-C, then SIGTERM, then SIGKILL, until the job and everything it started are gone."""
        self._signal(signal.SIGINT)
        rc = self._wait(self.grace)
        if rc is None:
            self.log("the job did not stop on Ctrl-C; sending SIGTERM")
            self._signal(signal.SIGTERM)
            rc = self._wait(self.term_wait)
        if rc is None:
            self.log("the job did not stop on SIGTERM; killing it")
            self._signal(signal.SIGKILL)
            rc = self._wait(self.term_wait)
        self._signal(signal.SIGKILL)   # anything left in its process group (the Lean REPL)
        if self.on_pause:
            subprocess.run(self.on_pause, shell=True, timeout=120, check=False)
        return rc

    # -- main loop
    def run(self) -> int:
        t0 = self.clock()
        quiet_since: float | None = None
        last_note = -1e18
        try:
            while True:
                remaining = self._remaining_hours(t0)
                if remaining is not None and remaining <= 0:
                    self._save("deadline", f"{self.hours} h passed")
                    self.log(f"{self.hours} h passed; stopping. Rerun to go on.")
                    return 0
                snap = self.probe()
                why = wait_reason(snap, self.lim)
                if why:
                    quiet_since = None
                    if why != self.state.reason or self.clock() - last_note >= 600:
                        self.log(f"waiting: {why} ({snap.line()})")
                        last_note = self.clock()
                    self._save("waiting", why, snap)
                    self.sleep(self.poll)
                    continue
                if quiet_since is None:
                    quiet_since = self.clock()
                first = self.state.starts == 0 and self.state.yields == 0 and self.state.failures == 0
                if not first and self.clock() - quiet_since < self.lim.cooldown:
                    self._save("waiting", "cooling down", snap)
                    self.sleep(self.poll)
                    continue
                if not self._take_lock():
                    if self.state.reason != "another guarded job is running":
                        self.log("waiting: another guarded job is running")
                    self._save("waiting", "another guarded job is running", snap)
                    self.sleep(self.poll)
                    continue

                self.log(f"starting the job ({snap.line()})")
                self._start(remaining)
                self._save("running", "", snap)
                began, swap0 = self.clock(), snap.swap_used_gb
                why_stop, rc = None, None
                while True:
                    rc = self.child.poll()
                    if rc is not None:
                        break
                    remaining = self._remaining_hours(t0)
                    if remaining is not None and remaining <= -0.05 and not self.hours_flag:
                        why_stop = f"{self.hours} h passed"
                    else:
                        s = self.probe()
                        why_stop = stop_reason(s, self.lim, swap0)
                        if why_stop:
                            snap = s
                    if why_stop:
                        break
                    self.sleep(self.poll)

                if why_stop:
                    self.log(f"stopping the job: {why_stop} ({snap.line()})")
                    self._save("stopping", why_stop, snap)
                    self.stop_child()
                    self._drop_lock()
                    self.state.yields += 1
                    self.log("job stopped; its finished work is saved. It starts again when there is room.")
                    quiet_since, last_note = None, -1e18
                    self._save("waiting", why_stop, snap)
                    continue

                self._drop_lock()
                ran = self.clock() - began
                if rc == 0:
                    self._save("done", "the job finished")
                    self.log("the job finished.")
                    return 0
                if ran >= self.min_good_run:
                    self.state.failures = 0
                self.state.failures += 1
                killed = rc < 0 and -rc == signal.SIGKILL
                note = "killed (likely out of memory)" if killed else f"exited with {rc}"
                if self.state.failures >= self.max_failures:
                    self._save("gave_up", f"{note}, {self.state.failures} times in a row")
                    self.log(f"the job {note}, {self.state.failures} times in a row; giving up. See its log.")
                    return rc if rc > 0 else 1
                self.log(f"the job {note}; trying again after the cooldown")
                quiet_since, last_note = None, -1e18
                self._save("waiting", note)
        except KeyboardInterrupt:
            self._interrupted = True
            self.log("interrupted: stopping the job; rerun the same command to resume")
            self.stop_child()
            self._drop_lock()
            self._save("interrupted", "Ctrl-C")
            return 130


def status_text(state_path: str | Path) -> str:
    p = Path(state_path)
    if not p.exists():
        return f"no guard state at {p}"
    s = json.loads(p.read_text())
    last = s.get("last") or {}
    snap = Snapshot(last.get("available_gb"), last.get("swap_used_gb"), last.get("pressure"),
                    bool(last.get("runner_busy"))) if last else None
    out = [f"{s['status']}" + (f": {s['reason']}" if s.get("reason") else ""),
           f"since {s.get('started_at')}, updated {s.get('updated_at')}",
           f"starts {s.get('starts', 0)}, stops for room {s.get('yields', 0)}, failed starts {s.get('failures', 0)}"]
    if snap:
        out.append(snap.line())
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m markov_lean.mac_guard",
                                description=__doc__.splitlines()[0],
                                usage="%(prog)s [options] -- command [args...]")
    lim = Limits()
    p.add_argument("--hours", type=float, help="stop after this many hours in all (paused time counts)")
    p.add_argument("--hours-flag", help="pass the hours left to each start as this flag, e.g. --hours-flag=--max-hours")
    p.add_argument("--min-free-gb", type=float, default=lim.min_free_gb)
    p.add_argument("--resume-free-gb", type=float, default=lim.resume_free_gb)
    p.add_argument("--max-swap-growth-gb", type=float, default=lim.max_swap_growth_gb)
    p.add_argument("--cooldown", type=float, default=lim.cooldown, help="seconds of quiet before a restart")
    p.add_argument("--poll", type=float, default=10.0, help="seconds between checks")
    p.add_argument("--grace", type=float, default=120.0, help="seconds to wait after Ctrl-C before SIGTERM")
    p.add_argument("--max-failures", type=int, default=3, help="give up after this many failed starts in a row")
    p.add_argument("--no-runner-yield", action="store_true", help="ignore GitHub runner jobs (memory checks only)")
    p.add_argument("--on-pause", help="shell command to run after each stop (e.g. ollama stop <model>)")
    p.add_argument("--state", help="JSON file with the guard's state")
    p.add_argument("--lock", default=str(DEFAULT_LOCK), help="lock file shared by guarded jobs ('' for none)")
    p.add_argument("--status", action="store_true", help="print the state in --state and exit")
    p.add_argument("--check", action="store_true", help="print the current readings and exit")
    p.add_argument("command", nargs=argparse.REMAINDER)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.status:
        if not args.state:
            raise SystemExit("--status needs --state")
        print(status_text(args.state))
        return 0
    if args.check:
        print(probe(not args.no_runner_yield).line())
        return 0
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        raise SystemExit("give the command after --")
    def _as_ctrl_c(signum, frame):  # noqa: ARG001 - signal handler signature
        raise KeyboardInterrupt

    # launchd's unload and a closed terminal stop the job cleanly too, instead of orphaning it
    signal.signal(signal.SIGTERM, _as_ctrl_c)
    signal.signal(signal.SIGHUP, _as_ctrl_c)
    limits = Limits(args.min_free_gb, args.resume_free_gb, args.max_swap_growth_gb, 1, args.cooldown)
    guard = Guard(cmd, limits, hours=args.hours, hours_flag=args.hours_flag, poll=args.poll, grace=args.grace,
                  max_failures=args.max_failures, on_pause=args.on_pause, state_path=args.state,
                  lock_path=args.lock or None,
                  probe_fn=lambda: probe(not args.no_runner_yield))
    guard.log(f"running {shlex.join(cmd)}")
    return guard.run()


if __name__ == "__main__":
    sys.exit(main())
