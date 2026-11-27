"""A fixed queue of long Mac Mini jobs, run strictly one after another over days.

``scripts/mac_queue.sh`` is the front end: it starts ``run`` in the background, keeps the Mac
awake, and has ``--status``, ``--stop``, ``--retry`` and ``--skip``. The queue itself is
``configs/mac_queue.yaml``: an ordered list of stages, each a shell command (or a wait for a
job that is already running) with how to tell that it finished, how to show its progress and,
optionally, how to clean up after a failure.

    python -m markov_lean.mac_queue run    [--config configs/mac_queue.yaml]
    python -m markov_lean.mac_queue status [--config ...]
    python -m markov_lean.mac_queue retry NAME | skip NAME [--config ...]

Rules:

* One stage at a time, in the listed order. The queue holds its own lock, so a second copy
  started by mistake (or at login) only says that the first is running.
* Each stage writes its own log (``<dir>/logs/NN_<name>.log``, appended per attempt) and its
  state, start and end time to ``<dir>/state.json``.
* A stage is **done** when its command exits 0 and its ``done_when`` checks pass. Exit 0 with a
  check failing (a time limit reached with work left) is **incomplete**; a non-zero exit is
  **failed**, and its ``on_fail`` command runs (for an extraction, freeing its build files).
  Neither stops the queue: later stages still run unless they list the stage under ``after``.
* Rerunning ``run`` skips done and skipped stages and runs the rest again; each stage's own
  script resumes from where it stopped. A stage cut off by a reboot, a crash or ``--stop`` is
  **interrupted** and runs again first.
* Before a stage starts, the disk holding the repository must have ``min_disk_gb`` free.

The stages keep their own memory guards (``markov_lean.mac_guard``), which stop them while a
GitHub runner job is active or memory is short. API and AWS credentials are removed from
every stage's environment, so no stage can spend money.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

import yaml

DEFAULT_CONFIG = "configs/mac_queue.yaml"
SECRETS = ("ANTHROPIC_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
           "AWS_PROFILE", "AWS_DEFAULT_PROFILE")
FINAL = ("done", "skipped")
GRACE = 600.0      # seconds a stage gets to stop on Ctrl-C (its guard allows 120 s per job)


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def hours_between(a: str | None, b: str | None) -> float | None:
    if not a:
        return None
    t0 = dt.datetime.strptime(a, "%Y-%m-%d %H:%M")
    t1 = dt.datetime.strptime(b, "%Y-%m-%d %H:%M") if b else dt.datetime.now()
    return (t1 - t0).total_seconds() / 3600


@dataclass
class Stage:
    name: str
    title: str = ""
    run: str | None = None
    wait_for: list[str] = field(default_factory=list)
    status: str | None = None
    done_when: list = field(default_factory=list)
    on_fail: str | None = None
    env: dict = field(default_factory=dict)
    after: list[str] = field(default_factory=list)
    min_disk_gb: float = 0.0
    max_hours: float | None = None
    poll: float = 120.0


@dataclass
class Queue:
    root: Path
    dir: Path
    stages: list[Stage]

    @classmethod
    def load(cls, config: str | Path, root: str | Path | None = None) -> "Queue":
        cfg = yaml.safe_load(Path(config).read_text())
        root = Path(root) if root else Path.cwd()
        stages = [Stage(**s) for s in cfg["stages"]]
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise SystemExit(f"{config}: stage names must be unique")
        for i, s in enumerate(stages):
            if bool(s.run) == bool(s.wait_for):
                raise SystemExit(f"{config}: stage {s.name} needs exactly one of run and wait_for")
            for d in s.after:
                if d not in names[:i]:
                    raise SystemExit(f"{config}: stage {s.name} is after {d}, which is not an earlier stage")
        d = Path(cfg.get("dir", "results/mac_queue"))
        return cls(root, d if d.is_absolute() else root / d, stages)

    # ------------------------------------------------------------------ state
    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    def read_state(self) -> dict:
        st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        st.setdefault("queue", {})
        st.setdefault("stages", {})
        for s in self.stages:
            st["stages"].setdefault(s.name, {"state": "pending", "attempts": 0})
        return st

    def write_state(self, st: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2))
        tmp.replace(self.state_path)

    def log_path(self, i: int, s: Stage) -> Path:
        return self.dir / "logs" / f"{i + 1:02d}_{s.name}.log"

    def stage(self, name: str) -> Stage:
        for s in self.stages:
            if s.name == name:
                return s
        raise SystemExit(f"no stage {name}; stages: {', '.join(x.name for x in self.stages)}")


# ---------------------------------------------------------------------- helpers
def stage_env(s: Stage) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in SECRETS}
    env.update({k: str(v) for k, v in s.env.items()})
    env["PYTHONUNBUFFERED"] = "1"
    return env


def free_gb(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 2**30


def processes() -> list[tuple[int, str]]:
    out = subprocess.run(["ps", "-ww", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid.isdigit():
            rows.append((int(pid), cmd))
    return rows


def matching(patterns: list[str], procs: list[tuple[int, str]] | None = None) -> list[int]:
    me = os.getpid()
    return [pid for pid, cmd in (processes() if procs is None else procs)
            if pid != me and all(p in cmd for p in patterns)]


def check(root: Path, item) -> tuple[bool, str]:
    """One done_when item: a path (glob) that must exist, or {glob, files, rows}."""
    if isinstance(item, str):
        hits = glob.glob(str(root / item))
        return bool(hits), f"{item} {'exists' if hits else 'is missing'}"
    files = sorted(glob.glob(str(root / item["glob"])))
    need_files, need_rows = int(item.get("files", 1)), int(item.get("rows", 1))
    counts = [sum(1 for line in open(f) if line.strip()) for f in files]
    ok = len(files) >= need_files and all(c >= need_rows for c in counts)
    return ok, f"{item['glob']}: {len(files)} of {need_files} files, rows {counts} (need {need_rows} each)"


def checks(root: Path, s: Stage) -> tuple[bool, list[str]]:
    """All done_when checks pass, and the messages of those that do not."""
    res = [check(root, it) for it in s.done_when]
    return all(ok for ok, _ in res), [m for ok, m in res if not ok]


def shown(p: Path, root: Path) -> str:
    """A path relative to the repository where it can be, else as it is."""
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Lock:
    def __init__(self, path: Path):
        self.path, self.f = path, None

    def take(self) -> bool:
        if fcntl is None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.path, "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self.f = f
        return True

    def drop(self) -> None:
        if self.f is not None:
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()
            self.f = None


def held(path: Path) -> bool:
    lk = Lock(path)
    if lk.take():
        lk.drop()
        return False
    return True


# ---------------------------------------------------------------------- runner
class Runner:
    def __init__(self, q: Queue, out=sys.stdout, procs=processes, sleep=time.sleep, clock=time.monotonic):
        self.q, self.out, self.procs, self.sleep, self.clock = q, out, procs, sleep, clock
        self.child: subprocess.Popen | None = None
        self.stop = False

    def say(self, msg: str) -> None:
        print(f"{now()} {msg}", file=self.out, flush=True)

    def _signal(self, signum, _frame) -> None:
        if not self.stop:
            self.say(f"stop requested ({signal.Signals(signum).name}); stopping the current stage cleanly")
        self.stop = True
        if self.child is not None and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                pass

    def _end_child(self) -> int:
        """Wait for the stopping child: Ctrl-C was sent; then SIGTERM, then SIGKILL."""
        for sig, wait in ((None, GRACE), (signal.SIGTERM, 60.0), (signal.SIGKILL, 30.0)):
            if sig is not None:
                try:
                    os.killpg(self.child.pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
            try:
                return self.child.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
        return -9

    def _shell(self, cmd: str, s: Stage, log: Path, st: dict | None = None) -> int:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            self.child = subprocess.Popen(
                ["bash", "-c", cmd], cwd=self.q.root, env=stage_env(s), stdout=f, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                # a queue started in the background inherits an ignored SIGINT; give the stage the
                # default back, so --stop reaches its guard as a clean Ctrl-C
                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
            if st is not None:
                st["child_pid"] = self.child.pid
            try:
                while True:
                    try:
                        rc = self.child.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        if self.stop:
                            rc = self._end_child()
                            break
            finally:
                self.child = None
        return rc

    def _wait(self, s: Stage, log: Path) -> int:
        log.parent.mkdir(parents=True, exist_ok=True)
        t0, gone, note = self.clock(), 0, -1e18
        with open(log, "a") as f:
            def w(msg):
                print(f"{now()} {msg}", file=f, flush=True)
            while not self.stop:
                pids = matching(s.wait_for, self.procs())
                if pids:
                    gone = 0
                    if self.clock() - note >= 1800:
                        w(f"still running: pid {', '.join(map(str, pids))}")
                        note = self.clock()
                else:
                    gone += 1
                    if gone >= 2:
                        w("no matching process for two checks in a row; the job has ended")
                        return 0
                if s.max_hours is not None and self.clock() - t0 > s.max_hours * 3600:
                    w(f"still running after {s.max_hours} h; giving up the wait")
                    return 1
                self.sleep(s.poll)
        return 130

    def run(self) -> int:
        q = self.q
        lock = Lock(q.dir / "queue.lock")
        if not lock.take():
            st = q.read_state()
            self.say(f"the queue is already running (pid {st['queue'].get('pid')}); "
                     "see scripts/mac_queue.sh --status")
            return 0
        old = {signal.SIGINT: signal.getsignal(signal.SIGINT), signal.SIGTERM: signal.getsignal(signal.SIGTERM)}
        for sig in old:
            signal.signal(sig, self._signal)
        try:
            return self._run()
        finally:
            for sig, h in old.items():
                signal.signal(sig, h)
            lock.drop()

    def _run(self) -> int:
        q = self.q
        st = q.read_state()
        # a stage left "running" means the queue died under it (reboot, crash, kill -9)
        for s in q.stages:
            r = st["stages"][s.name]
            if r["state"] == "running":
                if alive(r.get("child_pid")) and s.run and any(
                        pid == r["child_pid"] for pid in matching([s.run], self.procs())):
                    self.say(f"stage {s.name}'s job (pid {r['child_pid']}) is still running from an earlier "
                             "queue; stop it or let it finish, then start the queue again")
                    return 1
                r.update(state="interrupted", reason="the queue stopped during this stage (reboot or crash)")
        st["queue"] = {"pid": os.getpid(), "started": now(), "ended": None, "current": None}
        q.write_state(st)
        self.say(f"queue started: {len(q.stages)} stages, state in {q.state_path}")

        for i, s in enumerate(q.stages):
            r = st["stages"][s.name]
            if r["state"] in FINAL:
                continue
            blockers = [d for d in s.after if st["stages"][d]["state"] not in FINAL]
            if blockers:
                r.update(state="blocked", reason=f"waits for {', '.join(blockers)} to finish")
                self.say(f"stage {i + 1} {s.name}: blocked ({r['reason']})")
                q.write_state(st)
                continue
            if s.min_disk_gb and free_gb(q.root) < s.min_disk_gb:
                r.update(state="failed", reason=f"only {free_gb(q.root):.0f} GB free; needs {s.min_disk_gb:.0f} GB",
                         ended=now())
                self.say(f"stage {i + 1} {s.name}: not started, {r['reason']}")
                q.write_state(st)
                continue

            log = q.log_path(i, s)
            r.update(state="running", started=now(), ended=None, reason="", exit_code=None,
                     attempts=r.get("attempts", 0) + 1, log=shown(log, q.root))
            r.pop("child_pid", None)
            st["queue"]["current"] = s.name
            q.write_state(st)
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a") as f:
                print(f"\n===== {now()} attempt {r['attempts']}: {s.title or s.name} =====", file=f)
                print(f"$ {s.run or 'wait for: ' + ' '.join(s.wait_for)}", file=f, flush=True)
            self.say(f"stage {i + 1} {s.name}: started (log {r['log']})")
            if s.run:
                rc = self._shell(s.run, s, log, r)
                q.write_state(st)
            else:
                rc = self._wait(s, log)
            r.pop("child_pid", None)
            r.update(ended=now(), exit_code=rc)

            if self.stop:
                r.update(state="interrupted", reason="stopped by --stop or Ctrl-C")
                st["queue"].update(ended=now(), current=None)
                q.write_state(st)
                self.say(f"stage {i + 1} {s.name}: interrupted; start the queue again to resume it")
                return 130
            if rc == 0:
                ok, msgs = checks(q.root, s)
                if ok:
                    r.update(state="done", reason="")
                else:
                    r.update(state="incomplete", reason="exited 0 with work left: " + "; ".join(msgs))
            else:
                r.update(state="failed", reason=f"exited with {rc}; see its log")
            q.write_state(st)
            self.say(f"stage {i + 1} {s.name}: {r['state']}" + (f" ({r['reason']})" if r["reason"] else ""))
            if r["state"] == "failed" and s.on_fail:
                with open(log, "a") as f:
                    print(f"----- {now()} after the failure: {s.on_fail}", file=f, flush=True)
                frc = self._shell(s.on_fail, s, log)
                self.say(f"stage {i + 1} {s.name}: cleanup {'done' if frc == 0 else f'exited with {frc}'}")

        st["queue"].update(ended=now(), current=None)
        q.write_state(st)
        left = [s.name for s in q.stages if st["stages"][s.name]["state"] not in FINAL]
        self.say("queue finished: every stage done" if not left else
                 f"queue finished; not done: {', '.join(left)} (see --status, then start the queue again)")
        return 0 if not left else 1


# ---------------------------------------------------------------------- status
def status_text(q: Queue, live: bool = True, timeout: float = 60.0) -> str:
    st = q.read_state()
    qs = st["queue"]
    running = held(q.dir / "queue.lock")
    if running:
        head = f"Queue running (pid {qs.get('pid')}), started {qs.get('started')}."
    elif qs.get("started"):
        head = f"Queue not running. Last run {qs.get('started')} to {qs.get('ended') or '(cut off)'}."
    else:
        head = "Queue not started yet."
    lines = [head, ""]
    for i, s in enumerate(q.stages):
        r = st["stages"][s.name]
        state = r["state"]
        if state == "running" and not running:
            state = "interrupted"
        h = hours_between(r.get("started"), r.get("ended") if state != "running" else None)
        span = ""
        if r.get("started"):
            span = f"{r['started'][5:]} to {r['ended'][5:] if r.get('ended') and state != 'running' else 'now'}"
            span += f" ({h:.1f} h)" if h is not None else ""
        tries = f"  try {r['attempts']}" if r.get("attempts", 0) > 1 else ""
        lines.append(f"{i + 1:>2} {s.name:<16} {state:<12} {span}{tries}")
        if r.get("reason") and state not in ("done", "running"):
            lines.append(f"{'':<20}{r['reason']}")
    cur = qs.get("current") if running else None
    if cur is None:
        cur = next((s.name for s in q.stages if st["stages"][s.name]["state"] not in FINAL), None)
    if cur and live:
        s = q.stage(cur)
        i = q.stages.index(s)
        lines += ["", f"{'Now' if running and qs.get('current') == cur else 'Next'}: {s.title or s.name}"]
        if s.status:
            try:
                p = subprocess.run(["bash", "-c", s.status], cwd=q.root, env=stage_env(s), capture_output=True,
                                   text=True, timeout=timeout)
                lines += ["  " + x for x in (p.stdout + p.stderr).strip().splitlines()]
            except subprocess.TimeoutExpired:
                lines.append("  (its status command took too long)")
        log = q.log_path(i, s)
        if log.exists():
            tail = [x for x in log.read_text(errors="replace").splitlines() if x.strip()][-5:]
            lines += ["", f"  last lines of {shown(log, q.root)}:"] + ["  " + x[:160] for x in tail]
    lines += ["", f"Free disk: {free_gb(q.root):.0f} GB. Logs: {shown(q.dir / 'logs', q.root)}/"]
    return "\n".join(lines)


def set_state(q: Queue, name: str, state: str, reason: str) -> str:
    if held(q.dir / "queue.lock"):
        raise SystemExit("the queue is running; stop it first (scripts/mac_queue.sh --stop)")
    q.stage(name)
    st = q.read_state()
    st["stages"][name].update(state=state, reason=reason)
    q.write_state(st)
    return f"{name}: {state}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m markov_lean.mac_queue")
    ap.add_argument("cmd", choices=["run", "status", "retry", "skip", "list"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)
    q = Queue.load(args.config)
    if args.cmd == "run":
        return Runner(q).run()
    if args.cmd == "status":
        print(status_text(q))
        return 0
    if args.cmd == "list":
        for i, s in enumerate(q.stages):
            print(f"{i + 1:>2} {s.name:<16} {s.title}")
            print(f"{'':<20}$ {s.run or 'wait for a process matching: ' + ' + '.join(s.wait_for)}")
        return 0
    if not args.name:
        ap.error(f"{args.cmd} needs a stage name")
    if args.cmd == "retry":
        print(set_state(q, args.name, "pending", "marked to run again"))
    else:
        print(set_state(q, args.name, "skipped", "skipped by hand"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
