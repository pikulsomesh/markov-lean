"""Tactic-mode wrapper around leanprover-community/repl.

The REPL reads one JSON command per blank-line-terminated block on stdin and writes one
JSON response per blank-line-terminated block on stdout. We use three commands:

* ``{"cmd": header}`` once, to import Mathlib and get a base ``env``;
* ``{"cmd": "<statement> := by\\n  sorry", "env": base}`` per theorem, which returns the
  initial ``proofState`` and goal through ``sorries``;
* ``{"tactic": t, "proofState": n}`` per step, which returns the new ``proofState`` and
  its ``goals``, or ``{"message": "Lean error: ..."}``.

Every request has a wall-clock timeout. On a timeout the REPL process is killed and a
fresh one is started, which invalidates every ``proofState`` id handed out before. The
``generation`` counter goes up on each restart so callers (the search) know to replay.

Run the REPL from inside the Lean project so imports resolve, i.e.
``lake env <repl checkout>/.lake/build/bin/repl`` with ``cwd=lean/``.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Same imports and opens DeepSeek-Prover-V1.5 puts in its prompts (prover/utils.py), so the
# code the model sees and the code Lean checks agree. DeepSeek sets maxHeartbeats 0; we keep
# a cap so a runaway tactic fails inside Lean before the wall-clock timeout kills the REPL.
DEFAULT_HEADER = (
    "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\n"
    "open BigOperators Real Nat Topology Rat\n"
)

NO_GOALS = "no goals"
_FORBIDDEN = re.compile(r"\b(sorry|admit)\b|\bnative_decide\b")


class LeanError(RuntimeError):
    """The REPL could not be started or returned something we cannot use."""


class LeanTimeout(LeanError):
    """A request ran past its timeout; the REPL has been restarted."""


@dataclass
class TacticResult:
    """Outcome of one tactic on one proof state."""

    ok: bool
    proof_state: int | None = None
    goals: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed: float = 0.0
    generation: int = 0

    @property
    def solved(self) -> bool:
        return self.ok and not self.goals

    @property
    def state_text(self) -> str:
        return goals_to_text(self.goals)


@dataclass
class TheoremState:
    """Root of a proof attempt: the initial proof state of ``statement``."""

    statement: str
    proof_state: int
    goals: list[str]
    generation: int

    @property
    def state_text(self) -> str:
        return goals_to_text(self.goals)


def goals_to_text(goals: list[str]) -> str:
    """Pretty-printed state in the same shape as LeanDojo's ``state_before``: goals joined
    by a blank line, ``no goals`` when the proof is closed."""
    return "\n\n".join(goals) if goals else NO_GOALS


def as_sorry_command(statement: str) -> str:
    """Turn ``theorem foo ... :=`` or ``... := by`` (with or without a trailing newline,
    DeepSeek's benchmark files have both) into ``... := by\\n  sorry``."""
    s = statement.rstrip()
    if s.endswith("sorry"):
        return s
    if s.endswith(":= by"):
        return s + "\n  sorry"
    if s.endswith(":="):
        return s + " by\n  sorry"
    raise ValueError(f"statement must end with ':=' or ':= by': {statement[-60:]!r}")


def default_repl_command(repl_dir: str | os.PathLike) -> list[str]:
    return ["lake", "env", str(Path(repl_dir).expanduser() / ".lake" / "build" / "bin" / "repl")]


class LeanEnv:
    """One REPL process plus the base environment holding ``header``.

    ``command`` is the argv that starts the REPL (see :func:`default_repl_command`);
    ``cwd`` is the Lean project directory. Use as a context manager or call ``close``.
    """

    def __init__(
        self,
        command: list[str] | str,
        cwd: str | os.PathLike | None = None,
        header: str = DEFAULT_HEADER,
        timeout: float = 60.0,
        startup_timeout: float = 900.0,
        forbid: re.Pattern[str] | None = _FORBIDDEN,
    ) -> None:
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        self.cwd = str(cwd) if cwd is not None else None
        self.header = header
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.forbid = forbid
        self.generation = 0
        self.base_env: int | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._stderr: Any = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()

    # -- process management ------------------------------------------------------------

    def start(self) -> None:
        if self._proc is not None:
            return
        # stderr goes to a file (a pipe nobody reads could fill and block the REPL); its
        # tail goes into the error when the REPL won't start or dies, e.g. "unknown
        # package 'Mathlib'" when lake runs outside the project or Mathlib isn't built.
        self._close_stderr()
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        try:
            self._proc = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as e:
            raise LeanError(f"could not start the REPL with {self.command} in {self.cwd}: {e}") from e
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self._proc, self._lines), daemon=True).start()
        try:
            resp = self._send({"cmd": self.header}, self.startup_timeout)
        except LeanError as e:
            tail = self.last_stderr
            raise type(e)(f"REPL did not start ({e}); command {self.command} in {self.cwd}"
                          + (f"; stderr: {tail}" if tail else "")) from e
        errors = _errors(resp)
        if "env" not in resp or errors:
            self.close()
            raise LeanError(f"header failed to elaborate: {errors or resp}")
        self.base_env = int(resp["env"])

    @staticmethod
    def _pump(proc: subprocess.Popen[str], out: queue.Queue[str | None]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            out.put(line)
        out.put(None)  # EOF

    def close(self) -> None:
        proc, self._proc = self._proc, None
        self.base_env = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    @property
    def last_stderr(self) -> str:
        """The last 2000 characters the REPL wrote to stderr (kept after it exits)."""
        f = self._stderr
        if f is None:
            return ""
        try:
            f.flush()
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - 2000))
            return f.read().decode("utf-8", errors="replace").strip()
        except (OSError, ValueError):
            return ""

    def _close_stderr(self) -> None:
        if self._stderr is not None:
            try:
                self._stderr.close()
            except OSError:
                pass
            self._stderr = None

    def restart(self) -> None:
        self.close()
        self.generation += 1
        self.start()

    def __enter__(self) -> "LeanEnv":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
        self._close_stderr()

    # -- protocol ----------------------------------------------------------------------

    def _send(self, payload: dict, timeout: float) -> dict:
        if self._proc is None or self._proc.stdin is None:
            raise LeanError("REPL is not running")
        with self._lock:
            try:
                self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n\n")
                self._proc.stdin.flush()
            except OSError as e:
                self._crashed()
                raise LeanError(f"REPL stdin closed: {e}") from e
            deadline = time.monotonic() + timeout
            buf: list[str] = []
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._timed_out()
                    raise LeanTimeout(f"no response in {timeout:.0f}s")
                try:
                    line = self._lines.get(timeout=remaining)
                except queue.Empty:
                    self._timed_out()
                    raise LeanTimeout(f"no response in {timeout:.0f}s") from None
                if line is None:
                    self._crashed()
                    raise LeanError("REPL exited")
                if line.strip():
                    buf.append(line)
                elif buf:
                    break
        try:
            return json.loads("".join(buf))
        except json.JSONDecodeError as e:
            raise LeanError(f"unparseable REPL output: {''.join(buf)[:500]!r}") from e

    def _timed_out(self) -> None:
        # The REPL may still be grinding on the request; its proof states are gone with it.
        self.close()
        self.generation += 1

    def _crashed(self) -> None:
        self.close()
        self.generation += 1

    def _ensure(self) -> None:
        if self._proc is None:
            self.start()

    # -- public API --------------------------------------------------------------------

    def init_theorem(self, statement: str) -> TheoremState:
        """Elaborate ``statement`` with a ``sorry`` proof and return its initial state."""
        self._ensure()
        try:
            resp = self._send(
                {"cmd": as_sorry_command(statement), "env": self.base_env}, self.timeout
            )
        except LeanTimeout:
            self.start()
            raise
        if "message" in resp and "sorries" not in resp:
            raise LeanError(resp["message"])
        errors = _errors(resp)
        if errors:
            raise LeanError("statement does not elaborate: " + " | ".join(errors))
        sorries = resp.get("sorries") or []
        if len(sorries) != 1 or sorries[0].get("proofState") is None:
            raise LeanError(f"expected one sorry with a proofState, got {sorries}")
        s = sorries[0]
        return TheoremState(
            statement=statement,
            proof_state=int(s["proofState"]),
            goals=[s["goal"]] if s.get("goal") else [],
            generation=self.generation,
        )

    def run_tactic(self, proof_state: int, tactic: str) -> TacticResult:
        """Run ``tactic`` on ``proof_state``. Never raises for Lean-side failures: those
        come back as ``ok=False`` with ``error`` set (``timeout`` after a restart)."""
        if self.forbid is not None and self.forbid.search(tactic):
            return TacticResult(ok=False, error="forbidden tactic", generation=self.generation)
        self._ensure()
        gen = self.generation
        t0 = time.monotonic()
        try:
            resp = self._send({"tactic": tactic, "proofState": proof_state}, self.timeout)
        except LeanTimeout:
            self.start()
            return TacticResult(
                ok=False, error="timeout", elapsed=time.monotonic() - t0, generation=gen
            )
        except LeanError as e:
            self.start()
            return TacticResult(ok=False, error=str(e), elapsed=time.monotonic() - t0, generation=gen)
        elapsed = time.monotonic() - t0
        if "message" in resp and "proofState" not in resp:
            return TacticResult(ok=False, error=resp["message"], elapsed=elapsed, generation=gen)
        errors = _errors(resp)
        if errors:
            return TacticResult(ok=False, error=" | ".join(errors), elapsed=elapsed, generation=gen)
        goals = list(resp.get("goals") or [])
        status = resp.get("proofStatus")
        # Newer REPLs check the whole proof term when goals run out; older ones send no
        # proofStatus, and then empty goals are all we can go on.
        if not goals and status is not None and status != "Completed":
            return TacticResult(ok=False, error=f"proof status: {status}", elapsed=elapsed, generation=gen)
        return TacticResult(
            ok=True,
            proof_state=int(resp["proofState"]),
            goals=goals,
            elapsed=elapsed,
            generation=gen,
        )

    def replay(self, statement: str, tactics: list[str]) -> TacticResult:
        """Re-create a state after a restart by replaying ``tactics`` from the root."""
        root = self.init_theorem(statement)
        res = TacticResult(ok=True, proof_state=root.proof_state, goals=root.goals, generation=self.generation)
        for t in tactics:
            assert res.proof_state is not None
            res = self.run_tactic(res.proof_state, t)
            if not res.ok:
                return res
        return res


def _errors(resp: dict) -> list[str]:
    return [m.get("data", "") for m in resp.get("messages") or [] if m.get("severity") == "error"]
