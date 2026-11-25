"""Search evaluation: miniF2F and PutnamBench under named prior conditions.

``markov-lean search-eval --config configs/search_eval.yaml --out results/search_eval``

Each *run* is one (benchmark, condition, seed). Its files live in
``<out>/<benchmark>/<condition>/seed<k>/``:

* ``summary.jsonl``: one line per problem (see ``docs/schema.md``, ``search_eval``);
* ``trace.jsonl``: every tactic tried, in the ``search_trace`` schema;
* ``trace.parquet`` and ``steps.parquet``: rebuilt from the JSONL at the end.

Design choices that matter for the paper:

* **Fixed budgets.** Every problem gets the same ``max_expansions`` and ``wall_time``
  (``budget`` in the config) under every condition; both are logged per problem.
* **Paired design.** Problems are visited in one fixed order (a hash of the theorem name
  and ``subset_seed``), and every condition and seed of a problem is queued before the next
  problem. Stopping early therefore leaves a random subset solved under *all* conditions,
  and ``--limit N`` takes the first ``N`` of that order, so a larger ``N`` later extends
  the same subset.
* **Common random numbers.** The run's seed goes to vLLM as the per-request sampling seed,
  so at the same goal state every condition sees the same tactic samples; conditions
  differ only in how the prior ranks them.
* **Resumable.** A problem is done once its summary line exists; rerunning the same
  command skips it. A problem whose search raised (prover down, REPL dead) is not written,
  so it is retried; a worker stops after three such errors in a row.

Conditions are prior specs (``factory`` and ``kwargs``, as in ``configs/search.yaml``).
:class:`RandomPrior` is the control: it mixes the LM distribution with *random* scores
using the same ``(1 - lam) * p_lm + lam * p_other`` rule as the area prior, so any gain of
the area prior over it is not just from perturbing the LM's ranking.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import yaml

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
AREA_FACTORY = "markov_lean.prior_loader:build_area_prior"
MAX_CONSECUTIVE_ERRORS = 3

try:  # POSIX advisory locks keep lines from different workers whole
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


# ---------------------------------------------------------------------- config


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(path: str | Path, base: Path = ROOT) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else base / p


@dataclass
class EvalConfig:
    """``configs/search_eval.yaml``, with the base search config merged in."""

    search_config: dict              # prover, lean, search sections for BestFirstSearch
    benchmarks: dict[str, dict]
    conditions: dict[str, dict]
    seeds: list[int] = field(default_factory=lambda: [0])
    msm_dir: str | None = None
    subset_seed: int = 0
    workers: int = 1
    cost: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "EvalConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "EvalConfig":
        raw = dict(raw)
        base: dict = {}
        if raw.get("base_config"):
            base = yaml.safe_load(_resolve(raw["base_config"]).read_text()) or {}
        search_config = _deep_merge(base, {k: raw[k] for k in ("prover", "lean") if k in raw})
        search_config["search"] = _deep_merge(search_config.get("search") or {}, raw.get("budget") or {})
        search_config.pop("prior", None)
        conditions = raw.get("conditions") or {"lm": {"factory": "lm"}}
        for name, spec in conditions.items():
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
                raise ValueError(f"condition name {name!r} must be letters, digits, _ . -")
            if not isinstance(spec, dict) or "factory" not in spec:
                raise ValueError(f"condition {name!r} needs a factory")
        return cls(
            search_config=search_config,
            benchmarks=raw.get("benchmarks") or {},
            conditions=conditions,
            seeds=[int(s) for s in raw.get("seeds", [0])],
            msm_dir=raw.get("msm_dir"),
            subset_seed=int(raw.get("subset_seed", 0)),
            workers=int(raw.get("workers", 1)),
            cost=raw.get("cost") or {},
        )

    @property
    def budget(self) -> dict:
        from markov_lean.search.best_first import SearchConfig

        sc = SearchConfig.from_dict(self.search_config.get("search"))
        return {"max_expansions": sc.max_expansions, "wall_time": sc.wall_time, "max_depth": sc.max_depth}

    def prior_spec(self, condition: str, seed: int) -> dict:
        """The condition's prior spec with ``msm_dir`` and the run seed filled in."""
        spec = copy.deepcopy(self.conditions[condition])
        kwargs = dict(spec.get("kwargs") or {})
        if spec["factory"] == AREA_FACTORY and "msm_dir" not in kwargs:
            if not self.msm_dir:
                raise ValueError(f"condition {condition!r} uses the area prior but no msm_dir is set")
            kwargs["msm_dir"] = self.msm_dir
        if spec["factory"] == f"{__name__}:RandomPrior":
            kwargs["seed"] = int(kwargs.get("seed", 0)) * 1_000_003 + seed
        spec["kwargs"] = kwargs
        return spec


# ---------------------------------------------------------------------- priors


def _unit_hash(*parts: Any) -> float:
    """Deterministic number in (0, 1) from ``parts``."""
    h = hashlib.blake2b("\x1f".join(map(str, parts)).encode(), digest_size=8).digest()
    return (int.from_bytes(h, "big") + 0.5) / 2.0**64


class RandomPrior:
    """Control prior: ``(1 - lam) * p_lm + lam * u`` with ``u`` random but reproducible.

    ``u`` for a candidate depends only on the seed, the goal state and the tactic, so the
    same state gets the same scores whatever order problems run in.
    """

    def __init__(self, lam: float = 0.3, seed: int = 0) -> None:
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        self.lam, self.seed = float(lam), int(seed)

    def __call__(self, state_text: str, candidates: Sequence[dict]) -> list[float]:
        from markov_lean.area_prior import logprobs_to_probs, reweight_prior

        if not candidates:
            return []
        p_lm = logprobs_to_probs([float(c["logprob"]) for c in candidates])
        u = [_unit_hash(self.seed, state_text, c["tactic"]) for c in candidates]
        return [float(x) for x in reweight_prior(p_lm, u, self.lam)]


# ---------------------------------------------------------------------- problems


_DOC_BEFORE = re.compile(r"/--(?:(?!-/).)*-/\s*$", re.S)
_SOLUTION_SORRY = re.compile(
    r"(?P<head>(?:noncomputable\s+)?(?:abbrev|def)\s+\S+[^\n]*?:=\s*)sorry[ \t]*\n(?P<ind>[ \t]*)--[ \t]*(?P<ans>[^\n]+)")


def _fill_solutions(preamble: str) -> str:
    """PutnamBench writes ``abbrev putnam_x_solution : T := sorry`` with the answer in a
    ``--`` comment on the next line. Put the answer in place of the ``sorry``."""
    return _SOLUTION_SORRY.sub(lambda m: f"{m['head']}{m['ans'].strip()}\n", preamble)


def parse_putnam_file(text: str, stem: str) -> dict | None:
    """``{"name", "statement", "preamble", "informal"}`` for the theorem in a PutnamBench
    file, or None when it has none. ``preamble`` holds everything between the imports and
    the theorem (opens, the solution abbrev with its answer filled in); ``statement`` ends
    with ``:=``. Raises ValueError when a ``sorry`` is left in the preamble."""
    from markov_lean.data import _DECL, _END

    decls = [m for m in _DECL.finditer(text) if m.group(1) == "theorem"]
    if not decls:
        return None
    m = next((d for d in decls if d.group(2) == stem), decls[0])
    end = _END.search(text, m.end())
    statement = text[m.start(): end.start() if end else len(text)].rstrip() + " :="
    before = text[: m.start()]
    informal = ""
    doc = _DOC_BEFORE.search(before)
    if doc:
        informal = doc.group(0).strip()
        before = before[: doc.start()]
    lines = [l for l in before.splitlines() if not l.strip().startswith("import ")]
    preamble = _fill_solutions("\n".join(lines) + "\n").strip()
    if re.search(r"\bsorry\b", preamble):
        raise ValueError("preamble has a sorry with no answer to fill in")
    return {"name": m.group(2), "statement": statement, "preamble": preamble, "informal": informal}


def load_putnam(root: str | Path, glob: str = "lean4/src/*.lean", source: str = "putnam") -> list:
    """PutnamBench theorems as ``Theorem`` objects; files that cannot be used are skipped
    with a warning (they are listed in the log, so the count is visible)."""
    from markov_lean.search.best_first import Theorem

    out = []
    for path in sorted(Path(root).glob(glob)):
        try:
            p = parse_putnam_file(path.read_text(errors="replace"), path.stem)
        except ValueError as e:
            log.warning("skipping %s: %s", path.name, e)
            continue
        if p is None:
            continue
        header = "import Mathlib\n\n" + (p["preamble"] + "\n\n" if p["preamble"] else "")
        out.append(Theorem(
            name=p["name"], statement=p["statement"], source=source, split="test",
            file_path=str(path.relative_to(root)),
            informal_prefix=p["informal"] + "\n" if p["informal"] else "",
            header=header, lean_preamble=p["preamble"],
        ))
    return out


def load_benchmark(name: str, spec: dict, lean_header: str | None = None) -> list:
    """All theorems of one configured benchmark, in file order."""
    from markov_lean.search.lean_env import DEFAULT_HEADER
    from markov_lean.search.runner import load_theorems

    path = _resolve(spec["path"])
    if not path.exists():
        raise FileNotFoundError(f"benchmark {name!r}: {path} not found (run `markov-lean fetch`)")
    fmt = spec.get("format", "jsonl")
    source = spec.get("source", name)
    if fmt == "jsonl":
        return list(load_theorems(path, source, splits=spec.get("splits"),
                                  env_header=lean_header or DEFAULT_HEADER))
    if fmt == "putnam":
        return load_putnam(path, spec.get("glob", "lean4/src/*.lean"), source)
    raise ValueError(f"benchmark {name!r}: unknown format {fmt!r}")


def order_problems(theorems: Iterable, subset_seed: int = 0) -> list:
    """The fixed visiting order: by a hash of the name, so any prefix is a random subset."""
    return sorted(theorems, key=lambda t: (_unit_hash("order", subset_seed, t.name), t.name))


_MINIF2F_GROUPS = (
    ("mathd_algebra", "algebra"), ("mathd_numbertheory", "number_theory"),
    ("amc12", "amc"), ("aime", "aime"), ("imo", "imo"), ("induction", "induction"),
    ("algebra", "algebra"), ("numbertheory", "number_theory"),
)


def name_area(name: str) -> str | None:
    """Problem group from a miniF2F name (``mathd_algebra_...`` -> ``algebra``); None when
    the name carries none (PutnamBench names don't)."""
    for prefix, group in _MINIF2F_GROUPS:
        if name.startswith(prefix):
            return group
    return None


# ---------------------------------------------------------------------- runs on disk


def run_dir(out: str | Path, benchmark: str, condition: str, seed: int) -> Path:
    return Path(out) / benchmark / condition / f"seed{seed}"


def read_summaries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut off by a crash
                log.warning("ignoring a broken line in %s", path)
    return rows


def done_ids(path: Path) -> set[str]:
    return {r["proof_id"] for r in read_summaries(path)}


@contextlib.contextmanager
def _locked_append(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield f
            f.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_result(directory: Path, summary: dict, trace: list[dict]) -> None:
    """Trace first, then summary: a summary line means the problem is fully written."""
    with _locked_append(directory / "trace.jsonl") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in trace))
    with _locked_append(directory / "summary.jsonl") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------- tasks


@dataclass(frozen=True)
class Task:
    benchmark: str
    condition: str
    seed: int
    index: int        # position in the visiting order


def plan_tasks(
    problems: dict[str, list],
    conditions: Sequence[str],
    seeds: Sequence[int],
    out: str | Path,
) -> tuple[list[Task], int]:
    """Tasks still to run, problem-major (all conditions and seeds of problem ``i`` before
    problem ``i + 1``, benchmarks interleaved), and how many are already done."""
    done = {(b, c, s): done_ids(run_dir(out, b, c, s) / "summary.jsonl")
            for b in problems for c in conditions for s in seeds}
    tasks, skipped = [], 0
    longest = max((len(v) for v in problems.values()), default=0)
    for i in range(longest):
        for b, thms in problems.items():
            if i >= len(thms):
                continue
            for c in conditions:
                for s in seeds:
                    if thms[i].proof_id in done[(b, c, s)]:
                        skipped += 1
                    else:
                        tasks.append(Task(b, c, s, i))
    return tasks, skipped


# ---------------------------------------------------------------------- cost estimate


DEFAULT_COST = {
    "hourly_usd": 0.8048,           # g6.xlarge on-demand, us-east-1 (check your region)
    "ebs_usd_per_hour": 0.0274,     # 200 GB gp3 at $0.08/GB-month
    "startup_hours": 0.4,           # setup check, vLLM load, Mathlib import
    "sec_per_expansion": 6.0,       # guess until a smoke run calibrates it
    "parallel_efficiency": 0.8,     # workers share one GPU and 4 vCPUs
    "expected_fraction": {"minif2f": 0.55, "putnam": 0.95},
}


@dataclass
class Estimate:
    rows: list[dict]
    workers: int
    hourly: float
    startup_hours: float

    @property
    def tasks(self) -> int:
        return sum(r["tasks"] for r in self.rows)

    def hours(self, which: str) -> float:
        return sum(r[which] for r in self.rows) / 3600.0 + self.startup_hours

    def usd(self, which: str) -> float:
        return self.hours(which) * self.hourly

    def table(self) -> str:
        lines = [f"{'benchmark':<10} {'condition':<16} {'tasks':>6} {'done':>5} "
                 f"{'expected h':>10} {'worst h':>8} {'source':<10}"]
        for r in self.rows:
            lines.append(f"{r['benchmark']:<10} {r['condition']:<16} {r['tasks']:>6} {r['done']:>5} "
                         f"{r['expected_s'] / 3600:>10.2f} {r['worst_s'] / 3600:>8.2f} {r['basis']:<10}")
        lines.append("")
        lines.append(f"{self.tasks} problem runs left on {self.workers} worker(s), "
                     f"plus {self.startup_hours:.1f} h startup, at ${self.hourly:.4f}/h (instance + disk):")
        lines.append(f"  expected  {self.hours('expected_s'):6.1f} h   ${self.usd('expected_s'):7.2f}")
        lines.append(f"  worst     {self.hours('worst_s'):6.1f} h   ${self.usd('worst_s'):7.2f}"
                     "   (every problem uses its full wall time)")
        return "\n".join(lines)


def estimate(
    counts: dict[tuple[str, str], tuple[int, int]],
    budget: dict,
    workers: int,
    cost: dict | None = None,
    observed: dict[tuple[str, str], list[float]] | None = None,
) -> Estimate:
    """Hours and dollars for the runs left.

    ``counts[(benchmark, condition)] = (tasks left, tasks done)``. Per problem run the
    worst case is the wall-time budget; the expected time is the mean elapsed of finished
    problems of that benchmark and condition when there are at least 5 (``observed``),
    else of that benchmark under any condition, else a guess:
    ``min(wall_time, max_expansions * sec_per_expansion) * expected_fraction``.
    Wall-clock time divides by ``workers * parallel_efficiency``.
    """
    c = _deep_merge(DEFAULT_COST, cost or {})
    observed = observed or {}
    speed = max(1.0, workers * float(c["parallel_efficiency"])) if workers > 1 else 1.0
    wall = float(budget["wall_time"])
    guess_full = min(wall, budget["max_expansions"] * float(c["sec_per_expansion"]))
    rows = []
    for (b, cond), (left, done) in counts.items():
        own = observed.get((b, cond), [])
        bench = [x for (bb, _), xs in observed.items() if bb == b for x in xs]
        if len(own) >= 5:
            per, basis = sum(own) / len(own), f"observed{len(own)}"
        elif len(bench) >= 5:
            per, basis = sum(bench) / len(bench), "benchmark"
        else:
            frac = float(c["expected_fraction"].get(b, 0.8))
            per, basis = guess_full * frac, "guess"
        per = min(per, wall)
        rows.append({"benchmark": b, "condition": cond, "tasks": left, "done": done,
                     "expected_s": left * per / speed, "worst_s": left * wall / speed, "basis": basis})
    return Estimate(rows, workers, float(c["hourly_usd"]) + float(c["ebs_usd_per_hour"]),
                    float(c["startup_hours"]))


def observed_elapsed(out: str | Path, benchmarks: Iterable[str], conditions: Iterable[str],
                     seeds: Iterable[int]) -> dict[tuple[str, str], list[float]]:
    obs: dict[tuple[str, str], list[float]] = {}
    for b in benchmarks:
        for c in conditions:
            for s in seeds:
                for r in read_summaries(run_dir(out, b, c, s) / "summary.jsonl"):
                    obs.setdefault((b, c), []).append(float(r.get("elapsed") or 0.0))
    return obs


# ---------------------------------------------------------------------- workers


def _import(spec: str) -> Any:
    import importlib

    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def make_env(search_config: dict):
    from markov_lean.search.lean_env import DEFAULT_HEADER, LeanEnv, default_repl_command

    lean = dict(search_config.get("lean") or {})
    cmd = lean.get("command") or default_repl_command(lean["repl_dir"])
    return LeanEnv(cmd, cwd=_resolve(lean.get("project_dir", "lean")),
                   header=lean.get("header", DEFAULT_HEADER),
                   timeout=float(lean.get("timeout", 60)),
                   startup_timeout=float(lean.get("startup_timeout", 900)))


def make_prover(search_config: dict, seed: int):
    """The proposer for ``prover.backend`` (vLLM by default, ``llama_cpp`` for a local
    GGUF) with the run seed; ``prover.factory`` (``module:callable``) swaps in another
    proposer, which tests use."""
    cfg = dict(search_config.get("prover") or {})
    factory = cfg.pop("factory", None)
    if factory:
        return _import(factory)(seed=seed, **cfg)
    from markov_lean.search.prover import make_proposer

    return make_proposer(cfg, seed=seed)


class Worker:
    """One REPL, one prover client per seed and one prior per (condition, seed), reused
    across problems."""

    def __init__(self, cfg: EvalConfig, problems: dict[str, list], out: str | Path, wid: int = 0) -> None:
        from markov_lean.search.best_first import SearchConfig

        self.cfg, self.problems, self.out, self.wid = cfg, problems, Path(out), wid
        self.search_cfg = SearchConfig.from_dict(cfg.search_config.get("search"))
        self.env = make_env(cfg.search_config)
        self._provers: dict[int, Any] = {}
        self._priors: dict[tuple[str, int], Any] = {}
        self._area: Any = None
        self._area_failed = False

    def close(self) -> None:
        for p in self._provers.values():
            close = getattr(p, "close", None)
            if callable(close):
                close()
        self.env.close()

    def prover(self, seed: int):
        if seed not in self._provers:
            self._provers[seed] = make_prover(self.cfg.search_config, seed)
        return self._provers[seed]

    def prior(self, condition: str, seed: int):
        from markov_lean.search.runner import load_prior

        key = (condition, seed)
        if key not in self._priors:
            self._priors[key] = load_prior(self.cfg.prior_spec(condition, seed))
        return self._priors[key]

    def root_area(self, state_text: str | None) -> int | None:
        """Most likely MSM area of the theorem's initial goal state, for per-area tables."""
        if not state_text or not self.cfg.msm_dir or self._area_failed:
            return None
        if self._area is None:
            try:
                from markov_lean.prior_loader import build_area_prior

                self._area = build_area_prior(self.cfg.msm_dir, lam=0.0)
            except Exception as e:  # noqa: BLE001 - area labels are optional
                log.warning("no root areas: %s", e)
                self._area_failed = True
                return None
        dist = self._area.area_of(state_text)
        return int(max(range(len(dist)), key=lambda i: dist[i]))

    def run(self, task: Task) -> dict:
        from markov_lean.search.best_first import BestFirstSearch

        thm = self.problems[task.benchmark][task.index]
        search = BestFirstSearch(self.env, self.prover(task.seed), self.prior(task.condition, task.seed),
                                 self.search_cfg)
        res = search.search(thm)
        root_state = res.trace[0]["state_before"] if res.trace else None
        summary = res.summary()
        summary.update({
            "benchmark": task.benchmark,
            "condition": task.condition,
            "seed": task.seed,
            "order": task.index,
            "max_expansions": self.search_cfg.max_expansions,
            "wall_time": self.search_cfg.wall_time,
            "root_area": self.root_area(root_state),
            "name_area": name_area(thm.name),
            "worker": self.wid,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        })
        append_result(run_dir(self.out, task.benchmark, task.condition, task.seed), summary, res.trace)
        return summary


def _work(cfg: EvalConfig, problems: dict[str, list], out: str, wid: int, tasks: Any,
          results: Any, deadline: float | None) -> None:
    """Worker loop: pull tasks until the queue's ``None``, the deadline or too many errors."""
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s w{wid} %(message)s")
    worker = Worker(cfg, problems, out, wid)
    errors = 0
    try:
        while True:
            task = tasks.get()
            if task is None:
                break
            if deadline is not None and time.time() >= deadline:
                results.put(("deadline", wid, None))
                break
            try:
                summary = worker.run(task)
                errors = 0
                results.put(("done", wid, summary))
            except Exception as e:  # noqa: BLE001 - keep going; the task is retried on resume
                errors += 1
                log.exception("task %s failed", task)
                results.put(("error", wid, {"task": task.__dict__, "error": repr(e)}))
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    results.put(("abort", wid, f"{errors} errors in a row, last: {e!r}"))
                    break
    finally:
        worker.close()
        results.put(("exit", wid, None))


class _ListQueue:
    """Same interface as a multiprocessing queue, for the in-process single-worker path."""

    def __init__(self, items: Iterable = ()) -> None:
        self.items = list(items)

    def get(self) -> Any:
        return self.items.pop(0)

    def put(self, item: Any) -> None:
        self.items.append(item)


def run_tasks(cfg: EvalConfig, problems: dict[str, list], tasks: list[Task], out: str | Path,
              workers: int = 1, deadline: float | None = None, hourly: float = 0.0,
              start_method: str = "spawn") -> dict:
    """Run ``tasks`` on ``workers`` processes (in-process when 1); returns counts."""
    out = str(out)
    stats = {"done": 0, "solved": 0, "errors": 0, "stopped": None}
    t0 = time.time()
    n = len(tasks)

    def handle(kind: str, wid: int, payload: Any) -> None:
        if kind == "done":
            stats["done"] += 1
            stats["solved"] += bool(payload["solved"])
            hrs = (time.time() - t0) / 3600
            eta = hrs / stats["done"] * (n - stats["done"]) if stats["done"] else float("nan")
            log.info("[%d/%d] %s %s seed%d %s: %s (%s, %d exp, %.0fs) | %.2f h, $%.2f so far, ~%.1f h left",
                     stats["done"], n, payload["benchmark"], payload["condition"], payload["seed"],
                     payload["theorem"], "solved" if payload["solved"] else "failed",
                     payload["stop_reason"], payload["expansions"], payload["elapsed"],
                     hrs, hrs * hourly, eta)
        elif kind == "error":
            stats["errors"] += 1
            log.warning("w%d: %s", wid, payload["error"])
        elif kind in ("abort", "deadline"):
            stats["stopped"] = payload or "time limit reached"
            log.warning("w%d stopped: %s", wid, stats["stopped"])

    if workers <= 1:
        q, results = _ListQueue([*tasks, None]), _ListQueue()
        _work(cfg, problems, out, 0, q, results, deadline)
        for item in results.items:
            handle(*item)
        return stats

    import multiprocessing as mp

    ctx = mp.get_context(start_method)
    q, results = ctx.Queue(), ctx.Queue()
    for t in tasks:
        q.put(t)
    for _ in range(workers):
        q.put(None)
    procs = [ctx.Process(target=_work, args=(cfg, problems, out, w, q, results, deadline), daemon=True)
             for w in range(workers)]
    for p in procs:
        p.start()
    alive = workers
    try:
        while alive:
            try:
                kind, wid, payload = results.get(timeout=5)
            except Exception:  # queue.Empty
                if not any(p.is_alive() for p in procs):
                    break
                continue
            if kind == "exit":
                alive -= 1
            else:
                handle(kind, wid, payload)
    except KeyboardInterrupt:
        log.warning("interrupted: stopping workers; finished problems are saved, rerun to resume")
        stats["stopped"] = "interrupted"
        for p in procs:
            p.terminate()
    for p in procs:
        p.join(timeout=30)
    return stats


# ---------------------------------------------------------------------- CLI


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _select(names: Sequence[str] | None, available: Iterable[str], what: str) -> list[str]:
    available = list(available)
    if not names:
        return available
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"unknown {what}: {unknown}; configured: {available}")
    return list(names)


def _prover_record(search_config: dict) -> dict:
    """Prover settings for runs.jsonl: no key, and the model file by name only."""
    rec = {k: v for k, v in (search_config.get("prover") or {}).items() if k != "api_key"}
    if rec.get("model_path"):
        rec["model_path"] = os.path.basename(str(rec["model_path"]))
    return rec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="markov-lean search-eval",
                                description="Proof search on miniF2F / PutnamBench under prior conditions.")
    p.add_argument("--config", default=str(ROOT / "configs" / "search_eval.yaml"))
    p.add_argument("--out", default=str(ROOT / "results" / "search_eval"))
    p.add_argument("--benchmark", action="append", help="repeatable; default all configured")
    p.add_argument("--condition", action="append", help="repeatable; default all configured")
    p.add_argument("--seed", action="append", type=int, help="repeatable; default the config's seeds")
    p.add_argument("--limit", type=int, help="first N problems per benchmark in the fixed random order")
    p.add_argument("--name", action="append", help="only these theorem names; repeatable")
    p.add_argument("--workers", type=int, help="parallel search processes (default: config)")
    p.add_argument("--endpoint", help="override prover.base_url, e.g. http://127.0.0.1:8000")
    p.add_argument("--backend", choices=["vllm", "llama_cpp"],
                   help="override prover.backend: vllm (default) or llama_cpp for a local GGUF (Mac smoke)")
    p.add_argument("--model-path", help="override prover.model_path, the GGUF file for --backend llama_cpp")
    p.add_argument("--msm-dir", help="override msm_dir (the saved Markov model for the area prior)")
    p.add_argument("--max-hours", type=float, help="stop taking new problems after this many hours")
    p.add_argument("--max-usd", type=float, help="same, as instance dollars at cost.hourly_usd")
    p.add_argument("--estimate", action="store_true", help="print the plan and cost, then exit")
    p.add_argument("--yes", action="store_true", help="run without the confirmation prompt")
    p.add_argument("--no-parquet", action="store_true", help="skip rebuilding Parquet files at the end")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = EvalConfig.load(args.config)
    if args.endpoint:
        cfg.search_config.setdefault("prover", {})["base_url"] = args.endpoint
    if args.backend:
        cfg.search_config.setdefault("prover", {})["backend"] = args.backend
    if args.model_path:
        cfg.search_config.setdefault("prover", {})["model_path"] = args.model_path
    if args.msm_dir:
        cfg.msm_dir = args.msm_dir
    benchmarks = _select(args.benchmark, cfg.benchmarks, "benchmark")
    conditions = _select(args.condition, cfg.conditions, "condition")
    seeds = args.seed or cfg.seeds
    workers = args.workers or cfg.workers
    for c in conditions:  # fail before any compute if a prior can't be specified
        cfg.prior_spec(c, seeds[0])

    header = (cfg.search_config.get("lean") or {}).get("header")
    problems: dict[str, list] = {}
    for b in benchmarks:
        thms = order_problems(load_benchmark(b, cfg.benchmarks[b], header), cfg.subset_seed)
        if args.name:
            thms = [t for t in thms if t.name in args.name]
        if args.limit is not None:
            thms = thms[: args.limit]
        problems[b] = thms

    out = Path(args.out)
    tasks, skipped = plan_tasks(problems, conditions, seeds, out)
    counts = {}
    for b in benchmarks:
        for c in conditions:
            left = sum(1 for t in tasks if t.benchmark == b and t.condition == c)
            counts[(b, c)] = (left, len(problems[b]) * len(seeds) - left)
    est = estimate(counts, cfg.budget, workers, cfg.cost,
                   observed_elapsed(out, benchmarks, conditions, seeds))
    print(f"Problems: " + ", ".join(f"{b} {len(problems[b])}" for b in benchmarks)
          + f" | seeds {list(seeds)} | budget {cfg.budget} | {skipped} problem runs already done")
    print(est.table())
    if args.estimate:
        return 0
    if not tasks:
        print("Nothing left to run.")
        return 0

    max_hours = args.max_hours
    if args.max_usd is not None:
        by_cost = args.max_usd / est.hourly
        max_hours = by_cost if max_hours is None else min(max_hours, by_cost)
    if max_hours is not None:
        print(f"Will stop taking new problems after {max_hours:.2f} h (${max_hours * est.hourly:.2f}).")
    if not args.yes:
        if not sys.stdin.isatty():
            print("Not a terminal: pass --yes to run.", file=sys.stderr)
            return 2
        if input("Run? [y/N] ").strip().lower() not in ("y", "yes"):
            return 1

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.jsonl", "a") as f:
        f.write(json.dumps({
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "commit": _git_commit(), "argv": argv if argv is not None else sys.argv[1:],
            "benchmarks": {b: len(problems[b]) for b in benchmarks}, "conditions":
            {c: cfg.prior_spec(c, 0) for c in conditions}, "seeds": list(seeds),
            "budget": cfg.budget, "workers": workers, "prover": _prover_record(cfg.search_config),
        }) + "\n")
    deadline = time.time() + max_hours * 3600 if max_hours is not None else None
    stats = run_tasks(cfg, problems, tasks, out, workers, deadline, est.hourly)
    if not args.no_parquet:
        from markov_lean.search.runner import write_parquet

        for b in benchmarks:
            for c in conditions:
                for s in seeds:
                    d = run_dir(out, b, c, s)
                    if (d / "trace.jsonl").exists():
                        write_parquet(d)
    print(f"ran {stats['done']} problem runs, solved {stats['solved']}, errors {stats['errors']}"
          + (f", stopped: {stats['stopped']}" if stats["stopped"] else ""))
    print(f"Report: markov-lean search-report --runs {out}")
    if est.hourly > 0:
        print("If you are done on the instance, stop it now (see docs/search_eval.md).")
    return 1 if stats["stopped"] and stats["stopped"] != "time limit reached" else 0


if __name__ == "__main__":
    sys.exit(main())
