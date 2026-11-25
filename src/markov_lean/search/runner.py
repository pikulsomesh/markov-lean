"""Run best-first search over a DeepSeek-Prover style benchmark file.

Reads ``{"name", "split", "formal_statement", "header", "informal_prefix"}`` rows (the
format of ``data/raw/deepseek_benchmarks/*.jsonl``) and writes, under ``out_dir``:

* ``summary.jsonl``: one line per theorem (solved, stop reason, proof text, budget used);
  appended as each theorem finishes, so a crashed run can ``--resume``;
* ``trace.jsonl``: every tactic tried (see ``best_first.TRACE_COLUMNS``), appended too;
* ``steps.parquet`` and ``trace.parquet``: written at the end; ``steps`` holds the found
  proofs in the ``steps`` schema of docs/schema.md.
"""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Iterator

from markov_lean.search.best_first import (
    STEPS_COLUMNS,
    TRACE_COLUMNS,
    BestFirstSearch,
    Prior,
    SearchConfig,
    Theorem,
    make_lm_prior,
)
from markov_lean.search.lean_env import DEFAULT_HEADER, LeanEnv, default_repl_command
from markov_lean.search.prover import ProverClient

log = logging.getLogger(__name__)


def split_header(header: str) -> tuple[list[str], str]:
    """``import`` lines, and everything else (opens, options) as a preamble."""
    imports, rest = [], []
    for line in header.splitlines():
        (imports if line.strip().startswith("import ") else rest).append(line)
    return [i.strip() for i in imports], "\n".join(rest).strip()


def load_theorems(
    path: str | Path,
    source: str,
    splits: list[str] | None = None,
    names: list[str] | None = None,
    limit: int | None = None,
    env_header: str = DEFAULT_HEADER,
) -> Iterator[Theorem]:
    env_imports = set(split_header(env_header)[0])
    n = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            split = {"valid": "val"}.get(r.get("split", "test"), r.get("split", "test"))
            if splits and split not in splits:
                continue
            if names and r["name"] not in names:
                continue
            header = r.get("header")
            imports, preamble = split_header(header or "")
            missing = set(imports) - env_imports
            if missing:
                log.warning("%s needs imports the REPL header lacks: %s; skipping", r["name"], missing)
                continue
            yield Theorem(
                name=r["name"],
                statement=r["formal_statement"],
                source=source,
                split=split,
                informal_prefix=r.get("informal_prefix") or "",
                header=header,
                lean_preamble=preamble,
            )
            n += 1
            if limit is not None and n >= limit:
                return


def load_prior(spec: dict | None) -> Prior:
    """``{"factory": "pkg.module:function", "kwargs": {...}}`` -> ``function(**kwargs)``.
    No spec (or ``factory: lm``) gives the LM-only prior."""
    spec = dict(spec or {})
    factory = spec.pop("factory", "lm")
    kwargs = spec.pop("kwargs", {}) or {}
    if factory == "lm":
        return make_lm_prior(**kwargs)
    mod, _, attr = factory.partition(":")
    if not attr:
        raise ValueError(f"prior factory must look like 'module:function', got {factory!r}")
    return getattr(importlib.import_module(mod), attr)(**kwargs)


def _done(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with open(summary_path) as f:
        return {json.loads(l)["proof_id"] for l in f if l.strip()}


def run(
    config: dict[str, Any],
    benchmark: str | Path,
    source: str,
    out_dir: str | Path,
    splits: list[str] | None = None,
    names: list[str] | None = None,
    limit: int | None = None,
    resume: bool = False,
    env: LeanEnv | None = None,
    prover: ProverClient | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lean_cfg = dict(config.get("lean") or {})
    header = lean_cfg.get("header", DEFAULT_HEADER)
    if env is None:
        cmd = lean_cfg.get("command") or default_repl_command(lean_cfg["repl_dir"])
        env = LeanEnv(
            cmd,
            cwd=lean_cfg.get("project_dir", "lean"),
            header=header,
            timeout=float(lean_cfg.get("timeout", 60)),
            startup_timeout=float(lean_cfg.get("startup_timeout", 900)),
        )
    if prover is None:
        prover = ProverClient(**(config.get("prover") or {}))
    search = BestFirstSearch(
        env, prover, load_prior(config.get("prior")), SearchConfig.from_dict(config.get("search"))
    )
    summary_path, trace_path = out / "summary.jsonl", out / "trace.jsonl"
    if not resume:
        for p in (summary_path, trace_path):
            p.unlink(missing_ok=True)
    skip = _done(summary_path) if resume else set()
    solved = total = 0
    with env:
        for thm in load_theorems(benchmark, source, splits, names, limit, header):
            if thm.proof_id in skip:
                continue
            res = search.search(thm)
            total += 1
            solved += res.solved
            with open(summary_path, "a") as f:
                f.write(json.dumps(res.summary(), ensure_ascii=False) + "\n")
            with open(trace_path, "a") as f:
                for row in res.trace:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.info("%s: %s (%s, %d expansions, %.0fs) [%d/%d solved]", thm.proof_id,
                     "solved" if res.solved else "failed", res.stop_reason, res.expansions,
                     res.elapsed, solved, total)
    write_parquet(out)
    return {"theorems": total, "solved": solved}


def write_parquet(out_dir: str | Path) -> None:
    """Rebuild ``trace.parquet`` and ``steps.parquet`` from ``trace.jsonl``."""
    import pandas as pd

    from markov_lean.search.best_first import SearchResult, proof_steps

    out = Path(out_dir)
    trace_path = out / "trace.jsonl"
    rows = []
    if trace_path.exists():
        with open(trace_path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    trace = pd.DataFrame(rows, columns=TRACE_COLUMNS)
    trace.to_parquet(out / "trace.parquet", index=False)
    steps: list[dict] = []
    for _, g in trace.groupby("proof_id", sort=False):
        recs = g.to_dict("records")
        for r in recs:
            if r["child_id"] is not None and r["child_id"] == r["child_id"]:  # not NaN
                r["child_id"] = int(r["child_id"])
            else:
                r["child_id"] = None
        fake = SearchResult(theorem=None, solved=True, stop_reason="", trace=recs)  # type: ignore[arg-type]
        steps.extend(proof_steps(fake))
    pd.DataFrame(steps, columns=STEPS_COLUMNS).to_parquet(out / "steps.parquet", index=False)
