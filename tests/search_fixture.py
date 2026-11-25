"""Small synthetic search-eval runs, written with the runner's own writer, for the search-report tests.

``python tests/search_fixture.py`` rewrites ``tests/fixtures/search_runs``;
``test_search_report_fixture.py`` checks the committed copy still matches.

Design (every number below is exact, so the tests can check the report by hand):

* ``minif2f``: 12 problems named like miniF2F (``mathd_algebra_*``, ...),
  ``root_area = i % 3``; ``putnam``: 6 problems, no name group, one statement
  error under ``lm``.
* Seeds 0 and 1. ``lm`` solves problems 0-3 in both seeds and problem 4 in
  seed 0 only, with ``10 + 2i`` (seed 0) or ``12 + 2i`` (seed 1) expansions
  and ``1 + i % 3`` tactics.
* ``area_l0.3`` solves problems 0-7 in both seeds, with exactly half of
  ``lm``'s expansions and one tactic more on the problems ``lm`` solves.
* ``random_l0.3`` repeats ``lm`` run for run.
* Unsolved runs stop at the 64-expansion budget, except problem 11 under
  ``lm`` seed 1, which hits the wall time after 30 expansions.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "search_runs"
SEEDS = (0, 1)
CONDITIONS = ("lm", "area_l0.3", "random_l0.3")
MINIF2F_NAMES = ("mathd_algebra", "mathd_numbertheory", "amc12a")
MAX_EXPANSIONS = 64
WALL_TIME = 300.0


def _lm_solves(i: int, seed: int) -> bool:
    return i < 4 or (i == 4 and seed == 0)


def _area_solves(i: int, seed: int) -> bool:
    return i < 8


def _lm_expansions(i: int, seed: int) -> int:
    return (10 if seed == 0 else 12) + 2 * i


def line(bench: str, cond: str, seed: int, i: int) -> dict:
    """One ``summary.jsonl`` line with every field ``search_eval.Worker.run`` writes."""
    if bench == "minif2f":
        name = f"{MINIF2F_NAMES[i % 3]}_{i}"
    else:
        name = f"putnam_2000_a{i}"
    lm_ok = _lm_solves(i, seed)
    solved = _area_solves(i, seed) if cond == "area_l0.3" else lm_ok
    stop = "solved" if solved else "budget"
    expansions = MAX_EXPANSIONS
    n_tactics = 0
    if solved:
        if cond == "area_l0.3":
            expansions = _lm_expansions(i, seed) // 2 if lm_ok else 20 + i
            n_tactics = 1 + i % 3 + (1 if lm_ok else 0)
        else:
            expansions = _lm_expansions(i, seed)
            n_tactics = 1 + i % 3
    if bench == "minif2f" and cond == "lm" and seed == 1 and i == 11:
        stop, expansions = "wall_time", 30
    if bench == "putnam" and cond == "lm" and i == 5:
        solved, stop, expansions, n_tactics = False, "statement error: unknown identifier 'Foo'", 0, 0
    from markov_lean.search_eval import name_area

    return {
        "proof_id": f"{bench}:{name}", "theorem": name, "source": bench, "split": "test",
        "solved": solved, "stop_reason": stop, "n_tactics": n_tactics, "expansions": expansions,
        "nodes": 3 * expansions + 1 if expansions else 1, "elapsed": round(0.5 * expansions + 0.25, 3),
        "proof": "\n".join(["norm_num"] * n_tactics) if solved else "",
        "benchmark": bench, "condition": cond, "seed": seed, "order": i,
        "max_expansions": MAX_EXPANSIONS, "wall_time": WALL_TIME,
        "root_area": i % 3 if bench == "minif2f" else None, "name_area": name_area(name),
        "worker": i % 3, "finished_at": f"2026-11-25T12:{i:02d}:{seed:02d}+00:00",
    }


def write(out: Path) -> Path:
    """Write the runs under ``out`` with ``search_eval.append_result`` and return ``out``."""
    from markov_lean.search_eval import append_result, run_dir

    for bench, n in (("minif2f", 12), ("putnam", 6)):
        for cond in CONDITIONS:
            for seed in SEEDS:
                for i in range(n):
                    append_result(run_dir(out, bench, cond, seed), line(bench, cond, seed, i), [])
    return out


if __name__ == "__main__":
    if FIXTURE.exists():
        shutil.rmtree(FIXTURE)
    print(write(FIXTURE), file=sys.stderr)
