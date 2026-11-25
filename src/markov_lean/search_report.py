"""Pass@k tables, paired tests and figures for best-first search runs.

Runs live under one root as ``<root>/<benchmark>/<condition>/seed<k>/summary.jsonl``
(one line per theorem, written by ``markov_lean.search.runner``). Conditions are
arbitrary strings (``lm`` is the no-prior baseline by convention).

* ``load_summaries``: every summary line as one row; a theorem repeated by a
  resumed run keeps its last line.
* ``pass_at_k`` / ``pass_table``: the unbiased pass@k estimator (Chen et al.
  2021) per problem, averaged over problems, with a percentile bootstrap CI
  over problems. A statement error counts as an unsolved problem.
* ``paired_compare``: every condition against the baseline on the problems
  both ran: paired bootstrap on the per-problem solve rate, and an exact
  McNemar test on solved-in-any-seed, Holm-adjusted within each benchmark.
* ``area_table`` / ``area_gain_table``: pass@1 by mathematical area, and each
  condition's paired gain over the baseline within each area.
* ``budget_curve`` / ``budget_gain_curve``: the share of runs solved within
  each expansion budget, and the paired gain over the baseline at each budget,
  both with bootstrap CIs over problems.
* ``search_stats`` / ``effort_compare``: proof length, expansions, nodes and
  stop reasons per condition, and expansions and proof length against the
  baseline on the runs both solved.
* ``write_report``: all of the above as CSV, Markdown, booktabs LaTeX tables
  and paper PDFs. ``main`` is the command-line entry point; ``--runs`` takes
  one or more paths, so ``--runs results/search_eval/*`` works.

matplotlib and seaborn are only needed for the figures and are imported lazily.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

KEY = ["benchmark", "condition", "seed", "proof_id"]
COLUMNS = [
    "proof_id", "theorem", "source", "split", "solved", "stop_reason", "n_tactics",
    "expansions", "nodes", "elapsed", "proof", "benchmark", "condition", "seed",
    "max_expansions", "wall_time", "root_area", "name_area", "backend", "model",
]
DEFAULT_KS = (1, 2, 4, 8, 16, 32)

__all__ = [
    "load_summaries",
    "pass_at_k",
    "pass_table",
    "holm",
    "paired_compare",
    "area_table",
    "budget_curve",
    "fig_pass_at_1",
    "fig_by_area",
    "fig_budget",
    "area_gain_table",
    "search_stats",
    "effort_compare",
    "budget_gain_curve",
    "fig_budget_gain",
    "fig_area_gain",
    "fig_effort",
    "mcnemar_power",
    "mcnemar_mde",
    "detectable_effects",
    "local_model_note",
    "write_report",
    "main",
]


# ---------------------------------------------------------------- loading
_PATTERNS = ("*/*/seed*/summary.jsonl", "*/seed*/summary.jsonl", "seed*/summary.jsonl", "summary.jsonl")


def _summary_paths(path: Path) -> list[Path]:
    """The ``summary.jsonl`` files at or under ``path``, at the run layout's depths.

    ``path`` can be a runs root, one benchmark's folder, one condition's
    folder, one seed's folder or a ``summary.jsonl`` itself, so a shell glob
    such as ``results/search_eval/*`` works. Deeper folders are not searched,
    so a smoke run nested inside another root is not picked up by accident.
    """
    if path.is_file():
        return [path] if path.name == "summary.jsonl" else []
    found: list[Path] = []
    for pat in _PATTERNS:
        found += [p for p in sorted(path.glob(pat)) if p.parent.name.startswith("seed")]
    return found


def load_summaries(roots: str | Path | Sequence[str | Path]) -> pd.DataFrame:
    """Every summary line under ``roots`` as one row, deduplicated on (benchmark, condition, seed, proof_id).

    ``roots`` is one path or several (see :func:`_summary_paths` for what each
    may point at). ``benchmark``, ``condition`` and ``seed`` come from the
    line, or from the path when the line lacks them. When the same key appears
    more than once (a resumed run, or two roots holding the same run) the last
    line wins, in the order the roots are given. Blank or truncated lines (a
    crash mid-write) are skipped. Missing optional columns are added as nulls.
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]
    paths: list[Path] = []
    for r in roots:
        for path in _summary_paths(Path(r)):
            path = path.resolve()
            if path not in paths:
                paths.append(path)
    rows: list[dict] = []
    for path in paths:
        if len(path.parts) < 4:
            continue
        bench, cond, seed_dir = path.parts[-4], path.parts[-3], path.parts[-2]
        try:
            seed = int(seed_dir[len("seed"):])
        except ValueError:
            seed = seed_dir[len("seed"):]
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                rec.setdefault("benchmark", bench)
                rec.setdefault("condition", cond)
                rec.setdefault("seed", seed)
                rows.append(rec)
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in COLUMNS})
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df.drop_duplicates(KEY, keep="last").reset_index(drop=True)
    df["solved"] = df["solved"].fillna(False).astype(bool)
    df["stop_reason"] = df["stop_reason"].fillna("").astype(str)
    df["proof_id"] = df["proof_id"].astype(str)
    for c in ("expansions", "nodes", "n_tactics", "elapsed", "max_expansions", "wall_time"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _statement_error(df: pd.DataFrame) -> pd.Series:
    return df["stop_reason"].astype(str).str.startswith("statement error")


# ---------------------------------------------------------------- estimators
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from ``n`` samples with ``c`` correct: ``1 - C(n-c, k) / C(n, k)``.

    Computed as a running product (Chen et al. 2021), so large ``n`` does not
    overflow. Needs ``0 <= c <= n`` and ``1 <= k <= n``.
    """
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _boot_means(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return values[idx].mean(axis=1)


def _boot_ci(values: Sequence[float], n_boot: int, ci: float, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI of the mean over the items of ``values``."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if n_boot <= 0:
        return float(v.mean()), float("nan"), float("nan")
    means = _boot_means(v, n_boot, rng)
    a = (1 - ci) / 2
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def _per_problem(sub: pd.DataFrame) -> pd.DataFrame:
    """Seeds run and seeds solved per problem."""
    return sub.groupby("proof_id", sort=False).agg(n=("seed", "nunique"), c=("solved", "sum"))


def _conditions(df: pd.DataFrame) -> list[tuple[str, str]]:
    return list(dict.fromkeys(zip(df["benchmark"], df["condition"])))


def pass_table(df: pd.DataFrame, ks: Iterable[int] = (1,), n_boot: int = 2000, ci: float = 0.95,
               seed: int = 0) -> pd.DataFrame:
    """One row per (benchmark, condition): pass@k with bootstrap CIs, as fractions.

    Only problems run under every seed of the condition count; a problem whose
    statement failed to elaborate counts as unsolved. ``n_seeds`` is the
    minimum number of seeds per problem, and ``pass@k`` columns are filled only
    for ``k <= n_seeds``. ``n_statement_errors`` counts problems with a
    statement error in any seed; ``mean_expansions_solved`` and
    ``median_elapsed`` are over runs.
    """
    ks = sorted(set(int(k) for k in ks))
    rng = np.random.default_rng(seed)
    out = []
    for bench, cond in _conditions(df):
        sub = df[(df["benchmark"] == bench) & (df["condition"] == cond)]
        n_all = sub["seed"].nunique()
        per = _per_problem(sub)
        per = per[per["n"] == n_all]
        sub = sub[sub["proof_id"].isin(per.index)]
        row: dict = {"benchmark": bench, "condition": cond, "n_problems": len(per),
                     "n_seeds": int(per["n"].min()) if len(per) else 0}
        for k in ks:
            if k > row["n_seeds"]:
                continue
            vals = [pass_at_k(int(n), int(c), k) for n, c in zip(per["n"], per["c"])]
            p, lo, hi = _boot_ci(vals, n_boot, ci, rng)
            row.update({f"pass@{k}": p, f"pass@{k}_lo": lo, f"pass@{k}_hi": hi})
        row["n_statement_errors"] = int(sub.loc[_statement_error(sub), "proof_id"].nunique())
        solved = sub.loc[sub["solved"], "expansions"]
        row["mean_expansions_solved"] = float(solved.mean()) if len(solved) else float("nan")
        row["median_elapsed"] = float(sub["elapsed"].median()) if len(sub) else float("nan")
        out.append(row)
    cols = ["benchmark", "condition", "n_problems", "n_seeds"]
    cols += [f"pass@{k}{s}" for k in ks for s in ("", "_lo", "_hi")]
    cols += ["n_statement_errors", "mean_expansions_solved", "median_elapsed"]
    res = pd.DataFrame(out, columns=cols)
    empty_k = [c for c in cols if c.startswith("pass@") and res[c].isna().all()]
    return res.drop(columns=empty_k) if len(res) else res


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adj = np.maximum.accumulate(adj)
    out = np.empty(m)
    out[order] = adj
    return out


def _mcnemar(only_a: int, only_b: int) -> float:
    from scipy.stats import binomtest

    n = only_a + only_b
    return 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)


def mcnemar_power(n: int, discordance: float, gain: float, alpha: float = 0.05) -> float:
    """Power of the exact two-sided McNemar test on ``n`` paired problems.

    ``discordance`` is the share of problems the two conditions disagree on and ``gain``
    the true difference in solve rates (``P(only cond) - P(only base)``). The number of
    discordant problems is Binomial(n, discordance); given it, the test is exact
    binomial at 1/2, and the count won by the condition is Binomial(D, (d + gain) / 2d).
    """
    from scipy.stats import binom

    d, g = float(discordance), abs(float(gain))
    if n <= 0 or d <= 0:
        return 0.0
    g = min(g, d)
    q = (d + g) / (2 * d)
    total = 0.0
    for D in range(1, n + 1):
        pd_ = binom.pmf(D, n, d)
        if pd_ < 1e-12:
            continue
        k = np.arange(D + 1)
        pv = np.minimum(1.0, 2 * np.minimum(binom.cdf(k, D, 0.5), binom.sf(k - 1, D, 0.5)))
        total += pd_ * float(binom.pmf(k, D, q)[pv <= alpha].sum())
    return total


def mcnemar_mde(n: int, discordance: float, alpha: float = 0.05, power: float = 0.8,
                step: float = 0.0025) -> float:
    """Smallest gain in solve rate (a share, not points) that the exact McNemar test finds
    with probability ``power`` on ``n`` problems; NaN when no gain up to ``discordance``
    reaches it (too few problems disagree)."""
    d = float(discordance)
    for g in np.arange(step, d + step / 2, step):
        if mcnemar_power(n, d, min(g, d), alpha) >= power:
            return float(min(g, d))
    return float("nan")


def detectable_effects(paired: pd.DataFrame, alpha: float = 0.05, power: float = 0.8) -> pd.DataFrame:
    """For each paired comparison, the gain it can detect at its size and observed discordance."""
    rows = []
    for r in paired.to_dict("records"):
        n = int(r["n_common"])
        disc = (r["only_cond"] + r["only_base"]) / n if n else float("nan")
        mde = mcnemar_mde(n, disc, alpha, power) if n and disc > 0 else float("nan")
        rows.append({"benchmark": r["benchmark"], "condition": r["condition"], "n_common": n,
                     "discordance": disc, "mde": mde})
    return pd.DataFrame(rows, columns=["benchmark", "condition", "n_common", "discordance", "mde"])


def _runs_records(summary_paths: Iterable[Path]) -> list[dict]:
    """``runs.jsonl`` lines from the folders above each run (``<out>/runs.jsonl``)."""
    seen: set[Path] = set()
    recs: list[dict] = []
    for sp in summary_paths:
        for d in list(Path(sp).resolve().parents)[1:4]:
            f = d / "runs.jsonl"
            if f in seen or not f.is_file():
                continue
            seen.add(f)
            for line in f.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    recs.append(rec)
    return recs


def local_model_note(df: pd.DataFrame, roots: str | Path | Sequence[str | Path] = ()) -> str | None:
    """A warning when any run used the local llama.cpp prover (the Mac), else None.

    Read from each summary line's ``backend`` and, for older lines, from ``runs.jsonl``.
    """
    local = "backend" in df.columns and (df["backend"].astype(str) == "llama_cpp").any()
    models = sorted({str(m) for m in df.get("model", pd.Series(dtype=object)).dropna()})
    if isinstance(roots, (str, Path)):
        roots = [roots]
    paths = [p for r in roots for p in _summary_paths(Path(r))]
    for rec in _runs_records(paths):
        prover = rec.get("prover") or {}
        if prover.get("backend") == "llama_cpp":
            local = True
            if prover.get("model_path"):
                models = sorted(set(models) | {os.path.basename(str(prover["model_path"]))})
    if not local:
        return None
    exp = pd.to_numeric(df["max_expansions"], errors="coerce").dropna()
    wall = pd.to_numeric(df["wall_time"], errors="coerce").dropna()
    budget = ""
    if len(exp) and len(wall):
        budget = f" with a small budget ({int(exp.max())} expansions, {wall.max():.0f} s per problem)"
    model = f" ({', '.join(models)})" if models else ""
    return (f"Local runs: a quantized prover{model} run by llama.cpp on one machine{budget}. "
            "Pass rates are not comparable with the vLLM runs on EC2 or with published numbers; "
            "only the paired comparisons between conditions within these runs are meaningful.")


def paired_compare(df: pd.DataFrame, baseline: str = "lm", n_boot: int = 2000, seed: int = 0,
                   ci: float = 0.95) -> pd.DataFrame:
    """Each condition against ``baseline``, per benchmark, on the problems both ran.

    ``rate_*`` is the mean over problems of the per-problem pass@1 (the share
    of that problem's seeds solved). ``diff_lo``/``diff_hi`` are a percentile
    paired bootstrap over problems. ``p_boot`` is two-sided and uses the
    centered bootstrap: the share of resampled differences whose distance from
    the observed difference is at least the observed difference's distance
    from 0, i.e. ``(1 + #{|d* - d| >= |d|}) / (1 + n_boot)``. The McNemar test
    is exact (binomial on the discordant pairs) on solved-in-any-seed, and
    ``p_holm`` adjusts it across the comparisons within each benchmark.
    Benchmarks without the baseline are skipped.
    """
    rng = np.random.default_rng(seed)
    cols = ["benchmark", "condition", "baseline", "n_common", "rate_cond", "rate_base", "diff",
            "diff_lo", "diff_hi", "p_boot", "only_cond", "only_base", "both", "neither",
            "p_mcnemar", "p_holm"]
    out = []
    a = (1 - ci) / 2
    for bench in dict.fromkeys(df["benchmark"]):
        sub = df[df["benchmark"] == bench]
        if baseline not in set(sub["condition"]):
            continue
        per = sub.groupby(["condition", "proof_id"], sort=False)["solved"].agg(["mean", "any"])
        base = per.loc[baseline]
        rows = []
        for cond in dict.fromkeys(sub["condition"]):
            if cond == baseline:
                continue
            other = per.loc[cond]
            common = other.index.intersection(base.index)
            rc = other.loc[common, "mean"].to_numpy(float)
            rb = base.loc[common, "mean"].to_numpy(float)
            ac = other.loc[common, "any"].to_numpy(bool)
            ab = base.loc[common, "any"].to_numpy(bool)
            d = rc - rb
            row = {"benchmark": bench, "condition": cond, "baseline": baseline, "n_common": len(common)}
            if len(common):
                diff = float(d.mean())
                if n_boot > 0:
                    boot = _boot_means(d, n_boot, rng)
                    lo, hi = float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))
                    p_boot = float((1 + np.sum(np.abs(boot - diff) >= abs(diff) - 1e-12)) / (1 + n_boot))
                else:
                    lo = hi = p_boot = float("nan")
                row.update(rate_cond=float(rc.mean()), rate_base=float(rb.mean()), diff=diff,
                           diff_lo=lo, diff_hi=hi, p_boot=min(1.0, p_boot))
            else:
                row.update(rate_cond=float("nan"), rate_base=float("nan"), diff=float("nan"),
                           diff_lo=float("nan"), diff_hi=float("nan"), p_boot=float("nan"))
            only_c, only_b = int((ac & ~ab).sum()), int((~ac & ab).sum())
            row.update(only_cond=only_c, only_base=only_b, both=int((ac & ab).sum()),
                       neither=int((~ac & ~ab).sum()), p_mcnemar=_mcnemar(only_c, only_b))
            rows.append(row)
        if rows:
            adj = holm([r["p_mcnemar"] for r in rows])
            for r, p in zip(rows, adj):
                r["p_holm"] = float(p)
        out.extend(rows)
    return pd.DataFrame(out, columns=cols)


def _area_labels(s: pd.Series) -> pd.Series:
    """Area values as strings, with nulls as ``"unknown"`` (integral floats lose their ``.0``)."""
    def fmt(v) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
            return "unknown"
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return str(int(v))
        return str(v)

    return s.map(fmt)


def _area_key(s: pd.Series) -> pd.Series:
    """Sort key for area labels: numeric areas in numeric order, then names, ``"unknown"`` last."""
    num = pd.to_numeric(s, errors="coerce")
    return pd.Series([(2, 0.0, a) if a == "unknown" else (0, n, "") if np.isfinite(n) else (1, 0.0, a)
                      for a, n in zip(s, num)], index=s.index)


def _problem_areas(df: pd.DataFrame, area_col: str) -> pd.Series:
    """One area label per (benchmark, proof_id): the last known one over every condition and seed.

    The first goal state is the same under every condition, so its area should
    be too; taking one label per problem keeps a problem from being split
    across areas when one run failed to label it.
    """
    d = df.assign(area=_area_labels(df[area_col]))
    out = d.groupby(["benchmark", "proof_id"], sort=False)["area"].last()
    known = d[d["area"] != "unknown"].groupby(["benchmark", "proof_id"], sort=False)["area"].last()
    out.loc[known.index] = known
    return out


def area_table(df: pd.DataFrame, area_col: str, n_boot: int = 1000, seed: int = 0,
               ci: float = 0.95) -> pd.DataFrame:
    """Pass@1 per (benchmark, condition, area) with a bootstrap CI over problems and ``n`` problems.

    Empty if ``area_col`` is not a column. Null areas are grouped as ``"unknown"``.
    """
    cols = ["benchmark", "condition", "area", "n", "pass@1", "pass@1_lo", "pass@1_hi"]
    if area_col not in df.columns:
        return pd.DataFrame(columns=cols)
    rng = np.random.default_rng(seed)
    d = df.assign(area=_area_labels(df[area_col]))
    # one area per problem: the last one seen, so a problem is not split across areas
    per = d.groupby(["benchmark", "condition", "proof_id"], sort=False).agg(
        rate=("solved", "mean"), area=("area", "last")).reset_index()
    out = []
    for (bench, cond, area), g in per.groupby(["benchmark", "condition", "area"], sort=False):
        p, lo, hi = _boot_ci(g["rate"].to_numpy(), n_boot, ci, rng)
        out.append({"benchmark": bench, "condition": cond, "area": area, "n": len(g),
                    "pass@1": p, "pass@1_lo": lo, "pass@1_hi": hi})
    res = pd.DataFrame(out, columns=cols)
    res = res.sort_values("area", key=_area_key, kind="stable")
    return res.sort_values("benchmark", kind="stable").reset_index(drop=True)


def area_gain_table(df: pd.DataFrame, area_col: str, baseline: str = "lm", n_boot: int = 1000,
                    seed: int = 0, ci: float = 0.95) -> pd.DataFrame:
    """Per-area gain of each condition over ``baseline``, paired on the problems both ran.

    One row per (benchmark, area, condition). ``diff`` is the mean over the
    area's problems of the per-problem pass@1 difference (condition minus
    baseline), with a percentile paired bootstrap CI over those problems.
    ``only_cond``/``only_base`` count problems solved in some seed by one side
    only. Each problem has one area (:func:`_problem_areas`). Empty if
    ``area_col`` is missing; benchmarks without the baseline are skipped.
    """
    cols = ["benchmark", "area", "condition", "baseline", "n", "rate_cond", "rate_base", "diff",
            "diff_lo", "diff_hi", "only_cond", "only_base"]
    if area_col not in df.columns or df.empty:
        return pd.DataFrame(columns=cols)
    rng = np.random.default_rng(seed)
    a = (1 - ci) / 2
    areas = _problem_areas(df, area_col)
    conds = list(dict.fromkeys(df["condition"]))
    out = []
    for bench in dict.fromkeys(df["benchmark"]):
        sub = df[df["benchmark"] == bench]
        if baseline not in set(sub["condition"]):
            continue
        per = sub.groupby(["condition", "proof_id"], sort=False)["solved"].agg(["mean", "any"])
        base = per.loc[baseline]
        bench_areas = areas.loc[bench]
        for cond in dict.fromkeys(sub["condition"]):
            if cond == baseline:
                continue
            other = per.loc[cond]
            common = other.index.intersection(base.index)
            labels = bench_areas.reindex(common).fillna("unknown").to_numpy()
            for area in dict.fromkeys(labels):
                ids = common[labels == area]
                d = other.loc[ids, "mean"].to_numpy(float) - base.loc[ids, "mean"].to_numpy(float)
                diff = float(d.mean())
                if n_boot > 0:
                    boot = _boot_means(d, n_boot, rng)
                    lo, hi = float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))
                else:
                    lo = hi = float("nan")
                ac, ab = other.loc[ids, "any"].to_numpy(bool), base.loc[ids, "any"].to_numpy(bool)
                out.append({"benchmark": bench, "area": area, "condition": cond, "baseline": baseline,
                            "n": len(ids), "rate_cond": float(other.loc[ids, "mean"].mean()),
                            "rate_base": float(base.loc[ids, "mean"].mean()), "diff": diff,
                            "diff_lo": lo, "diff_hi": hi, "only_cond": int((ac & ~ab).sum()),
                            "only_base": int((~ac & ab).sum())})
    res = pd.DataFrame(out, columns=cols)
    if res.empty:
        return res
    res = res.sort_values("condition", key=lambda c: c.map(conds.index), kind="stable")
    res = res.sort_values("area", key=_area_key, kind="stable")
    benches = list(dict.fromkeys(df["benchmark"]))
    return res.sort_values("benchmark", key=lambda c: c.map(benches.index), kind="stable").reset_index(drop=True)


STOP_KINDS = ("budget", "wall_time", "exhausted", "statement error")


def _stop_kind(s: pd.Series) -> pd.Series:
    """``stop_reason`` without detail: ``statement error: unknown id`` becomes ``statement error``."""
    return s.astype(str).str.split(":", n=1).str[0].str.strip()


def search_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Proof-length and search-effort statistics per (benchmark, condition), over runs.

    ``len_*`` is the length in tactics of the proofs found (solved runs only);
    ``exp_solved_*`` the expansions a solved run used; ``nodes_mean`` the mean
    search nodes created per run and ``nodes_per_expansion`` total nodes over
    total expansions (all runs); ``stop_*`` the share of runs that stopped for
    each reason (``stop_other`` for anything else that is not ``solved``).
    """
    cols = ["benchmark", "condition", "n_runs", "n_solved", "solve_rate", "len_mean", "len_median",
            "len_q25", "len_q75", "len_max", "exp_solved_mean", "exp_solved_median", "nodes_mean",
            "nodes_solved_median", "nodes_per_expansion", "elapsed_median", "elapsed_solved_median"]
    cols += [f"stop_{k.replace(' ', '_')}" for k in STOP_KINDS] + ["stop_other"]
    out = []

    def q(x: pd.Series, f: float) -> float:
        x = x.dropna()
        return float(x.quantile(f)) if len(x) else float("nan")

    def mean(x: pd.Series) -> float:
        x = x.dropna()
        return float(x.mean()) if len(x) else float("nan")

    for bench, cond in _conditions(df):
        sub = df[(df["benchmark"] == bench) & (df["condition"] == cond)]
        solved = sub[sub["solved"]]
        length = pd.to_numeric(solved["n_tactics"], errors="coerce")
        nodes = pd.to_numeric(sub["nodes"], errors="coerce")
        exps = pd.to_numeric(sub["expansions"], errors="coerce")
        both = nodes.notna() & exps.notna()
        stop = _stop_kind(sub["stop_reason"])
        row = {"benchmark": bench, "condition": cond, "n_runs": len(sub), "n_solved": len(solved),
               "solve_rate": float(sub["solved"].mean()) if len(sub) else float("nan"),
               "len_mean": mean(length), "len_median": q(length, 0.5), "len_q25": q(length, 0.25),
               "len_q75": q(length, 0.75), "len_max": float(length.max()) if length.notna().any() else float("nan"),
               "exp_solved_mean": mean(solved["expansions"]), "exp_solved_median": q(solved["expansions"], 0.5),
               "nodes_mean": mean(nodes),
               "nodes_solved_median": q(pd.to_numeric(solved["nodes"], errors="coerce"), 0.5),
               "nodes_per_expansion": float(nodes[both].sum() / exps[both].sum()) if exps[both].sum() > 0
               else float("nan"),
               "elapsed_median": q(sub["elapsed"], 0.5), "elapsed_solved_median": q(solved["elapsed"], 0.5)}
        for k in STOP_KINDS:
            row[f"stop_{k.replace(' ', '_')}"] = float((stop == k).mean()) if len(sub) else float("nan")
        other = ~stop.isin(STOP_KINDS) & ~sub["solved"].to_numpy()
        row["stop_other"] = float(other.mean()) if len(sub) else float("nan")
        out.append(row)
    return pd.DataFrame(out, columns=cols)


def effort_compare(df: pd.DataFrame, baseline: str = "lm", n_boot: int = 2000, seed: int = 0,
                   ci: float = 0.95) -> pd.DataFrame:
    """Search effort of each condition against ``baseline`` on the runs both solved.

    Runs are paired on (problem, seed): the seed fixes the LM's samples, so a
    pair differs only in the prior's ranking. ``exp_ratio`` is the geometric
    mean over problems of expansions(condition) / expansions(baseline)
    (per-problem mean of log2 ratios over its seeds, then averaged), below 1
    when the condition needs fewer prover calls; ``len_diff`` is the mean
    difference in proof length (tactics). CIs are percentile bootstraps over
    problems; ``p_wilcoxon`` is a two-sided signed-rank test on the per-problem
    log ratios. Conditioning on runs both solved selects easier problems, so
    read this next to the pass@k gain, not instead of it.
    """
    cols = ["benchmark", "condition", "baseline", "n_pairs", "n_problems", "exp_cond_median",
            "exp_base_median", "exp_ratio", "exp_ratio_lo", "exp_ratio_hi", "len_cond_mean",
            "len_base_mean", "len_diff", "len_diff_lo", "len_diff_hi", "p_wilcoxon"]
    rng = np.random.default_rng(seed)
    out = []
    for bench in dict.fromkeys(df["benchmark"]):
        sub = df[df["benchmark"] == bench]
        if baseline not in set(sub["condition"]):
            continue
        keep = ["proof_id", "seed", "expansions", "n_tactics"]
        base = sub[(sub["condition"] == baseline) & sub["solved"]][keep]
        for cond in dict.fromkeys(sub["condition"]):
            if cond == baseline:
                continue
            other = sub[(sub["condition"] == cond) & sub["solved"]][keep]
            j = other.merge(base, on=["proof_id", "seed"], suffixes=("_c", "_b"))
            j = j.dropna(subset=["expansions_c", "expansions_b"])
            row = {"benchmark": bench, "condition": cond, "baseline": baseline, "n_pairs": len(j)}
            j = j.assign(lr=np.log2(np.maximum(j["expansions_c"], 1)) - np.log2(np.maximum(j["expansions_b"], 1)),
                         ld=j["n_tactics_c"] - j["n_tactics_b"])
            per = j.groupby("proof_id", sort=False)[["lr", "ld"]].mean()
            row["n_problems"] = len(per)
            row["exp_cond_median"] = float(j["expansions_c"].median()) if len(j) else float("nan")
            row["exp_base_median"] = float(j["expansions_b"].median()) if len(j) else float("nan")
            m, lo, hi = _boot_ci(per["lr"].to_numpy(float), n_boot, ci, rng)
            row.update(exp_ratio=2.0 ** m, exp_ratio_lo=2.0 ** lo, exp_ratio_hi=2.0 ** hi)
            row["len_cond_mean"] = float(j["n_tactics_c"].mean()) if len(j) else float("nan")
            row["len_base_mean"] = float(j["n_tactics_b"].mean()) if len(j) else float("nan")
            m, lo, hi = _boot_ci(per["ld"].dropna().to_numpy(float), n_boot, ci, rng)
            row.update(len_diff=m, len_diff_lo=lo, len_diff_hi=hi)
            row["p_wilcoxon"] = _wilcoxon(per["lr"].to_numpy(float))
            out.append(row)
    return pd.DataFrame(out, columns=cols)


def _wilcoxon(x: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p on ``x``; 1.0 when every value is 0, NaN when empty."""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    if not np.any(x != 0):
        return 1.0
    from scipy.stats import wilcoxon

    return float(wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)


def _default_grid(max_e: int, n: int = 24) -> np.ndarray:
    max_e = max(int(max_e), 1)
    grid = np.unique(np.round(np.logspace(0, np.log10(max_e), n)).astype(int))
    return np.unique(np.append(grid, max_e))


def _grid_for(df: pd.DataFrame, grid: Sequence[int] | None) -> np.ndarray:
    if grid is None:
        cap = pd.to_numeric(df["max_expansions"], errors="coerce").max() if "max_expansions" in df else np.nan
        if not np.isfinite(cap):
            cap = pd.to_numeric(df["expansions"], errors="coerce").max()
        grid = _default_grid(int(cap) if np.isfinite(cap) else 1)
    return np.asarray(sorted(set(int(g) for g in grid)))


def _within_budget(sub: pd.DataFrame, grid: np.ndarray) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """Problems, solved-within-budget counts (problems x budgets) and runs per problem."""
    codes, ids = pd.factorize(sub["proof_id"])
    e = pd.to_numeric(sub["expansions"], errors="coerce").to_numpy(float)
    s = sub["solved"].to_numpy(bool)
    with np.errstate(invalid="ignore"):
        hit = s[:, None] & (e[:, None] <= grid[None, :])
    counts = np.zeros((len(ids), len(grid)))
    np.add.at(counts, codes, hit)
    return pd.Index(ids), counts, np.bincount(codes, minlength=len(ids)).astype(float)


def _weights(n_items: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap resamples as multinomial counts per item (``n_boot`` x ``n_items``)."""
    return rng.multinomial(n_items, np.full(n_items, 1.0 / n_items), size=n_boot).astype(float)


def budget_curve(df: pd.DataFrame, grid: Sequence[int] | None = None, n_boot: int = 1000, seed: int = 0,
                 ci: float = 0.95) -> pd.DataFrame:
    """Share of problem-seed runs solved within ``expansions <= e``, per (benchmark, condition, e).

    The default grid is about 24 log-spaced integers from 1 up to the largest
    ``max_expansions`` (or the largest ``expansions`` when that is missing),
    always including the maximum. ``solved_lo``/``solved_hi`` are a
    percentile bootstrap over problems (all of a problem's seeds resampled
    together); NaN when ``n_boot`` is 0.
    """
    cols = ["benchmark", "condition", "budget", "solved_frac", "solved_lo", "solved_hi", "n_runs"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    grid = _grid_for(df, grid)
    rng = np.random.default_rng(seed)
    a = (1 - ci) / 2
    out = []
    for bench, cond in _conditions(df):
        sub = df[(df["benchmark"] == bench) & (df["condition"] == cond)]
        _, counts, runs = _within_budget(sub, grid)
        frac = counts.sum(axis=0) / runs.sum() if runs.sum() else np.full(len(grid), np.nan)
        lo = hi = np.full(len(grid), np.nan)
        if n_boot > 0 and len(runs):
            w = _weights(len(runs), n_boot, rng)
            boot = (w @ counts) / (w @ runs)[:, None]
            lo, hi = np.quantile(boot, a, axis=0), np.quantile(boot, 1 - a, axis=0)
        for i, g in enumerate(grid):
            out.append({"benchmark": bench, "condition": cond, "budget": int(g), "solved_frac": float(frac[i]),
                        "solved_lo": float(lo[i]), "solved_hi": float(hi[i]), "n_runs": int(runs.sum())})
    return pd.DataFrame(out, columns=cols)


def budget_gain_curve(df: pd.DataFrame, baseline: str = "lm", grid: Sequence[int] | None = None,
                      n_boot: int = 1000, seed: int = 0, ci: float = 0.95) -> pd.DataFrame:
    """Gain over ``baseline`` in the share solved within each expansion budget, paired on problems.

    For each problem both ran, the share of its seeds solved within ``e``
    expansions under the condition minus under the baseline, averaged over
    problems (``diff``, a fraction), with a percentile paired bootstrap over
    problems. Same default grid as :func:`budget_curve`.
    """
    cols = ["benchmark", "condition", "baseline", "budget", "diff", "diff_lo", "diff_hi", "n_common"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    grid = _grid_for(df, grid)
    rng = np.random.default_rng(seed)
    a = (1 - ci) / 2
    out = []
    for bench in dict.fromkeys(df["benchmark"]):
        sub = df[df["benchmark"] == bench]
        if baseline not in set(sub["condition"]):
            continue
        ids_b, cnt_b, runs_b = _within_budget(sub[sub["condition"] == baseline], grid)
        frac_b = pd.DataFrame(cnt_b / runs_b[:, None], index=ids_b)
        for cond in dict.fromkeys(sub["condition"]):
            if cond == baseline:
                continue
            ids_c, cnt_c, runs_c = _within_budget(sub[sub["condition"] == cond], grid)
            frac_c = pd.DataFrame(cnt_c / runs_c[:, None], index=ids_c)
            common = ids_c.intersection(ids_b)
            d = (frac_c.loc[common] - frac_b.loc[common]).to_numpy(float)
            diff = d.mean(axis=0) if len(common) else np.full(len(grid), np.nan)
            lo = hi = np.full(len(grid), np.nan)
            if n_boot > 0 and len(common):
                w = _weights(len(common), n_boot, rng)
                boot = (w @ d) / len(common)
                lo, hi = np.quantile(boot, a, axis=0), np.quantile(boot, 1 - a, axis=0)
            for i, g in enumerate(grid):
                out.append({"benchmark": bench, "condition": cond, "baseline": baseline, "budget": int(g),
                            "diff": float(diff[i]), "diff_lo": float(lo[i]), "diff_hi": float(hi[i]),
                            "n_common": len(common)})
    return pd.DataFrame(out, columns=cols)


# ---------------------------------------------------------------- figures
def _half_width(t: pd.DataFrame, col: str) -> pd.Series:
    """Symmetric error for a bar: the larger side of the asymmetric CI, in percent."""
    err = np.maximum(t[f"{col}_hi"] - t[col], t[col] - t[f"{col}_lo"]) * 100
    return err.fillna(0.0)


def fig_pass_at_1(table: pd.DataFrame, figsize=None):
    """Pass@1 (%) by condition, grouped by benchmark, from :func:`pass_table`.

    ``figures.search_ablation`` draws one symmetric error per bar, so the bar's
    error is the larger half of the bootstrap CI, ``max(hi - p, p - lo)``: the
    drawn interval covers the CI and may overstate its short side.
    """
    from markov_lean import figures

    t = table.copy()
    t["pass@1 (%)"] = t["pass@1"] * 100
    t["err"] = _half_width(table, "pass@1")
    if figsize is None:
        figsize = (figures.COLUMN_WIDTH if t["condition"].nunique() <= 4 else figures.TEXT_WIDTH, 2.4)
    fig = figures.search_ablation(t, method_col="condition", metric_col="pass@1 (%)",
                                  group_col="benchmark", err_col="err",
                                  metric_label="pass@1 (%)", figsize=figsize)
    fig.axes[0].set_ylim(bottom=0)  # the symmetric error can reach below 0
    return fig


def fig_by_area(areas: pd.DataFrame, benchmark: str, figsize=None):
    """Pass@1 (%) by condition, grouped by area, for one benchmark (errors as in :func:`fig_pass_at_1`)."""
    from markov_lean import figures

    t = areas[areas["benchmark"] == benchmark].copy()
    if t.empty:
        raise ValueError(f"no area rows for benchmark {benchmark!r}")
    t["pass@1 (%)"] = t["pass@1"] * 100
    t["err"] = _half_width(t, "pass@1")
    t["area"] = t["area"].astype(str)
    fig = figures.search_ablation(t, method_col="condition", metric_col="pass@1 (%)",
                                  group_col="area", err_col="err", metric_label="pass@1 (%)",
                                  figsize=figsize or (figures.TEXT_WIDTH, 2.4))
    ax = fig.axes[0]
    ax.set_ylim(bottom=0)
    ax.set_xlabel("area")
    return fig


def fig_budget(curve: pd.DataFrame, conditions: Sequence[str] | None = None, figsize=None):
    """Solved (%) against the expansion budget (log x), one panel per benchmark, one line per condition.

    Condition colors follow ``conditions`` (default: order of first appearance
    in ``curve``). The bootstrap CI is shaded when ``curve`` has it. The
    benchmark names the y axis of each panel; no title.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    from markov_lean import figures

    conds = list(conditions) if conditions is not None else list(dict.fromkeys(curve["condition"]))
    benches = list(dict.fromkeys(curve["benchmark"]))
    width = figures.COLUMN_WIDTH if len(benches) <= 1 else figures.TEXT_WIDTH
    with matplotlib.rc_context(figures.paper_rc()):
        fig, axes = plt.subplots(1, max(len(benches), 1), figsize=figsize or (width, 2.2),
                                 layout="constrained", squeeze=False)
        for ax, bench in zip(axes[0], benches):
            sub = curve[curve["benchmark"] == bench]
            for i, cond in enumerate(conds):
                c = sub[sub["condition"] == cond].sort_values("budget")
                if c.empty:
                    continue
                ax.plot(c["budget"], c["solved_frac"] * 100, color=figures.color_for(i),
                        linewidth=1.3, label=cond)
                if {"solved_lo", "solved_hi"} <= set(c.columns) and c["solved_lo"].notna().any():
                    ax.fill_between(c["budget"], c["solved_lo"] * 100, c["solved_hi"] * 100,
                                    color=figures.color_for(i), alpha=0.15, linewidth=0)
            ax.set_xscale("log")
            ax.set_xlabel("expansion budget")
            ax.set_ylabel(f"{bench}: solved (%)")
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        if len(conds) > 1:
            handles, labels = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside upper center", ncol=min(len(conds), 4),
                       handlelength=1.2, columnspacing=1.0)
    return fig


def _colors(conditions: Sequence[str]) -> dict[str, str]:
    from markov_lean import figures

    return {c: figures.color_for(i) for i, c in enumerate(conditions)}


def fig_budget_gain(gain: pd.DataFrame, conditions: Sequence[str] | None = None, figsize=None):
    """Gain over the baseline (points) against the expansion budget, from :func:`budget_gain_curve`.

    One panel per benchmark, one line per condition with its paired bootstrap
    CI shaded, and a line at 0. Colors follow ``conditions``, which should
    start with the baseline so each condition keeps the color it has in the
    other figures.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    from markov_lean import figures

    conds = list(conditions) if conditions is not None else list(dict.fromkeys(gain["condition"]))
    colors = _colors(conds)
    benches = list(dict.fromkeys(gain["benchmark"]))
    width = figures.COLUMN_WIDTH if len(benches) <= 1 else figures.TEXT_WIDTH
    with matplotlib.rc_context(figures.paper_rc()):
        fig, axes = plt.subplots(1, max(len(benches), 1), figsize=figsize or (width, 2.2),
                                 layout="constrained", squeeze=False)
        for ax, bench in zip(axes[0], benches):
            sub = gain[gain["benchmark"] == bench]
            ax.axhline(0, color=figures.TEXT_SECONDARY, linewidth=0.8)
            for cond in conds:
                c = sub[sub["condition"] == cond].sort_values("budget")
                if c.empty:
                    continue
                ax.plot(c["budget"], c["diff"] * 100, color=colors[cond], linewidth=1.3, label=cond)
                if c["diff_lo"].notna().any():
                    ax.fill_between(c["budget"], c["diff_lo"] * 100, c["diff_hi"] * 100,
                                    color=colors[cond], alpha=0.15, linewidth=0)
            base = sub["baseline"].iloc[0] if len(sub) else ""
            ax.set_xscale("log")
            ax.set_xlabel("expansion budget")
            ax.set_ylabel(f"{bench}: gain over {base} (pts)")
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if len(labels) > 1:
            fig.legend(handles, labels, loc="outside upper center", ncol=min(len(labels), 4),
                       handlelength=1.2, columnspacing=1.0)
    return fig


def fig_area_gain(gains: pd.DataFrame, benchmark: str, conditions: Sequence[str] | None = None, figsize=None):
    """Per-area gain over the baseline (points) with paired bootstrap CIs, for one benchmark.

    One row per area (top to bottom in :func:`area_gain_table` order), one
    point and asymmetric CI whisker per condition, and a line at 0. Colors
    follow ``conditions`` as in :func:`fig_budget_gain`.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    from markov_lean import figures

    t = gains[gains["benchmark"] == benchmark]
    if t.empty:
        raise ValueError(f"no area-gain rows for benchmark {benchmark!r}")
    conds = [c for c in (conditions if conditions is not None else dict.fromkeys(t["condition"]))
             if c in set(t["condition"])]
    colors = _colors(list(conditions) if conditions is not None else conds)
    areas = list(dict.fromkeys(t["area"].astype(str)))
    n_areas = len(areas)
    height = 0.9 + 0.22 * n_areas * max(1, len(conds) / 2)
    step = 0.7 / max(len(conds), 1)
    with matplotlib.rc_context(figures.paper_rc()):
        fig, ax = plt.subplots(figsize=figsize or (figures.COLUMN_WIDTH, min(height, 8.0)), layout="constrained")
        ax.axvline(0, color=figures.TEXT_SECONDARY, linewidth=0.8)
        for j, cond in enumerate(conds):
            c = t[t["condition"] == cond].assign(area=lambda x: x["area"].astype(str)).set_index("area")
            c = c.reindex(areas)
            y = np.arange(n_areas) + (j - (len(conds) - 1) / 2) * step
            x = c["diff"].to_numpy(float) * 100
            err = np.vstack([x - c["diff_lo"].to_numpy(float) * 100, c["diff_hi"].to_numpy(float) * 100 - x])
            err = np.where(np.isfinite(err), err, 0.0)
            ax.errorbar(x, y, xerr=err, fmt="o", markersize=3, color=colors[cond], ecolor=colors[cond],
                        elinewidth=1, capsize=2, label=cond)
        ax.set_yticks(np.arange(n_areas))
        ax.set_yticklabels(areas)
        ax.invert_yaxis()
        ax.set_ylabel("area")
        base = t["baseline"].iloc[0]
        ax.set_xlabel(f"{benchmark}: gain over {base} in pass@1 (pts)")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        if len(conds) > 1:
            fig.legend(loc="outside upper center", ncol=min(len(conds), 3), handlelength=1.0,
                       columnspacing=1.0)
    return fig


def fig_effort(df: pd.DataFrame, conditions: Sequence[str] | None = None, figsize=None):
    """Expansions used by solved runs (log y), one box per condition, one panel per benchmark."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    from markov_lean import figures

    sns = figures._sns()
    solved = df[df["solved"] & df["expansions"].notna()]
    conds = list(conditions) if conditions is not None else list(dict.fromkeys(df["condition"]))
    colors = _colors(conds)
    benches = list(dict.fromkeys(df["benchmark"]))
    width = figures.COLUMN_WIDTH if len(benches) <= 1 else figures.TEXT_WIDTH
    with matplotlib.rc_context(figures.paper_rc()):
        fig, axes = plt.subplots(1, max(len(benches), 1), figsize=figsize or (width, 2.4),
                                 layout="constrained", squeeze=False)
        for ax, bench in zip(axes[0], benches):
            sub = solved[solved["benchmark"] == bench]
            order = [c for c in conds if c in set(sub["condition"])]
            if order:
                sns.boxplot(data=sub, x="condition", y="expansions", hue="condition", order=order,
                            hue_order=order, palette={c: colors[c] for c in order}, legend=False,
                            width=0.6, fliersize=1.5, linewidth=0.8, saturation=1, ax=ax)
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.set_xlabel("")
            ax.set_ylabel(f"{bench}: expansions to solve")
            ax.tick_params(axis="x", labelrotation=30)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
    return fig


# ---------------------------------------------------------------- writers
BENCH_LABELS = {"minif2f": "miniF2F", "putnam": "PutnamBench", "proofnet": "ProofNet"}


def _pct(p: float, lo: float | None = None, hi: float | None = None) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    s = f"{100 * p:.1f}"
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{100 * lo:.1f}, {100 * hi:.1f}]"
    return s


def _pts(d: float, lo: float | None = None, hi: float | None = None) -> str:
    """A difference of fractions in signed percentage points, with an optional CI."""
    if d is None or not np.isfinite(d):
        return "--"
    s = f"{100 * d:+.1f}"
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    return s


def _num(x: float, fmt: str = "{:.1f}") -> str:
    return fmt.format(x) if x is not None and np.isfinite(x) else "--"


def _pval(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _md_table(headers: Sequence[str], rows: Iterable[Sequence]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def _tex_pval(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


class _Labels:
    """LaTeX names for conditions and benchmarks: a given label is raw LaTeX, anything else is escaped."""

    def __init__(self, conditions: dict[str, str] | None = None, benchmarks: dict[str, str] | None = None):
        self.conditions = dict(conditions or {})
        self.benchmarks = {**BENCH_LABELS, **(benchmarks or {})}

    def cond(self, c: str) -> str:
        return self.conditions.get(c, _tex_escape(str(c)))

    def bench(self, b: str) -> str:
        return self.benchmarks.get(b, _tex_escape(str(b)))


def _latex_table(table: pd.DataFrame, conditions: Sequence[str], k: int = 1, labels: _Labels | None = None,
                 bold_best: bool = True) -> str:
    """booktabs table: one row per condition, one column per benchmark, ``xx.x [lo, hi]`` percent.

    With ``bold_best`` the highest point estimate in each column is bold.
    """
    labels = labels or _Labels()
    benches = list(dict.fromkeys(table["benchmark"]))
    col = f"pass@{k}"
    lines = [
        r"\begin{tabular}{l" + "c" * len(benches) + "}",
        r"\toprule",
        "Condition & " + " & ".join(labels.bench(b) for b in benches) + r" \\",
        r"\midrule",
    ]
    idx = table.set_index(["condition", "benchmark"])
    best = table.groupby("benchmark")[col].max() if col in table.columns else pd.Series(dtype=float)
    for cond in conditions:
        cells = []
        for b in benches:
            if (cond, b) in idx.index and col in idx.columns:
                r = idx.loc[(cond, b)]
                cell = _pct(r[col], r[f"{col}_lo"], r[f"{col}_hi"])
                if bold_best and np.isfinite(r[col]) and r[col] == best.get(b) and " [" in cell:
                    head, tail = cell.split(" [", 1)
                    cell = rf"\textbf{{{head}}} [{tail}"
                cells.append(cell)
            else:
                cells.append("--")
        lines.append(labels.cond(cond) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def _latex_pass_multi(table: pd.DataFrame, conditions: Sequence[str], ks: Sequence[int],
                      labels: _Labels | None = None) -> str:
    """booktabs table of pass@k point estimates (percent): benchmarks across, one column per k under each."""
    labels = labels or _Labels()
    benches = list(dict.fromkeys(table["benchmark"]))
    ks = [k for k in ks if f"pass@{k}" in table.columns]
    idx = table.set_index(["condition", "benchmark"])
    lines = [r"\begin{tabular}{l" + ("c" * len(ks)) * len(benches) + "}", r"\toprule"]
    head = ["Condition"] + [rf"\multicolumn{{{len(ks)}}}{{c}}{{{labels.bench(b)}}}" for b in benches]
    lines.append(" & ".join(head) + r" \\")
    rules = [rf"\cmidrule(lr){{{2 + i * len(ks)}-{1 + (i + 1) * len(ks)}}}" for i in range(len(benches))]
    lines.append(" ".join(rules))
    lines.append(" & " + " & ".join(f"$k={k}$" for _ in benches for k in ks) + r" \\")
    lines.append(r"\midrule")
    best = {(b, k): table.loc[table["benchmark"] == b, f"pass@{k}"].max() for b in benches for k in ks}
    for cond in conditions:
        cells = []
        for b in benches:
            for k in ks:
                v = idx.loc[(cond, b), f"pass@{k}"] if (cond, b) in idx.index else np.nan
                cell = _num(100 * v if np.isfinite(v) else np.nan)
                if np.isfinite(v) and v == best[(b, k)]:
                    cell = rf"\textbf{{{cell}}}"
                cells.append(cell)
        lines.append(labels.cond(cond) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def _latex_paired(paired: pd.DataFrame, labels: _Labels | None = None) -> str:
    """booktabs table of each condition against the baseline: gain in points [CI], discordant counts, p values."""
    labels = labels or _Labels()
    lines = [r"\begin{tabular}{llrcccc}", r"\toprule",
             r"Benchmark & Condition & $n$ & $\Delta$ pass@1 (pts) & Only cond./base & $p_\mathrm{boot}$ & "
             r"$p_\mathrm{Holm}$ \\", r"\midrule"]
    first = True
    for bench, g in paired.groupby("benchmark", sort=False):
        if not first:
            lines.append(r"\midrule")
        first = False
        for i, r in enumerate(g.to_dict("records")):
            name = labels.bench(bench) if i == 0 else ""
            lines.append(f"{name} & {labels.cond(r['condition'])} & {r['n_common']} & "
                         f"{_pts(r['diff'], r['diff_lo'], r['diff_hi'])} & {r['only_cond']}/{r['only_base']} & "
                         f"{_tex_pval(r['p_boot'])} & {_tex_pval(r['p_holm'])}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def _latex_area_gain(gains: pd.DataFrame, conditions: Sequence[str], labels: _Labels | None = None) -> str:
    """booktabs table of per-area gains: a block per benchmark, one row per area, one column per condition."""
    labels = labels or _Labels()
    conds = [c for c in conditions if c in set(gains["condition"])]
    lines = [r"\begin{tabular}{lr" + "c" * len(conds) + "}", r"\toprule",
             "Area & $n$ & " + " & ".join(labels.cond(c) for c in conds) + r" \\", r"\midrule"]
    first = True
    for bench, g in gains.groupby("benchmark", sort=False):
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.append(rf"\multicolumn{{{2 + len(conds)}}}{{l}}{{\emph{{{labels.bench(bench)}}}}} \\")
        idx = g.set_index(["area", "condition"])
        for area in dict.fromkeys(g["area"]):
            n = int(g.loc[g["area"] == area, "n"].max())
            cells = []
            for c in conds:
                if (area, c) in idx.index:
                    r = idx.loc[(area, c)]
                    cells.append(_pts(r["diff"], r["diff_lo"], r["diff_hi"]))
                else:
                    cells.append("--")
            lines.append(f"{_tex_escape(str(area))} & {n} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def _latex_stats(stats: pd.DataFrame, effort: pd.DataFrame, labels: _Labels | None = None) -> str:
    """booktabs table of search effort per condition: solve rate, proof length, expansions, nodes, stops, ratio."""
    labels = labels or _Labels()
    eff = effort.set_index(["benchmark", "condition"]) if len(effort) else None
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Benchmark & Condition & Solved (\%) & Length & Exp.\ to solve & Nodes/run & "
             r"Budget stops (\%) & Exp.\ ratio \\", r"\midrule"]
    first = True
    for bench, g in stats.groupby("benchmark", sort=False):
        if not first:
            lines.append(r"\midrule")
        first = False
        for i, r in enumerate(g.to_dict("records")):
            name = labels.bench(bench) if i == 0 else ""
            length = (f"{_num(r['len_median'], '{:.0f}')} [{_num(r['len_q25'], '{:.0f}')}, "
                      f"{_num(r['len_q75'], '{:.0f}')}]" if np.isfinite(r["len_median"]) else "--")
            ratio = "--"
            if eff is not None and (bench, r["condition"]) in eff.index:
                e = eff.loc[(bench, r["condition"])]
                if np.isfinite(e["exp_ratio"]):
                    ratio = f"{e['exp_ratio']:.2f}"
                    if np.isfinite(e["exp_ratio_lo"]):
                        ratio += f" [{e['exp_ratio_lo']:.2f}, {e['exp_ratio_hi']:.2f}]"
            stops = r["stop_budget"] + r["stop_wall_time"]
            lines.append(f"{name} & {labels.cond(r['condition'])} & {_num(100 * r['solve_rate'])} & {length} & "
                         f"{_num(r['exp_solved_median'], '{:.0f}')} & {_num(r['nodes_mean'])} & "
                         f"{_num(100 * stops)} & {ratio}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def _pick_area_col(df: pd.DataFrame, area_col: str | None) -> str | None:
    if area_col:
        return area_col if area_col in df.columns else None
    for c in ("root_area", "name_area"):
        if c in df.columns and df[c].notna().any():
            return c
    return None


def _report_md(table: pd.DataFrame, paired: pd.DataFrame, areas: pd.DataFrame | None,
               ks: Sequence[int], baseline: str, area_col: str | None, n_boot: int,
               gains: pd.DataFrame | None = None, stats: pd.DataFrame | None = None,
               effort: pd.DataFrame | None = None, note: str | None = None,
               detect: pd.DataFrame | None = None) -> str:
    parts = ["# Search report", ""]
    if note:
        parts += [f"> **{note}**", ""]
    headers = ["benchmark", "condition", "problems", "seeds"] + [f"pass@{k} (%)" for k in ks]
    headers += ["stmt errors", "mean exp. (solved)", "median time (s)"]
    rows = []
    for r in table.to_dict("records"):
        cells = [r["benchmark"], r["condition"], r["n_problems"], r["n_seeds"]]
        cells += [_pct(r.get(f"pass@{k}", np.nan), r.get(f"pass@{k}_lo"), r.get(f"pass@{k}_hi")) for k in ks]
        mexp, med = r["mean_expansions_solved"], r["median_elapsed"]
        cells += [r["n_statement_errors"], f"{mexp:.1f}" if np.isfinite(mexp) else "--",
                  f"{med:.1f}" if np.isfinite(med) else "--"]
        rows.append(cells)
    parts += ["## pass@k", "",
              f"Mean over problems of the unbiased pass@k; brackets are 95% percentile bootstrap CIs "
              f"over problems ({n_boot} resamples). Statement errors count as unsolved.", "",
              _md_table(headers, rows), ""]
    parts += ["## Paired comparisons", ""]
    if paired.empty:
        parts += [f"No benchmark has the baseline condition `{baseline}`.", ""]
    else:
        parts += [f"Each condition against `{baseline}` on the problems both ran. Diff is the mean per-problem "
                  "pass@1 difference in points with a paired bootstrap CI; McNemar is exact on "
                  "solved-in-any-seed, Holm-adjusted within each benchmark.", ""]
        rows = [[r["benchmark"], r["condition"], r["n_common"], _pct(r["rate_cond"]), _pct(r["rate_base"]),
                 _pct(r["diff"], r["diff_lo"], r["diff_hi"]), _pval(r["p_boot"]),
                 f'{r["only_cond"]}/{r["only_base"]}', _pval(r["p_mcnemar"]), _pval(r["p_holm"])]
                for r in paired.to_dict("records")]
        parts += [_md_table(["benchmark", "condition", "n", "rate (%)", "base (%)", "diff (pts)", "p boot",
                             "only cond/base", "p McNemar", "p Holm"], rows), ""]
    if detect is not None and not detect.empty:
        parts += ["## Detectable effect", "",
                  "The smallest gain in solve rate each comparison would find with 80% power (exact McNemar, "
                  "two-sided, alpha 0.05, no Holm correction), at its number of problems and the share of "
                  "problems the two conditions disagree on so far. A dash: too few problems disagree for "
                  "any gain to reach 80%.", ""]
        rows = [[r["benchmark"], r["condition"], r["n_common"], _pct(r["discordance"]), _pct(r["mde"])]
                for r in detect.to_dict("records")]
        parts += [_md_table(["benchmark", "condition", "n", "disagree (%)", "detectable gain (pts)"], rows), ""]
    if areas is not None and not areas.empty:
        parts += [f"## pass@1 by area (`{area_col}`)", ""]
        rows = [[r["benchmark"], r["area"], r["condition"], r["n"], _pct(r["pass@1"], r["pass@1_lo"], r["pass@1_hi"])]
                for r in areas.to_dict("records")]
        parts += [_md_table(["benchmark", "area", "condition", "n", "pass@1 (%)"], rows), ""]
    if gains is not None and not gains.empty:
        parts += [f"## Gain over `{baseline}` by area (`{area_col}`)", "",
                  "Mean per-problem pass@1 difference in points on the area's problems both ran, with a "
                  "paired bootstrap CI. Small areas have wide intervals; no multiple-comparison correction.", ""]
        rows = [[r["benchmark"], r["area"], r["condition"], r["n"], _pts(r["diff"], r["diff_lo"], r["diff_hi"]),
                 f'{r["only_cond"]}/{r["only_base"]}'] for r in gains.to_dict("records")]
        parts += [_md_table(["benchmark", "area", "condition", "n", "gain (pts)", "only cond/base"], rows), ""]
    if stats is not None and not stats.empty:
        parts += ["## Proof length and search effort", "",
                  "Over runs. Length is in tactics, over solved runs; stops are shares of all runs.", ""]
        rows = [[r["benchmark"], r["condition"], r["n_runs"], _pct(r["solve_rate"]),
                 f'{_num(r["len_median"], "{:.0f}")} [{_num(r["len_q25"], "{:.0f}")}, {_num(r["len_q75"], "{:.0f}")}]',
                 _num(r["len_mean"], "{:.2f}"), _num(r["exp_solved_median"], "{:.0f}"), _num(r["nodes_mean"]),
                 _num(r["nodes_per_expansion"], "{:.2f}"), _pct(r["stop_budget"]), _pct(r["stop_wall_time"]),
                 _pct(r["stop_exhausted"]), _pct(r["stop_statement_error"])]
                for r in stats.to_dict("records")]
        parts += [_md_table(["benchmark", "condition", "runs", "solved (%)", "length median [IQR]", "length mean",
                             "exp. to solve (median)", "nodes/run", "nodes/exp.", "budget (%)", "wall time (%)",
                             "exhausted (%)", "stmt error (%)"], rows), ""]
    if effort is not None and not effort.empty:
        parts += [f"## Effort on runs both solved (against `{baseline}`)", "",
                  "Pairs are (problem, seed) runs solved under both conditions. The expansion ratio is the "
                  "geometric mean over problems of condition / baseline expansions (below 1: fewer prover "
                  "calls); Wilcoxon is on the per-problem log ratios. Only problems both solve count, so "
                  "this is about speed, not coverage.", ""]
        rows = [[r["benchmark"], r["condition"], r["n_pairs"], r["n_problems"],
                 f'{_num(r["exp_cond_median"], "{:.0f}")} / {_num(r["exp_base_median"], "{:.0f}")}',
                 (f'{r["exp_ratio"]:.2f} [{r["exp_ratio_lo"]:.2f}, {r["exp_ratio_hi"]:.2f}]'
                  if np.isfinite(r["exp_ratio_lo"]) else _num(r["exp_ratio"], "{:.2f}")),
                 (f'{r["len_diff"]:+.2f} [{r["len_diff_lo"]:+.2f}, {r["len_diff_hi"]:+.2f}]'
                  if np.isfinite(r["len_diff_lo"]) else _num(r["len_diff"], "{:+.2f}")),
                 _pval(r["p_wilcoxon"])] for r in effort.to_dict("records")]
        parts += [_md_table(["benchmark", "condition", "pairs", "problems", "median exp. cond/base",
                             "exp. ratio", "length diff (tactics)", "p Wilcoxon"], rows), ""]
    return "\n".join(parts)


def write_report(root: str | Path | Sequence[str | Path], out_dir: str | Path, baseline: str = "lm",
                 ks: Sequence[int] | None = None, area_col: str | None = None, n_boot: int = 2000, seed: int = 0,
                 figures: bool = True, labels: dict[str, str] | None = None,
                 bench_labels: dict[str, str] | None = None) -> dict[str, Path]:
    """Load the runs under ``root`` (one path or several) and write every table and figure to ``out_dir``.

    Returns the paths by name. Always writes ``pass_at_k.csv``, ``paired.csv``,
    ``budget.csv``, ``budget_gain.csv``, ``search_stats.csv``, ``effort.csv``,
    ``report.md`` and LaTeX ``pass_at_k.tex``, ``paired.tex`` and
    ``search_stats.tex``, plus ``pass_at_k_all.tex`` when more than one k is
    reported. When an area column is available (``area_col``, else
    ``root_area``, else ``name_area``, whichever has values) also
    ``by_area.csv``, ``by_area_gain.csv`` and ``by_area_gain.tex``. With
    ``figures``: ``fig_pass_at_1.pdf``, ``fig_budget.pdf``,
    ``fig_budget_gain.pdf``, ``fig_effort.pdf`` and, per benchmark,
    ``fig_by_area_<benchmark>.pdf`` and ``fig_area_gain_<benchmark>.pdf``
    (the gain figures only when a benchmark has the baseline and another
    condition). ``ks`` defaults to the powers of two up to 32 that every
    condition has seeds for. Conditions keep one order (baseline first, then
    first appearance) and so one color. ``labels`` and ``bench_labels`` map
    condition and benchmark names to raw LaTeX for the ``.tex`` tables
    (benchmarks default to miniF2F, PutnamBench, ProofNet).
    """
    df = load_summaries(root)
    if df.empty:
        raise FileNotFoundError(f"no */*/seed*/summary.jsonl under {root}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    conds = list(dict.fromkeys(df["condition"]))
    if baseline in conds:
        conds = [baseline] + [c for c in conds if c != baseline]
    benches = list(dict.fromkeys(df["benchmark"]))
    order = df["benchmark"].map(benches.index) * len(conds) + df["condition"].map(conds.index)
    df = df.iloc[np.argsort(order.to_numpy(), kind="stable")].reset_index(drop=True)
    tex = _Labels(labels, bench_labels)

    if ks is None:
        min_seeds = df.groupby(["benchmark", "condition"])["seed"].nunique().min()
        ks = [k for k in DEFAULT_KS if k <= min_seeds]
    ks = sorted(set(int(k) for k in ks)) or [1]
    if 1 not in ks:
        ks = [1] + ks
    table = pass_table(df, ks=ks, n_boot=n_boot, seed=seed)
    ks = [k for k in ks if f"pass@{k}" in table.columns]
    paired = paired_compare(df, baseline=baseline, n_boot=n_boot, seed=seed)
    acol = _pick_area_col(df, area_col)
    areas = area_table(df, acol, n_boot=n_boot, seed=seed) if acol else None
    gains = area_gain_table(df, acol, baseline=baseline, n_boot=n_boot, seed=seed) if acol else None
    curve = budget_curve(df, n_boot=n_boot, seed=seed)
    gain_curve = budget_gain_curve(df, baseline=baseline, n_boot=n_boot, seed=seed)
    stats = search_stats(df)
    effort = effort_compare(df, baseline=baseline, n_boot=n_boot, seed=seed)

    paths: dict[str, Path] = {}
    frames = (("pass_at_k", table), ("paired", paired), ("by_area", areas), ("by_area_gain", gains),
              ("budget", curve), ("budget_gain", gain_curve), ("search_stats", stats), ("effort", effort))
    for name, frame in frames:
        if frame is not None:
            paths[name] = out / f"{name}.csv"
            frame.to_csv(paths[name], index=False)
    note = local_model_note(df, root)
    detect = detectable_effects(paired)
    paths["detectable"] = out / "detectable.csv"
    detect.to_csv(paths["detectable"], index=False)
    paths["report"] = out / "report.md"
    paths["report"].write_text(_report_md(table, paired, areas, ks, baseline, acol, n_boot,
                                          gains=gains, stats=stats, effort=effort, note=note,
                                          detect=detect))
    texts = {"latex": ("pass_at_k.tex", _latex_table(table, conds, labels=tex)),
             "latex_paired": ("paired.tex", _latex_paired(paired, tex)),
             "latex_stats": ("search_stats.tex", _latex_stats(stats, effort, tex))}
    if len(ks) > 1:
        texts["latex_pass_all"] = ("pass_at_k_all.tex", _latex_pass_multi(table, conds, ks, tex))
    if gains is not None:
        texts["latex_area_gain"] = ("by_area_gain.tex", _latex_area_gain(gains, conds, tex))
    for name, (fname, text) in texts.items():
        paths[name] = out / fname
        paths[name].write_text((f"% {note}\n" if note else "") + text)
    if note:
        paths["local_note"] = out / "LOCAL_MODEL.txt"
        paths["local_note"].write_text(note + "\n")

    if figures:
        import matplotlib.pyplot as plt

        from markov_lean.figures import save_pdf

        made = [("fig_pass_at_1", lambda: fig_pass_at_1(table)), ("fig_budget", lambda: fig_budget(curve, conds)),
                ("fig_effort", lambda: fig_effort(df, conds))]
        if not gain_curve.empty:
            made.append(("fig_budget_gain", lambda: fig_budget_gain(gain_curve, conds)))
        if areas is not None and not areas.empty:
            for bench in dict.fromkeys(areas["benchmark"]):
                made.append((f"fig_by_area_{bench}", lambda b=bench: fig_by_area(areas, b)))
        if gains is not None and not gains.empty:
            for bench in dict.fromkeys(gains["benchmark"]):
                made.append((f"fig_area_gain_{bench}", lambda b=bench: fig_area_gain(gains, b, conds)))
        for name, make in made:
            fig = make()
            paths[name] = save_pdf(fig, out / f"{name}.pdf")
            plt.close(fig)
    return paths


def _mapping(items: Sequence[str] | None, flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"{flag} wants NAME=LABEL, got {item!r}")
        out[key] = value
    return out


def main(argv: list[str] | None = None) -> int:
    """``search-report``: tables, tests and figures for the runs under ``--runs``."""
    p = argparse.ArgumentParser(prog="markov-lean search-report", description=__doc__.splitlines()[0])
    p.add_argument("--runs", required=True, nargs="+",
                   help="one or more of: a root holding <benchmark>/<condition>/seed<k>/summary.jsonl, a "
                        "benchmark, condition or seed folder, or a summary.jsonl (a shell glob works)")
    p.add_argument("--out", default=None, help="output folder (default <common parent of --runs>/report)")
    p.add_argument("--baseline", default="lm", help="condition the others are compared with")
    p.add_argument("--k", type=int, action="append", default=None, help="pass@k to report (repeatable)")
    p.add_argument("--area-col", default=None, help="area column (default root_area, else name_area)")
    p.add_argument("--label", action="append", default=None, metavar="CONDITION=LATEX",
                   help="row label for a condition in the .tex tables (raw LaTeX; repeatable)")
    p.add_argument("--bench-label", action="append", default=None, metavar="BENCHMARK=LATEX",
                   help="column label for a benchmark in the .tex tables (raw LaTeX; repeatable)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    labels = _mapping(args.label, "--label")
    bench_labels = _mapping(args.bench_label, "--bench-label")
    if args.out:
        out = Path(args.out)
    else:
        dirs = [Path(r) if Path(r).is_dir() else Path(r).parent for r in args.runs]
        out = Path(os.path.commonpath([str(d.resolve()) for d in dirs])) / "report"
    if not args.no_figures:
        import matplotlib

        if matplotlib.get_backend().lower() not in ("agg", "pdf"):
            matplotlib.use("Agg")
    paths = write_report(args.runs, out, baseline=args.baseline, ks=args.k, area_col=args.area_col,
                         n_boot=args.n_boot, seed=args.seed, figures=not args.no_figures, labels=labels,
                         bench_labels=bench_labels)
    print(paths["report"])
    if "local_note" in paths:
        print(paths["local_note"].read_text().strip())
    table = pd.read_csv(paths["pass_at_k"])
    show = table[["benchmark", "condition", "n_problems", "n_seeds", "pass@1", "pass@1_lo", "pass@1_hi"]].copy()
    for c in ("pass@1", "pass@1_lo", "pass@1_hi"):
        show[c] = (show[c] * 100).round(1)
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
