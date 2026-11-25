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
* ``area_table`` / ``budget_curve``: pass@1 by mathematical area, and the share
  of runs solved within each expansion budget.
* ``write_report``: all of the above as CSV, Markdown, a booktabs LaTeX table
  and paper PDFs. ``main`` is the command-line entry point.

matplotlib and seaborn are only needed for the figures and are imported lazily.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

KEY = ["benchmark", "condition", "seed", "proof_id"]
COLUMNS = [
    "proof_id", "theorem", "source", "split", "solved", "stop_reason", "n_tactics",
    "expansions", "nodes", "elapsed", "proof", "benchmark", "condition", "seed",
    "max_expansions", "wall_time", "root_area", "name_area",
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
    "write_report",
    "main",
]


# ---------------------------------------------------------------- loading
def load_summaries(root: str | Path) -> pd.DataFrame:
    """Every ``<root>/*/*/seed*/summary.jsonl`` line as one row, deduplicated on (benchmark, condition, seed, proof_id).

    ``benchmark``, ``condition`` and ``seed`` come from the line, or from the
    path when the line lacks them. Blank or truncated lines (a crash mid-write)
    are skipped. Missing optional columns are added as nulls.
    """
    root = Path(root)
    rows: list[dict] = []
    for path in sorted(root.glob("*/*/seed*/summary.jsonl")):
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
    for c in ("expansions", "elapsed", "max_expansions", "wall_time"):
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

    def area_key(s: pd.Series) -> pd.Series:  # numeric areas in numeric order, then names, "unknown" last
        num = pd.to_numeric(s, errors="coerce")
        return pd.Series([(2, 0.0, a) if a == "unknown" else (0, n, "") if np.isfinite(n) else (1, 0.0, a)
                          for a, n in zip(s, num)], index=s.index)

    res = res.sort_values("area", key=area_key, kind="stable")
    return res.sort_values("benchmark", kind="stable").reset_index(drop=True)


def _default_grid(max_e: int, n: int = 24) -> np.ndarray:
    max_e = max(int(max_e), 1)
    grid = np.unique(np.round(np.logspace(0, np.log10(max_e), n)).astype(int))
    return np.unique(np.append(grid, max_e))


def budget_curve(df: pd.DataFrame, grid: Sequence[int] | None = None) -> pd.DataFrame:
    """Share of problem-seed runs solved within ``expansions <= e``, per (benchmark, condition, e).

    The default grid is about 24 log-spaced integers from 1 up to the largest
    ``max_expansions`` (or the largest ``expansions`` when that is missing),
    always including the maximum.
    """
    cols = ["benchmark", "condition", "budget", "solved_frac", "n_runs"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    if grid is None:
        cap = pd.to_numeric(df["max_expansions"], errors="coerce").max() if "max_expansions" in df else np.nan
        if not np.isfinite(cap):
            cap = pd.to_numeric(df["expansions"], errors="coerce").max()
        grid = _default_grid(int(cap) if np.isfinite(cap) else 1)
    grid = np.asarray(sorted(set(int(g) for g in grid)))
    out = []
    for bench, cond in _conditions(df):
        sub = df[(df["benchmark"] == bench) & (df["condition"] == cond)]
        e = pd.to_numeric(sub["expansions"], errors="coerce").to_numpy(float)
        s = sub["solved"].to_numpy(bool)
        for g in grid:
            out.append({"benchmark": bench, "condition": cond, "budget": int(g),
                        "solved_frac": float((s & (e <= g)).mean()) if len(s) else float("nan"),
                        "n_runs": len(s)})
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
    in ``curve``). The benchmark names the y axis of each panel; no title.
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


# ---------------------------------------------------------------- writers
def _pct(p: float, lo: float | None = None, hi: float | None = None) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    s = f"{100 * p:.1f}"
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{100 * lo:.1f}, {100 * hi:.1f}]"
    return s


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


def _latex_table(table: pd.DataFrame, conditions: Sequence[str], k: int = 1) -> str:
    """booktabs table: one row per condition, one column per benchmark, ``xx.x [lo, hi]`` percent."""
    benches = list(dict.fromkeys(table["benchmark"]))
    col = f"pass@{k}"
    lines = [
        r"\begin{tabular}{l" + "c" * len(benches) + "}",
        r"\toprule",
        "Condition & " + " & ".join(_tex_escape(b) for b in benches) + r" \\",
        r"\midrule",
    ]
    idx = table.set_index(["condition", "benchmark"])
    for cond in conditions:
        cells = []
        for b in benches:
            if (cond, b) in idx.index and col in idx.columns:
                r = idx.loc[(cond, b)]
                cells.append(_pct(r[col], r[f"{col}_lo"], r[f"{col}_hi"]))
            else:
                cells.append("--")
        lines.append(_tex_escape(cond) + " & " + " & ".join(cells) + r" \\")
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
               ks: Sequence[int], baseline: str, area_col: str | None, n_boot: int) -> str:
    parts = ["# Search report", ""]
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
    if areas is not None and not areas.empty:
        parts += [f"## pass@1 by area (`{area_col}`)", ""]
        rows = [[r["benchmark"], r["area"], r["condition"], r["n"], _pct(r["pass@1"], r["pass@1_lo"], r["pass@1_hi"])]
                for r in areas.to_dict("records")]
        parts += [_md_table(["benchmark", "area", "condition", "n", "pass@1 (%)"], rows), ""]
    return "\n".join(parts)


def write_report(root: str | Path, out_dir: str | Path, baseline: str = "lm", ks: Sequence[int] | None = None,
                 area_col: str | None = None, n_boot: int = 2000, seed: int = 0,
                 figures: bool = True) -> dict[str, Path]:
    """Load the runs under ``root`` and write every table and figure to ``out_dir``; return the paths by name.

    Writes ``pass_at_k.csv``, ``paired.csv``, ``budget.csv``, ``report.md``,
    ``pass_at_k.tex`` and, when an area column is available (``area_col``, else
    ``root_area``, else ``name_area``, whichever has values), ``by_area.csv``.
    With ``figures``: ``fig_pass_at_1.pdf``, ``fig_budget.pdf`` and one
    ``fig_by_area_<benchmark>.pdf`` per benchmark. ``ks`` defaults to the
    powers of two up to 32 that every condition has seeds for. Conditions keep
    one order (baseline first, then first appearance) and so one color.
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
    curve = budget_curve(df)

    paths: dict[str, Path] = {}
    for name, frame in (("pass_at_k", table), ("paired", paired), ("by_area", areas), ("budget", curve)):
        if frame is not None:
            paths[name] = out / f"{name}.csv"
            frame.to_csv(paths[name], index=False)
    paths["report"] = out / "report.md"
    paths["report"].write_text(_report_md(table, paired, areas, ks, baseline, acol, n_boot))
    paths["latex"] = out / "pass_at_k.tex"
    paths["latex"].write_text(_latex_table(table, conds))

    if figures:
        import matplotlib.pyplot as plt

        from markov_lean.figures import save_pdf

        made = [("fig_pass_at_1", fig_pass_at_1(table)), ("fig_budget", fig_budget(curve, conds))]
        if areas is not None and not areas.empty:
            for bench in dict.fromkeys(areas["benchmark"]):
                made.append((f"fig_by_area_{bench}", fig_by_area(areas, bench)))
        for name, fig in made:
            paths[name] = save_pdf(fig, out / f"{name}.pdf")
            plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    """``search-report``: tables, tests and figures for the runs under ``--runs``."""
    p = argparse.ArgumentParser(prog="markov-lean search-report", description=__doc__.splitlines()[0])
    p.add_argument("--runs", required=True, help="root holding <benchmark>/<condition>/seed<k>/summary.jsonl")
    p.add_argument("--out", default=None, help="output folder (default <runs>/report)")
    p.add_argument("--baseline", default="lm", help="condition the others are compared with")
    p.add_argument("--k", type=int, action="append", default=None, help="pass@k to report (repeatable)")
    p.add_argument("--area-col", default=None, help="area column (default root_area, else name_area)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    out = Path(args.out) if args.out else Path(args.runs) / "report"
    if not args.no_figures:
        import matplotlib

        if matplotlib.get_backend().lower() not in ("agg", "pdf"):
            matplotlib.use("Agg")
    paths = write_report(args.runs, out, baseline=args.baseline, ks=args.k, area_col=args.area_col,
                         n_boot=args.n_boot, seed=args.seed, figures=not args.no_figures)
    print(paths["report"])
    table = pd.read_csv(paths["pass_at_k"])
    show = table[["benchmark", "condition", "n_problems", "n_seeds", "pass@1", "pass@1_lo", "pass@1_hi"]].copy()
    for c in ("pass@1", "pass@1_lo", "pass@1_hi"):
        show[c] = (show[c] * 100).round(1)
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
