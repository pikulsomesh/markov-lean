"""Cross-dataset figures for the full-scale run, in the paper style of ``markov_lean.figures``.

Each function takes the ``DatasetResult`` list from ``markov_lean.project``
and returns a Figure; ``figures.save_all`` writes them.  Datasets keep their
order, and dataset ``i`` keeps color slot ``i`` in every figure.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .figures import COLUMN_WIDTH, TEXT_SECONDARY, TEXT_WIDTH, _blue_cmap, _new_ax, _sns, _style_colorbar, \
    _themed, color_for
from .project import DatasetResult, interval, proof_stats_frame

__all__ = ["switch_rates", "occupancy", "dwell_times", "switches_by_length"]


def _with_moves(results: Sequence[DatasetResult]) -> list[tuple[int, DatasetResult]]:
    return [(i, r) for i, r in enumerate(results) if r.has_transitions]


@_themed
def switch_rates(results: Sequence[DatasetResult], *, statistic: str = "pooled_switch_rate", ax=None,
                 figsize=(COLUMN_WIDTH, 2.4)):
    """One point per dataset with its 95% bootstrap interval; datasets without moves are left out."""
    rows = _with_moves(results)
    fig, ax = _new_ax(ax, figsize)
    for y, (i, r) in enumerate(rows):
        est = float(r.point[statistic])
        lo, hi = interval(r.draws[statistic]) if statistic in r.draws else (est, est)
        ax.errorbar(est, y, xerr=[[est - float(lo)], [float(hi) - est]], fmt="o", ms=4, mew=0,
                    color=color_for(i), ecolor=color_for(i), elinewidth=1.2, capsize=2)
    ax.set_yticks(range(len(rows)), [r.name for _, r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("area switches per move" if statistic == "pooled_switch_rate" else statistic.replace("_", " "))
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)
    return fig


@_themed
def occupancy(results: Sequence[DatasetResult], *, annotate: bool = False, ax=None, figsize=None):
    """Share of goal states in each macrostate, one row per dataset."""
    sns = _sns()
    m = len(results[0].point["occupancy"]) if results else 0
    frame = pd.DataFrame([np.nan_to_num(r.point["occupancy"]) for r in results],
                         index=pd.Index([r.name for r in results], name="dataset"),
                         columns=pd.Index([str(k) for k in range(m)], name="macrostate"))
    if figsize is None:
        figsize = (COLUMN_WIDTH if m <= 8 else TEXT_WIDTH, 0.3 * len(results) + 1.0)
    fig, ax = _new_ax(ax, figsize)
    sns.heatmap(frame, ax=ax, cmap=_blue_cmap(), vmin=0, vmax=max(float(frame.to_numpy().max()), 1e-9),
                annot=annotate, fmt=".2f", annot_kws={"fontsize": "x-small"}, linewidths=1, linecolor="white",
                cbar_kws={"shrink": 0.85, "label": "share of goal states"})
    ax.tick_params(length=0)
    ax.set_ylabel("")
    _style_colorbar(ax)
    return fig


@_themed
def dwell_times(results: Sequence[DatasetResult], *, max_dwell: int = 30, ax=None, figsize=(COLUMN_WIDTH, 2.4)):
    """Share of runs in one macrostate that last at least d steps (all runs, censored ones included)."""
    fig, ax = _new_ax(ax, figsize)
    for i, r in _with_moves(results):
        runs = r.tables.runs
        moved = set(r.tables.proof_ids[r.tables.n_moves > 0].astype(str))
        d = runs.loc[runs["proof_id"].astype(str).isin(moved), "dwell"].to_numpy()
        if not len(d):
            continue
        grid = np.arange(1, min(max_dwell, int(d.max())) + 1)
        surv = (d[None, :] >= grid[:, None]).mean(1)
        ax.step(grid, surv, where="post", color=color_for(i), lw=1.5, label=r.name)
    ax.set_yscale("log")
    ax.set_xlabel("dwell d (proof steps)")
    ax.set_ylabel("share of runs lasting $\\geq$ d")
    ax.grid(axis="y")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize="x-small")
    return fig


@_themed
def switches_by_length(results: Sequence[DatasetResult], *, bins: Sequence[int] = (2, 4, 8, 12, 16, 24, 32, 64),
                       ax=None, figsize=(COLUMN_WIDTH, 2.4)):
    """Mean per-proof switch rate against proof length, so datasets of different lengths compare fairly."""
    fig, ax = _new_ax(ax, figsize)
    edges = np.asarray(list(bins) + [np.inf])
    for i, r in _with_moves(results):
        ps = proof_stats_frame(r.tables)
        ps = ps[ps["n_moves"] > 0]
        b = np.digitize(ps["n_steps"], edges) - 1
        g = ps.assign(bin=b)[b >= 0].groupby("bin")["switch_rate"]
        mean, n = g.mean(), g.size()
        keep = n[n >= 5].index
        if not len(keep):
            continue
        sem = (g.std() / np.sqrt(n)).loc[keep]
        x = edges[keep]
        ax.errorbar(x, mean.loc[keep], yerr=1.96 * sem.fillna(0), fmt="o-", ms=3, mew=0, lw=1.2,
                    color=color_for(i), ecolor=color_for(i), elinewidth=1, capsize=1.5, label=r.name)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("proof length (steps, bin start)")
    ax.set_ylabel("switch rate per move")
    ax.grid(axis="y")
    ax.tick_params(axis="x", colors=TEXT_SECONDARY)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize="x-small")
    return fig
