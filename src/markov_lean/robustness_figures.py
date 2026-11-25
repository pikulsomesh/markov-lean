"""Figures for ``markov-lean robustness``, drawn in the paper style of ``figures.py``.

Each function takes the tables ``robustness.run`` writes and returns a Figure;
:func:`save_figures` writes them all as PDFs.  No titles inside the graphic
(ICML puts captions under figures).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .figures import (COLUMN_WIDTH, GRID, TEXT_SECONDARY, TEXT_WIDTH, _blue_cmap, _new_ax, _plt, _sns,
                      _style_colorbar, _themed, color_for, save_pdf)

GROUP_NAMES = {
    "macrostate": "macrostates (MSM)",
    "microstate": "microstates",
    "folder": "Mathlib folder",
    "folder2": "Mathlib folder (2 levels)",
    "namespace": "namespace",
    "namespace2": "namespace (2 levels)",
    "kmeans_steps": "k-means, no dynamics",
    "kmeans_microstates": "k-means on microstates",
    "proof_label": "judges' proof label",
    "random": "random",
}


def _pilot_slice(sweep: pd.DataFrame, setting: dict) -> pd.DataFrame:
    s = sweep
    for col, key in (("featurizer", "featurizer"), ("n_microstates", "n_microstates"), ("lag", "lag")):
        if col in s and key in setting:
            s = s[s[col] == setting[key]]
    return s.sort_values("k")


@_themed
def nmi_by_k(sweep: pd.DataFrame, kmeans: pd.DataFrame, baselines: pd.DataFrame, setting: dict,
             ceiling: float | None = None, null_q95: float | None = None, *, metric: str = "nmi", ax=None,
             figsize=(COLUMN_WIDTH, 2.4)):
    """Agreement with the judges against the number of macrostates.

    The macrostate line is the pilot's own setting; the band spans every
    other featurizer, microstate count and lag in the sweep.  Static
    groupings with a fixed number of groups are horizontal lines.
    """
    fig, ax = _new_ax(ax, figsize)
    if len(sweep) and metric in sweep:
        band = sweep.groupby("k")[metric].agg(["min", "max"]).reset_index()
        ax.fill_between(band["k"], band["min"], band["max"], color=color_for(0), alpha=0.15, lw=0,
                        label="MSM, all settings")
        own = _pilot_slice(sweep, setting)
        if len(own):
            ax.plot(own["k"], own[metric], color=color_for(0), lw=2, marker="o", ms=3, label="MSM, pilot setting")
    if len(kmeans) and metric in kmeans:
        ax.plot(kmeans["k"], kmeans[metric], color=color_for(1), lw=2, marker="o", ms=3, label="k-means, no dynamics")
    b = baselines.set_index("grouping")
    for i, g in enumerate(("folder", "namespace", "proof_label")):
        if g in b.index and np.isfinite(b.loc[g, metric]):
            ax.axhline(b.loc[g, metric], color=color_for(2 + i), lw=1.2, ls="--", label=GROUP_NAMES[g])
    if ceiling is not None and np.isfinite(ceiling):
        ax.axhline(ceiling, color=TEXT_SECONDARY, lw=1, ls=":", label="judge vs judge")
    if null_q95 is not None and np.isfinite(null_q95):
        ax.axhspan(0, null_q95, color=GRID, lw=0, zorder=0, label="null, 95th pct")
    ax.set_xlabel("macrostates $k$")
    ax.set_ylabel(metric.upper() + " with judge areas")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2, handlelength=1.5,
              columnspacing=1.0, fontsize="x-small")
    return fig


@_themed
def baseline_bars(baselines: pd.DataFrame, *, metric: str = "nmi", ax=None, figsize=(COLUMN_WIDTH, 2.6)):
    """Each grouping's agreement with the judges, with 95% bootstrap intervals over proofs."""
    d = baselines.sort_values(metric).reset_index(drop=True)
    fig, ax = _new_ax(ax, figsize)
    y = np.arange(len(d))
    colors = [color_for(0) if g == "macrostate" else "#9c9c9c" for g in d["grouping"]]
    ax.barh(y, d[metric], color=colors, height=0.7)
    lo, hi = d.get(f"{metric}_lo"), d.get(f"{metric}_hi")
    if lo is not None and hi is not None:
        err = np.vstack([d[metric] - lo, hi - d[metric]]).clip(min=0)
        ax.errorbar(d[metric], y, xerr=err, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1, capsize=2)
    ax.set_yticks(y, [GROUP_NAMES.get(g, g) for g in d["grouping"]])
    ax.set_xlabel(metric.upper() + " with judge areas")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    return fig


@_themed
def sweep_heatmaps(sweep: pd.DataFrame, k: int, *, value: str = "nmi", label: str | None = None,
                   figsize=None):
    """One heatmap per featurizer: ``value`` at ``k`` macrostates over microstate count and lag."""
    _, plt = _plt()
    sns = _sns()
    s = sweep[sweep["k"] == k]
    feats = list(dict.fromkeys(s["featurizer"]))
    n = max(len(feats), 1)
    fig, axes = plt.subplots(1, n, figsize=figsize or (TEXT_WIDTH, 1.9), layout="constrained", squeeze=False)
    vmin, vmax = (s[value].min(), s[value].max()) if len(s) else (0, 1)
    for ax, f in zip(axes[0], feats):
        piv = s[s["featurizer"] == f].pivot_table(index="n_microstates", columns="lag", values=value)
        sns.heatmap(piv, ax=ax, cmap=_blue_cmap(), vmin=vmin, vmax=vmax, annot=True, fmt=".2f",
                    annot_kws={"fontsize": 6}, cbar=ax is axes[0][-1], linewidths=0.5, linecolor="white",
                    cbar_kws={"label": label or value})
        _style_colorbar(ax)
        ax.set_xlabel("lag (steps)")
        ax.set_ylabel("microstates" if ax is axes[0][0] else "")
        ax.text(0.5, 1.02, f, transform=ax.transAxes, ha="center", va="bottom", fontsize="small")
    return fig


@_themed
def ck_by_lag(sweep: pd.DataFrame, setting: dict, tol: float, ks: list[int] | None = None, *, ax=None,
              figsize=(COLUMN_WIDTH, 2.2)):
    """Worst Chapman-Kolmogorov error against lag, one line per macrostate count."""
    s = sweep
    for col in ("featurizer", "n_microstates"):
        if col in setting:
            s = s[s[col] == setting[col]]
    ks = ks or sorted(s["k"].unique())[:: max(1, len(s["k"].unique()) // 5)]
    fig, ax = _new_ax(ax, figsize)
    for i, k in enumerate(ks):
        d = s[s["k"] == k].sort_values("lag")
        ax.plot(d["lag"], d["ck_max_error"], marker="o", ms=3, lw=1.5, color=color_for(i), label=f"k={k}")
    ax.axhline(tol, color=TEXT_SECONDARY, ls="--", lw=1)
    ax.set_xlabel("lag (proof steps)")
    ax.set_ylabel("max CK error")
    ax.grid(axis="y")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=min(len(ks), 5), borderaxespad=0.2,
              handlelength=1.2, columnspacing=0.8, fontsize="x-small")
    return fig


@_themed
def memory_gains(memory: pd.DataFrame, *, ax=None, figsize=(COLUMN_WIDTH, 2.2)):
    """Held-out log-likelihood gain of each order over the one below it, per sequence."""
    d = memory.dropna(subset=["gain_vs_previous"]).copy()
    fig, ax = _new_ax(ax, figsize)
    seqs = list(dict.fromkeys(d["sequence"]))
    orders = sorted(d["order"].unique())
    w = 0.8 / max(len(seqs), 1)
    for i, sq in enumerate(seqs):
        e = d[d["sequence"] == sq].set_index("order").reindex(orders)
        x = np.arange(len(orders)) + (i - (len(seqs) - 1) / 2) * w
        ax.bar(x, e["gain_vs_previous"], width=w, color=color_for(i), label=sq)
        err = np.vstack([e["gain_vs_previous"] - e["gain_lo"], e["gain_hi"] - e["gain_vs_previous"]]).clip(min=0)
        ax.errorbar(x, e["gain_vs_previous"], yerr=err, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1, capsize=2)
    ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
    ax.set_xticks(np.arange(len(orders)), [f"{o} vs {o - 1}" for o in orders])
    ax.set_xlabel("order")
    ax.set_ylabel("held-out log-lik. gain (nats/step)")
    ax.grid(axis="y")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=min(len(seqs), 3), borderaxespad=0.2,
              fontsize="x-small", handlelength=1.2)
    return fig


def save_figures(summary: dict, sweep: pd.DataFrame, outdir: str | Path) -> list[Path]:
    """Write every robustness figure that has data, as PDFs in ``outdir``."""
    _, plt = _plt()
    out = Path(outdir)
    base = pd.DataFrame(summary["baselines"]["table"])
    kmeans = pd.DataFrame(summary.get("kmeans_by_k", []))
    ss = summary.get("sweep_summary") or {}
    setting = ss.get("pilot_setting") or {"k": summary["baselines"]["coverage"]["k"]}
    ceiling = (summary.get("judge_ceiling") or {}).get("nmi")
    null_q95 = summary["baselines"]["null"]["proof"]["nmi_q95"]
    mem = pd.DataFrame([
        {"sequence": m["sequence"], "order": int(o), "gain_vs_previous": (m.get(f"gain_{o}_vs_{int(o) - 1}") or {})
         .get("mean"), "gain_lo": ((m.get(f"gain_{o}_vs_{int(o) - 1}") or {}).get("ci") or [np.nan])[0],
         "gain_hi": ((m.get(f"gain_{o}_vs_{int(o) - 1}") or {}).get("ci") or [np.nan, np.nan])[1]}
        for m in summary.get("memory", []) if "orders" in m for o in m["orders"] if int(o) > 0])
    figs = [("baselines_nmi", baseline_bars(base, metric="nmi")),
            ("baselines_ari", baseline_bars(base, metric="ari"))]
    figs.append(("nmi_by_k", nmi_by_k(sweep, kmeans, base, setting, ceiling, null_q95)))
    figs.append(("ari_by_k", nmi_by_k(sweep, kmeans, base, setting, (summary.get("judge_ceiling") or {})
                                      .get("ari"), None, metric="ari")))
    if len(sweep) and "nmi" in sweep:
        k = int(setting["k"])
        if (sweep["k"] == k).any():
            figs.append(("sweep_nmi", sweep_heatmaps(sweep, k, value="nmi", label="NMI")))
            figs.append(("sweep_ck_error", sweep_heatmaps(sweep, k, value="ck_max_error", label="max CK error")))
        figs.append(("ck_by_lag", ck_by_lag(sweep, setting, float(summary["config"]["ck_tol"]))))
    if len(mem):
        figs.append(("memory_gain", memory_gains(mem)))
    paths = []
    for name, fig in figs:
        paths.append(save_pdf(fig, out / name))
        plt.close(fig)
    return paths
