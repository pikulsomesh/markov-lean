"""Paper figures over the shared tables in ``docs/schema.md``.

Every public plotting function takes arrays or DataFrames, draws on a new
Figure (or on an ``ax`` you pass), and returns the Figure. Nothing here reads
files or runs the pipeline; call :func:`save_pdf` to write a figure.

The palette is the validated default categorical order (light mode, since the
paper prints on white) and a one-hue blue ramp for magnitudes. Colors follow
the entity: macrostate ``i`` always gets slot ``i`` so a figure that drops a
state does not repaint the rest.

matplotlib is an optional dependency (the ``figures`` extra), so it is imported
lazily inside each function.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Validated categorical order, light surface (dataviz reference palette).
CATEGORICAL = (
    "#257de0",  # blue
    "#f47230",  # orange
    "#00b894",  # aqua
    "#f2a600",  # yellow
    "#ee7fac",  # magenta
    "#008a17",  # green
    "#6a4ce2",  # violet
    "#ec4647",  # red
)
OTHER = "#9c9c9c"
TEXT = "#191919"
TEXT_SECONDARY = "#666666"
GRID = "#e5e5e5"
# One-hue blue ramp, light to dark, for heatmaps.
BLUE_RAMP = ("#f4f8fe", "#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e")

__all__ = [
    "CATEGORICAL",
    "color_for",
    "nmi",
    "area_transitions",
    "implied_timescales",
    "chapman_kolmogorov",
    "transition_heatmap",
    "judge_cluster_confusion",
    "area_flow",
    "search_ablation",
    "save_pdf",
]


# ---------------------------------------------------------------- helpers
def _plt():
    import matplotlib

    import matplotlib.pyplot as plt

    return matplotlib, plt


def color_for(i: int) -> str:
    """Categorical color for entity ``i``; past the eighth slot everything is "Other" gray."""
    return CATEGORICAL[i] if 0 <= i < len(CATEGORICAL) else OTHER


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def _new_ax(ax, figsize):
    _, plt = _plt()
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    return fig, ax


def _blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ml_blue", BLUE_RAMP)


def _ink_on(rgba) -> str:
    """Dark or light text, whichever reads on a cell of this color."""
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return TEXT if lum > 0.55 else "#ffffff"


def nmi(a: Sequence, b: Sequence) -> float:
    """Normalized mutual information with arithmetic-mean normalization.

    Matches ``sklearn.metrics.normalized_mutual_info_score`` with its default
    ``average_method="arithmetic"``. Returns 1.0 when both labelings are a
    single class (the sklearn convention) and 0.0 when only one is.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D and the same length")
    if a.size == 0:
        raise ValueError("empty labelings")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    ka, kb = ai.max() + 1, bi.max() + 1
    if ka == 1 and kb == 1:
        return 1.0
    joint = np.zeros((ka, kb))
    np.add.at(joint, (ai, bi), 1)
    joint /= joint.sum()
    pa, pb = joint.sum(1), joint.sum(0)

    def h(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / np.outer(pa, pb)[nz])).sum())
    denom = 0.5 * (h(pa) + h(pb))
    if denom == 0:
        return 0.0
    return float(np.clip(mi / denom, 0.0, 1.0))


def area_transitions(
    assignments: pd.DataFrame,
    area_col: str = "macrostate",
    changes_only: bool = True,
) -> pd.DataFrame:
    """Count step-to-step moves between areas within each proof.

    ``assignments`` follows the schema (``proof_id``, ``step`` and the area
    column, e.g. ``macrostate`` or a judge's ``msc``). Returns columns
    ``from_area``, ``to_area``, ``count`` sorted by count, descending.
    """
    for col in ("proof_id", "step", area_col):
        if col not in assignments.columns:
            raise KeyError(f"assignments is missing column {col!r}")
    df = assignments.sort_values(["proof_id", "step"])
    nxt = df.groupby("proof_id", sort=False)[area_col].shift(-1)
    pairs = pd.DataFrame({"from_area": df[area_col].to_numpy(), "to_area": nxt.to_numpy()})
    pairs = pairs.dropna(subset=["to_area"])
    if changes_only:
        pairs = pairs[pairs["from_area"] != pairs["to_area"]]
    if pairs.empty:
        return pd.DataFrame({"from_area": [], "to_area": [], "count": []}).astype({"count": int})
    out = pairs.groupby(["from_area", "to_area"]).size().rename("count").reset_index()
    try:  # shift() turns int areas into floats; put the original dtype back
        out["to_area"] = out["to_area"].astype(df[area_col].dtype)
    except (TypeError, ValueError):
        pass
    return out.sort_values("count", ascending=False, kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- figures
def implied_timescales(
    lags: Sequence[float],
    timescales: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    *,
    lag_unit: str = "proof steps",
    ax=None,
    figsize=(3.4, 2.6),
):
    """Implied timescales against lag time.

    ``timescales`` has shape ``(n_lags, n_timescales)``; ``lower``/``upper``
    (same shape) draw a confidence band. The shaded region under the diagonal
    is where a timescale is faster than the lag and cannot be resolved.
    """
    lags = np.asarray(lags, dtype=float)
    ts = np.asarray(timescales, dtype=float)
    if ts.ndim == 1:
        ts = ts[:, None]
    if ts.shape[0] != lags.size:
        raise ValueError("timescales must have one row per lag")
    for name, arr in (("lower", lower), ("upper", upper)):
        if arr is not None and np.asarray(arr).reshape(ts.shape[0], -1).shape != ts.shape:
            raise ValueError(f"{name} must have the same shape as timescales")
    fig, ax = _new_ax(ax, figsize)
    for k in range(ts.shape[1]):
        c = color_for(k)
        ax.plot(lags, ts[:, k], color=c, lw=2, marker="o", ms=3.5, label=f"$t_{{{k + 2}}}$")
        if lower is not None and upper is not None:
            lo = np.asarray(lower, float).reshape(ts.shape)[:, k]
            hi = np.asarray(upper, float).reshape(ts.shape)[:, k]
            ax.fill_between(lags, lo, hi, color=c, alpha=0.18, lw=0)
    ax.fill_between(lags, np.full_like(lags, 1e-12), lags, color=GRID, lw=0, zorder=0)
    ax.plot(lags, lags, color=TEXT_SECONDARY, lw=1, ls="--", zorder=1)
    ax.set_yscale("log")
    positive = ts[np.isfinite(ts) & (ts > 0)]
    if positive.size:
        ax.set_ylim(min(lags.min(), positive.min()) * 0.7, positive.max() * 1.5)
    ax.set_xlabel(f"lag $\\tau$ ({lag_unit})")
    ax.set_ylabel(f"implied timescale ({lag_unit})")
    ax.grid(axis="y", color=GRID, lw=0.6)
    _style(ax)
    if ts.shape[1] > 1:
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    return fig


def chapman_kolmogorov(
    lags: Sequence[float],
    predicted: np.ndarray,
    estimated: np.ndarray,
    estimated_err: np.ndarray | None = None,
    *,
    labels: Sequence[str] | None = None,
    lag_unit: str = "proof steps",
    figsize: tuple[float, float] | None = None,
):
    """Chapman-Kolmogorov test as an ``m x m`` grid.

    ``predicted[j]`` is the model's ``T(tau)^k`` and ``estimated[j]`` is
    ``T(k tau)`` estimated directly, both ``(m, m)`` macrostate transition
    matrices at ``lags[j] = k tau``. Panel ``(i, l)`` shows ``P(i -> l)``.
    ``estimated_err`` (same shape) draws error bars on the estimate.
    """
    lags = np.asarray(lags, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    est = np.asarray(estimated, dtype=float)
    if pred.shape != est.shape or pred.ndim != 3 or pred.shape[1] != pred.shape[2]:
        raise ValueError("predicted and estimated must both be (n_lags, m, m)")
    if pred.shape[0] != lags.size:
        raise ValueError("one matrix per lag")
    m = pred.shape[1]
    labels = list(labels) if labels is not None else [str(i) for i in range(m)]
    if len(labels) != m:
        raise ValueError("labels must have one entry per macrostate")
    _, plt = _plt()
    figsize = figsize or (1.35 * m + 0.6, 1.2 * m + 0.6)
    fig, axes = plt.subplots(m, m, sharex=True, sharey=True, figsize=figsize, layout="constrained", squeeze=False)
    for i in range(m):
        for l in range(m):
            ax = axes[i, l]
            ax.plot(lags, pred[:, i, l], color=color_for(0), lw=2, label="model $T(\\tau)^k$")
            if estimated_err is not None:
                err = np.asarray(estimated_err, float)[:, i, l]
                ax.errorbar(lags, est[:, i, l], yerr=err, fmt="none", ecolor=color_for(1), elinewidth=1)
            ax.plot(lags, est[:, i, l], ls="none", marker="o", ms=4, color=color_for(1), label="estimate $T(k\\tau)$")
            ax.set_ylim(-0.05, 1.05)
            ax.text(0.95, 0.9, f"{labels[i]}$\\to${labels[l]}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color=TEXT)
            _style(ax)
    fig.supxlabel(f"lag $k\\tau$ ({lag_unit})", fontsize=9, color=TEXT)
    fig.supylabel("transition probability", fontsize=9, color=TEXT)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncol=2, frameon=False, fontsize=7)
    return fig


def transition_heatmap(
    T: np.ndarray,
    labels: Sequence[str] | None = None,
    *,
    annotate: bool = True,
    check_stochastic: bool = True,
    ax=None,
    figsize=(3.2, 2.8),
):
    """Macrostate transition matrix as a heatmap (rows: from, columns: to)."""
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be square")
    if check_stochastic and not np.allclose(T.sum(1), 1.0, atol=1e-6):
        raise ValueError("rows of T must sum to 1")
    m = T.shape[0]
    labels = list(labels) if labels is not None else [str(i) for i in range(m)]
    fig, ax = _new_ax(ax, figsize)
    cmap = _blue_cmap()
    im = ax.imshow(T, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(m), labels)
    ax.set_yticks(range(m), labels)
    ax.set_xlabel("to")
    ax.set_ylabel("from")
    if annotate:
        for i in range(m):
            for j in range(m):
                ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color=_ink_on(cmap(T[i, j])))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelsize=8, labelcolor=TEXT)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY, labelcolor=TEXT)
    return fig


def judge_cluster_confusion(
    clusters: Sequence | pd.Series,
    judge_labels: Sequence | pd.Series,
    *,
    normalize: str | None = "row",
    cluster_name: str = "macrostate",
    judge_name: str = "judge MSC",
    ax=None,
    figsize=(4.0, 3.0),
):
    """Contingency of macrostates (rows) against judge MSC codes (columns), NMI in the title.

    ``normalize`` is ``"row"`` (each macrostate sums to 1), ``"all"`` or
    ``None`` (raw counts). Pass the ``macrostate`` column of ``assignments``
    and the ``msc`` column of ``labels`` after merging on ``proof_id, step``.
    """
    c = pd.Series(np.asarray(clusters), name=cluster_name)
    j = pd.Series(np.asarray(judge_labels), name=judge_name)
    if len(c) != len(j):
        raise ValueError("clusters and judge_labels must be the same length")
    keep = c.notna() & j.notna()
    c, j = c[keep], j[keep]
    if c.empty:
        raise ValueError("no rows with both a cluster and a judge label")
    score = nmi(c.to_numpy(), j.astype(str).to_numpy())
    table = pd.crosstab(c, j)
    if normalize == "row":
        shown = table.div(table.sum(1), axis=0)
    elif normalize == "all":
        shown = table / table.to_numpy().sum()
    elif normalize is None:
        shown = table.astype(float)
    else:
        raise ValueError("normalize must be 'row', 'all' or None")
    fig, ax = _new_ax(ax, figsize)
    cmap = _blue_cmap()
    vmax = 1.0 if normalize else max(float(shown.to_numpy().max()), 1.0)
    im = ax.imshow(shown.to_numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(shown.shape[1]), [str(x) for x in shown.columns], rotation=90)
    ax.set_yticks(range(shown.shape[0]), [str(x) for x in shown.index])
    ax.set_xlabel(judge_name)
    ax.set_ylabel(cluster_name)
    ax.set_title(f"NMI = {score:.2f}", fontsize=9, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelsize=7, labelcolor=TEXT)
    ax.title.set_color(TEXT)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY, labelcolor=TEXT)
    fig.nmi = score  # handy for captions and tests
    return fig


def area_flow(
    transitions: pd.DataFrame,
    *,
    top: int | None = 12,
    area_names: dict | None = None,
    gap: float = 0.02,
    ax=None,
    figsize=(4.2, 3.4),
):
    """Sankey-style flow between areas: a stacked "from" bar, a stacked "to" bar, bands between.

    ``transitions`` has columns ``from_area``, ``to_area``, ``count`` (see
    :func:`area_transitions`). Only the ``top`` largest flows are drawn; the
    title says how many moves that covers. Band color follows the source area.
    """
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MPath

    for col in ("from_area", "to_area", "count"):
        if col not in transitions.columns:
            raise KeyError(f"transitions is missing column {col!r}")
    df = transitions[transitions["count"] > 0].copy()
    if df.empty:
        raise ValueError("no flows to draw")
    total = float(df["count"].sum())
    df = df.sort_values("count", ascending=False, kind="stable")
    if top is not None:
        df = df.head(top)
    shown = float(df["count"].sum())
    names = area_names or {}
    # a stable area order (and so a stable color) from the full table, not the cut
    order = sorted(set(transitions["from_area"]) | set(transitions["to_area"]), key=lambda x: str(x))
    color = {a: color_for(i) for i, a in enumerate(order)}

    def stack(col):
        sums = df.groupby(col)["count"].sum()
        sums = sums.reindex([a for a in order if a in sums.index])
        n = len(sums)
        scale = (1 - gap * max(n - 1, 0)) / shown
        pos, y = {}, 1.0
        for a, v in sums.items():
            h = v * scale
            pos[a] = [y - h, y, y]  # bottom, top, cursor (bands fill top-down)
            y -= h + gap
        return pos, scale

    left, scale = stack("from_area")
    right, _ = stack("to_area")
    fig, ax = _new_ax(ax, figsize)
    xl, xr, w = 0.0, 1.0, 0.06
    for a, (b, t, _) in left.items():
        ax.add_patch(Rectangle((xl - w, b), w, t - b, color=color[a], lw=0))
        ax.text(xl - w - 0.02, (b + t) / 2, str(names.get(a, a)), ha="right", va="center", fontsize=7, color=TEXT)
    for a, (b, t, _) in right.items():
        ax.add_patch(Rectangle((xr, b), w, t - b, color=color[a], lw=0))
        ax.text(xr + w + 0.02, (b + t) / 2, str(names.get(a, a)), ha="left", va="center", fontsize=7, color=TEXT)
    for _, row in df.iterrows():
        h = row["count"] * scale
        lt = left[row["from_area"]][2]
        rt = right[row["to_area"]][2]
        left[row["from_area"]][2] -= h
        right[row["to_area"]][2] -= h
        xm = (xl + xr) / 2
        verts = [
            (xl, lt), (xm, lt), (xm, rt), (xr, rt),
            (xr, rt - h), (xm, rt - h), (xm, lt - h), (xl, lt - h), (xl, lt),
        ]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color[row["from_area"]], alpha=0.35, lw=0))
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")
    ax.text(xl - w / 2, 1.05, "from", ha="center", fontsize=8, color=TEXT_SECONDARY)
    ax.text(xr + w / 2, 1.05, "to", ha="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title(f"top {len(df)} flows, {shown / total:.0%} of {int(total)} moves", fontsize=8,
                 color=TEXT_SECONDARY, loc="left")
    return fig


def search_ablation(
    results: pd.DataFrame,
    *,
    method_col: str = "method",
    metric_col: str = "solved",
    group_col: str | None = None,
    err_col: str | None = None,
    metric_label: str = "problems solved (%)",
    ax=None,
    figsize=(3.6, 2.6),
):
    """Bars for a search ablation: one bar per method, grouped by benchmark if ``group_col`` is set.

    Method color follows the method's first appearance in ``results`` so the
    same method keeps its color across benchmarks and figures built from the
    same table.
    """
    cols = [method_col, metric_col] + [c for c in (group_col, err_col) if c]
    for col in cols:
        if col not in results.columns:
            raise KeyError(f"results is missing column {col!r}")
    methods = list(dict.fromkeys(results[method_col]))
    groups = list(dict.fromkeys(results[group_col])) if group_col else [None]
    fig, ax = _new_ax(ax, figsize)
    n = len(methods)
    width = 0.8 / n
    for mi, meth in enumerate(methods):
        xs, ys, es = [], [], []
        for gi, g in enumerate(groups):
            sel = results[method_col] == meth
            if group_col:
                sel &= results[group_col] == g
            rows = results[sel]
            if rows.empty:
                continue
            if len(rows) > 1:
                raise ValueError(f"more than one row for method {meth!r} in group {g!r}")
            xs.append(gi - 0.4 + width * (mi + 0.5))
            ys.append(float(rows[metric_col].iloc[0]))
            es.append(float(rows[err_col].iloc[0]) if err_col else 0.0)
        ax.bar(xs, ys, width=width * 0.92, color=color_for(mi), label=str(meth),
               yerr=es if err_col else None, error_kw={"elinewidth": 1, "ecolor": TEXT_SECONDARY, "capsize": 2})
    if group_col:
        ax.set_xticks(range(len(groups)), [str(g) for g in groups])
    else:
        ax.set_xticks([])
    ax.set_ylabel(metric_label)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _style(ax)
    if n > 1:
        ax.legend(frameon=False, fontsize=7, ncol=min(n, 3), loc="upper left")
    return fig


def save_pdf(fig, path: str | Path, **kwargs) -> Path:
    """Save ``fig`` as a PDF (creating parent folders) and return the path."""
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, format="pdf", **kwargs)
    return path


def save_all(figures: Iterable[tuple[str, object]], outdir: str | Path) -> list[Path]:
    """Save ``(name, fig)`` pairs as ``outdir/name.pdf``."""
    return [save_pdf(fig, Path(outdir) / name) for name, fig in figures]
