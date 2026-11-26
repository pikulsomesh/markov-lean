"""Paper figures over the shared tables in ``docs/schema.md``.

Every public plotting function takes arrays or DataFrames, draws on a new
Figure (or on an ``ax`` you pass), and returns the Figure. Nothing here reads
files or runs the pipeline; call :func:`save_pdf` to write a figure.

Style: seaborn's ``ticks`` look at ``paper`` scale with a sans-serif font
(Arial, falling back to Helvetica, Liberation Sans and DejaVu Sans where Arial
is not installed). Every figure is drawn inside :func:`paper_rc`, so importing
this module changes no global matplotlib settings; call :func:`use_paper_style`
to apply the same look to your own plots. PDFs embed TrueType fonts, never
Type 3, which the ICML instructions warn against. Every figure is drawn at the
size it prints, the ICML column (3.25 in) or the full text width (6.75 in), and
the paper includes it at that width, so its 8 pt text (7 pt tick labels) prints
at 8 pt; :func:`save_pdf` keeps the drawn size instead of cropping. The ICML
instructions put captions under figures and no titles inside the graphic, so
no figure draws a title by default; figures that compute a headline number
(NMI, the share of moves a flow plot covers) attach it to the Figure for the
caption instead.

Color: the validated default categorical order (light mode, since the paper
prints on white) and a one-hue blue ramp for magnitudes. Colors follow the
entity: macrostate ``i`` always gets slot ``i`` so a figure that drops a state
does not repaint the rest.

matplotlib and seaborn are optional dependencies (the ``figures`` extra), so
they are imported lazily.
"""
from __future__ import annotations

import functools
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
# Arial first; the rest are metric-compatible or close stand-ins for machines without it.
SANS_SERIF = ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans")
# ICML 2027 two-column layout, inches.
COLUMN_WIDTH = 3.25
TEXT_WIDTH = 6.75
# Printed text sizes in points: labels, titles and legend titles at BASE_PT,
# tick labels, legends and annotations at SMALL_PT (the floor for printed text).
BASE_PT = 8.0
SMALL_PT = 7.0

__all__ = [
    "CATEGORICAL",
    "COLUMN_WIDTH",
    "TEXT_WIDTH",
    "SANS_SERIF",
    "color_for",
    "paper_rc",
    "use_paper_style",
    "nmi",
    "area_transitions",
    "implied_timescales",
    "chapman_kolmogorov",
    "chapman_kolmogorov_diagonal",
    "transition_heatmap",
    "judge_cluster_confusion",
    "area_flow",
    "search_ablation",
    "save_pdf",
    "save_all",
]


# ---------------------------------------------------------------- style
def _plt():
    import matplotlib

    import matplotlib.pyplot as plt

    return matplotlib, plt


def _sns():
    import seaborn as sns

    return sns


def color_for(i: int) -> str:
    """Categorical color for entity ``i``; past the eighth slot everything is "Other" gray."""
    return CATEGORICAL[i] if 0 <= i < len(CATEGORICAL) else OTHER


def paper_rc(font_scale: float = 1.0) -> dict:
    """matplotlib rc settings for the paper: seaborn ``ticks`` + ``paper`` context, sans-serif.

    Text sizes are set in points, not scaled from seaborn's defaults: axis
    labels and titles 8 pt, tick labels and legends 7 pt, a little under the
    9 pt ICML caption size. They are the printed sizes as long as the figure is
    included at the width it was drawn. ``font_scale`` multiplies them all.
    """
    from cycler import cycler

    sns = _sns()
    rc: dict = {}
    rc.update(sns.axes_style("ticks"))
    rc.update(sns.plotting_context("paper"))
    base, small = BASE_PT * font_scale, SMALL_PT * font_scale
    rc.update({
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "figure.titlesize": base,
        "figure.labelsize": base,
        "legend.title_fontsize": base,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small,
    })
    rc.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(SANS_SERIF),
        # math in the same sans-serif as the text; symbols the font lacks
        # (arrows, Greek in some fonts) fall back to STIX Sans
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.fallback": "stixsans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": cycler(color=list(CATEGORICAL)),
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": TEXT,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelcolor": TEXT,
        "ytick.labelcolor": TEXT,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 300,
    })
    return rc


def use_paper_style(font_scale: float = 1.0) -> None:
    """Apply :func:`paper_rc` globally (for notebooks and one-off plots)."""
    _plt()[0].rcParams.update(paper_rc(font_scale))


def _themed(fn):
    """Draw ``fn``'s figure inside the paper style without touching global rcParams."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        matplotlib, _ = _plt()
        with matplotlib.rc_context(paper_rc()):
            return fn(*args, **kwargs)

    return wrapper


def _new_ax(ax, figsize):
    _, plt = _plt()
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    return fig, ax


def _area_label(a, area_names: dict | None = None) -> str:
    """``area_names[a]`` if given, else "28 Measure, integration" for an MSC code, else ``str(a)``."""
    if area_names and a in area_names:
        return str(area_names[a])
    from .msc import short_title

    return short_title(a)


def _natural_key(x) -> list:
    import re

    return [(0, int(t), "") if t.isdigit() else (1, 0, t) for t in re.split(r"(\d+)", str(x)) if t]


def _blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ml_blue", BLUE_RAMP)


def _style_colorbar(ax) -> None:
    if ax.collections and getattr(ax.collections[0], "colorbar", None) is not None:
        cb = ax.collections[0].colorbar
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0)


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
@_themed
def implied_timescales(
    lags: Sequence[float],
    timescales: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    *,
    lag_unit: str = "proof steps",
    ax=None,
    figsize=(COLUMN_WIDTH, 2.2),
):
    """Implied timescales against lag time.

    ``timescales`` has shape ``(n_lags, n_timescales)``; ``lower``/``upper``
    (same shape) draw a confidence band. The shaded region under the diagonal
    is where a timescale is faster than the lag and cannot be resolved.
    """
    sns = _sns()
    lags = np.asarray(lags, dtype=float)
    ts = np.asarray(timescales, dtype=float)
    if ts.ndim == 1:
        ts = ts[:, None]
    if ts.shape[0] != lags.size:
        raise ValueError("timescales must have one row per lag")
    for name, arr in (("lower", lower), ("upper", upper)):
        if arr is not None and np.asarray(arr).reshape(ts.shape[0], -1).shape != ts.shape:
            raise ValueError(f"{name} must have the same shape as timescales")
    n = ts.shape[1]
    names = [f"$t_{{{k + 2}}}$" for k in range(n)]
    long = pd.DataFrame({
        "lag": np.repeat(lags, n),
        "timescale": ts.ravel(),
        "process": np.tile(names, lags.size),
    })
    fig, ax = _new_ax(ax, figsize)
    ax.fill_between(lags, np.full_like(lags, 1e-12), lags, color=GRID, lw=0, zorder=0)
    ax.plot(lags, lags, color=TEXT_SECONDARY, lw=1, ls="--", zorder=1)
    if lower is not None and upper is not None:
        lo = np.asarray(lower, float).reshape(ts.shape)
        hi = np.asarray(upper, float).reshape(ts.shape)
        for k in range(n):
            ax.fill_between(lags, lo[:, k], hi[:, k], color=color_for(k), alpha=0.18, lw=0)
    sns.lineplot(
        data=long, x="lag", y="timescale", hue="process", hue_order=names,
        palette={nm: color_for(k) for k, nm in enumerate(names)},
        marker="o", markersize=3.5, markeredgewidth=0, linewidth=1.5, ax=ax, legend=n > 1,
    )
    ax.set_yscale("log")
    positive = ts[np.isfinite(ts) & (ts > 0)]
    if positive.size:
        ax.set_ylim(min(lags.min(), positive.min()) * 0.7, positive.max() * 1.5)
    ax.set_xlabel(f"lag $\\tau$ ({lag_unit})")
    ax.set_ylabel(f"implied timescale ({lag_unit})")
    ax.grid(axis="y")
    if n > 1:  # above the plot, where it cannot sit on a line; at most 6 per row so 7 pt fits the column
        ax.legend(title=None, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=min(n, 6),
                  borderaxespad=0.2, handlelength=1.2, handletextpad=0.4, columnspacing=0.9,
                  fontsize=SMALL_PT)
    return fig


def _ck_arrays(lags, predicted, estimated, estimated_err, labels):
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
    err = None if estimated_err is None else np.asarray(estimated_err, float).reshape(pred.shape)
    return lags, pred, est, err, m, labels


def _ck_panel(ax, lags, pred, est, err, i, l) -> None:
    ax.plot(lags, pred[:, i, l], color=color_for(0), lw=1.2, label="model $T(\\tau)^k$")
    if err is not None:
        ax.errorbar(lags, est[:, i, l], yerr=err[:, i, l], fmt="none", ecolor=color_for(1), elinewidth=0.8)
    ax.plot(lags, est[:, i, l], ls="none", marker="o", ms=2.5, mew=0, color=color_for(1),
            label="estimate $T(k\\tau)$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "", "1"])
    ax.tick_params(length=2, pad=1.5)


@_themed
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
    """Chapman-Kolmogorov test as an ``m x m`` grid (the appendix figure).

    ``predicted[j]`` is the model's ``T(tau)^k`` and ``estimated[j]`` is
    ``T(k tau)`` estimated directly, both ``(m, m)`` macrostate transition
    matrices at ``lags[j] = k tau``. Panel ``(i, l)`` shows the transition probability ``P(i -> l)``;
    rows are labelled "from" down the left and columns "to" along the top, so
    the panels carry no text of their own. ``estimated_err`` (same shape)
    draws error bars on the estimate. The default size is the full text width,
    for a ``figure*``. :func:`chapman_kolmogorov_diagonal` draws the main-text
    version at the column width.
    """
    lags, pred, est, err, m, labels = _ck_arrays(lags, predicted, estimated, estimated_err, labels)
    _, plt = _plt()
    if figsize is None:  # full text width (the appendix figure*), panels about 0.6 in wide
        figsize = (TEXT_WIDTH, min(8.5, 0.9 + 0.55 * m))
    fig, axes = plt.subplots(m, m, sharex=True, sharey=True, figsize=figsize, layout="constrained", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)
    for i in range(m):
        for l in range(m):
            ax = axes[i, l]
            _ck_panel(ax, lags, pred, est, err, i, l)
            if i == 0:
                ax.set_title(f"to {labels[l]}", fontsize=SMALL_PT, pad=2)
            if l == 0:
                ax.set_ylabel(f"from {labels[i]}", fontsize=SMALL_PT, labelpad=2)
    # no figure-wide y label: it would sit on the "from" row labels; the caption says what is plotted
    fig.supxlabel(f"lag $k\\tau$ ({lag_unit})")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncol=2)
    return fig


@_themed
def chapman_kolmogorov_diagonal(
    lags: Sequence[float],
    predicted: np.ndarray,
    estimated: np.ndarray,
    estimated_err: np.ndarray | None = None,
    *,
    labels: Sequence[str] | None = None,
    lag_unit: str = "proof steps",
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Chapman-Kolmogorov test, stay-in-place panels only: ``P(i -> i)`` for each macrostate.

    The same inputs as :func:`chapman_kolmogorov`, drawn as ``m`` panels in
    rows of ``ncols`` (default 5) at the column width, for the main text.
    """
    lags, pred, est, err, m, labels = _ck_arrays(lags, predicted, estimated, estimated_err, labels)
    _, plt = _plt()
    ncols = ncols or min(m, 5)
    nrows = -(-m // ncols)
    if figsize is None:
        figsize = (COLUMN_WIDTH, 0.62 + 0.78 * nrows)
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=figsize, layout="constrained",
                             squeeze=False)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)
    for k, ax in enumerate(axes.flat):
        if k >= m:
            ax.set_visible(False)
            continue
        _ck_panel(ax, lags, pred, est, err, k, k)
        ax.set_title(f"{labels[k]}$\\to${labels[k]}", fontsize=SMALL_PT, pad=2)
    fig.supxlabel(f"lag $k\\tau$ ({lag_unit})")
    fig.supylabel("P(stay)")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncol=2)
    return fig


@_themed
def transition_heatmap(
    T: np.ndarray,
    labels: Sequence[str] | None = None,
    *,
    annotate: bool = False,
    check_stochastic: bool = True,
    ax=None,
    figsize=(COLUMN_WIDTH, 2.6),
):
    """Macrostate transition matrix as a heatmap (rows: from, columns: to).

    Cells carry no numbers by default (the colour scale reads them); pass
    ``annotate=True`` for a small matrix on a slide.
    """
    sns = _sns()
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be square")
    if check_stochastic and not np.allclose(T.sum(1), 1.0, atol=1e-6):
        raise ValueError("rows of T must sum to 1")
    m = T.shape[0]
    labels = list(labels) if labels is not None else [str(i) for i in range(m)]
    if len(labels) != m:
        raise ValueError("labels must have one entry per macrostate")
    frame = pd.DataFrame(T, index=pd.Index(labels, name="from"), columns=pd.Index(labels, name="to"))
    fig, ax = _new_ax(ax, figsize)
    sns.heatmap(
        frame, ax=ax, cmap=_blue_cmap(), vmin=0, vmax=1, square=True,
        annot=annotate, fmt=".2f", annot_kws={"fontsize": SMALL_PT},
        linewidths=1, linecolor="white", cbar_kws={"shrink": 0.85},
    )
    ax.tick_params(length=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _style_colorbar(ax)
    return fig


@_themed
def judge_cluster_confusion(
    clusters: Sequence | pd.Series,
    judge_labels: Sequence | pd.Series,
    *,
    normalize: str | None = "row",
    annotate: bool | None = None,
    show_nmi: bool = False,
    cluster_name: str = "macrostate",
    judge_name: str = "judge MSC",
    area_names: dict | None = None,
    ax=None,
    figsize=(COLUMN_WIDTH, 3.0),
):
    """Contingency of macrostates (rows) against judge MSC codes (columns).

    The NMI between the two labelings is attached as ``fig.nmi`` for the
    caption; ``show_nmi=True`` also writes it above the plot (for slides and
    drafts, since ICML wants no titles inside figures).

    ``normalize`` is ``"row"`` (each macrostate sums to 1), ``"all"`` or
    ``None`` (raw counts). Pass the ``macrostate`` column of ``assignments``
    and the ``msc`` column of ``labels`` after merging on ``proof_id, step``.
    ``annotate`` defaults to no numbers in the cells (the colour scale reads
    them). MSC codes on the columns are labelled with their short names
    (:func:`markov_lean.msc.short_title`); ``area_names`` (code -> label)
    overrides that, and labels that are not MSC codes are left as they are.
    """
    sns = _sns()
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
    table = table.loc[sorted(table.index, key=_natural_key)]  # M1, M2, ..., M10, not M1, M10, M2
    if normalize == "row":
        shown = table.div(table.sum(axis=1), axis=0)
    elif normalize == "all":
        shown = table / table.to_numpy().sum()
    elif normalize is None:
        shown = table.astype(float)
    else:
        raise ValueError("normalize must be 'row', 'all' or None")
    if annotate is None:
        annotate = False
    shown.index = shown.index.map(str)
    shown.columns = shown.columns.map(lambda a: _area_label(a, area_names))
    vmax = 1.0 if normalize else max(float(shown.to_numpy().max()), 1.0)
    fig, ax = _new_ax(ax, figsize)
    sns.heatmap(
        shown, ax=ax, cmap=_blue_cmap(), vmin=0, vmax=vmax,
        annot=annotate, fmt=".2f" if normalize else ".0f", annot_kws={"fontsize": SMALL_PT},
        linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.85}, xticklabels=1, yticklabels=1,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel(judge_name)
    ax.set_ylabel(cluster_name)
    if show_nmi:
        ax.set_title(f"NMI = {score:.2f}", loc="left")
    ax.tick_params(length=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _style_colorbar(ax)
    fig.nmi = score  # for the caption
    return fig


@_themed
def area_flow(
    transitions: pd.DataFrame,
    *,
    top: int | None = 12,
    area_names: dict | None = None,
    gap: float = 0.02,
    show_summary: bool = False,
    ax=None,
    figsize=(COLUMN_WIDTH, 2.8),
):
    """Sankey-style flow between areas: a stacked "from" bar, a stacked "to" bar, bands between.

    ``transitions`` has columns ``from_area``, ``to_area``, ``count`` (see
    :func:`area_transitions`). Only the ``top`` largest flows are drawn. For
    the caption, ``fig.flows_shown``, ``fig.moves_shown`` and ``fig.moves_total``
    say how much of the table that covers; ``show_summary=True`` also writes it
    above the plot. Band color follows the source area. Areas that are MSC
    codes are labelled "code short name" (:func:`markov_lean.msc.short_title`);
    ``area_names`` (area -> label) overrides that, and other labels such as
    macrostates are left as they are.
    seaborn has no flow plot, so this one is drawn with matplotlib patches in
    the same style.
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
    xl, xr, w = 0.0, 1.0, 0.05
    for a, (b, t, _) in left.items():
        ax.add_patch(Rectangle((xl - w, b), w, t - b, color=color[a], lw=0))
        ax.text(xl - w - 0.02, (b + t) / 2, _area_label(a, area_names), ha="right", va="center", fontsize=SMALL_PT)
    for a, (b, t, _) in right.items():
        ax.add_patch(Rectangle((xr, b), w, t - b, color=color[a], lw=0))
        ax.text(xr + w + 0.02, (b + t) / 2, _area_label(a, area_names), ha="left", va="center", fontsize=SMALL_PT)
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
    ax.set_xlim(-0.62, 1.62)  # room for "28 Measure, integration" at 7 pt on both sides
    ax.set_ylim(-0.08, 1.02)
    ax.axis("off")
    ax.text(xl - w / 2, -0.03, "from", ha="center", va="top", fontsize=SMALL_PT, color=TEXT_SECONDARY)
    ax.text(xr + w / 2, -0.03, "to", ha="center", va="top", fontsize=SMALL_PT, color=TEXT_SECONDARY)
    if show_summary:
        ax.set_title(f"top {len(df)} flows, {shown / total:.0%} of {int(total)} moves", loc="left",
                     fontsize=SMALL_PT, color=TEXT_SECONDARY)
    fig.flows_shown, fig.moves_shown, fig.moves_total = len(df), int(shown), int(total)
    return fig


@_themed
def search_ablation(
    results: pd.DataFrame,
    *,
    method_col: str = "method",
    metric_col: str = "solved",
    group_col: str | None = None,
    err_col: str | None = None,
    errorbar=("ci", 95),
    metric_label: str = "problems solved (%)",
    ax=None,
    figsize=(COLUMN_WIDTH, 2.4),
):
    """Bars for a search ablation: one bar per method, grouped by benchmark if ``group_col`` is set.

    ``results`` is long-form: one row per method (and benchmark), or one row
    per run when there are several seeds, in which case seaborn draws the mean
    with ``errorbar`` (default a 95% bootstrap CI). Pass ``err_col`` instead to
    draw errors you computed yourself; that needs exactly one row per bar.
    Method color follows the method's first appearance in ``results`` so the
    same method keeps its color across benchmarks and figures.
    """
    sns = _sns()
    cols = [method_col, metric_col] + [c for c in (group_col, err_col) if c]
    for col in cols:
        if col not in results.columns:
            raise KeyError(f"results is missing column {col!r}")
    methods = list(dict.fromkeys(results[method_col]))
    data = results.copy()
    gcol = group_col or "__group__"
    if group_col is None:
        data[gcol] = ""
    groups = list(dict.fromkeys(data[gcol]))
    if err_col and data.duplicated([method_col, gcol]).any():
        raise ValueError("err_col needs exactly one row per method (and group)")
    fig, ax = _new_ax(ax, figsize)
    sns.barplot(
        data=data, x=gcol, y=metric_col, hue=method_col, order=groups, hue_order=methods,
        palette={m: color_for(i) for i, m in enumerate(methods)},
        errorbar=None if err_col else errorbar, err_kws={"linewidth": 1, "color": TEXT_SECONDARY},
        capsize=0.1, saturation=1, width=0.8, ax=ax,
    )
    if err_col:
        from matplotlib.container import BarContainer

        lookup = data.set_index([method_col, gcol])[err_col]
        bar_containers = [c for c in ax.containers if isinstance(c, BarContainer)]
        for meth, container in zip(methods, bar_containers):
            for bar in container:
                if not np.isfinite(bar.get_height()):
                    continue
                xc = bar.get_x() + bar.get_width() / 2
                g = groups[int(round(xc))]
                if (meth, g) not in lookup.index:
                    continue
                ax.errorbar(xc, bar.get_height(), yerr=float(lookup[(meth, g)]), fmt="none",
                            ecolor=TEXT_SECONDARY, elinewidth=1, capsize=2)
    ax.set_xlabel("")
    ax.set_ylabel(metric_label)
    if group_col is None:
        ax.set_xticks([])
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    legend = ax.get_legend()
    if len(methods) > 1:  # above the bars, where it cannot cover one
        sns.move_legend(ax, "lower left", bbox_to_anchor=(0, 1.0), title=None,
                        ncol=min(len(methods), 3), borderaxespad=0.2, handlelength=1.0, columnspacing=1.0)
    elif legend is not None:
        legend.remove()
    return fig


def save_pdf(fig, path: str | Path, **kwargs) -> Path:
    """Save ``fig`` as a PDF (creating parent folders) and return the path."""
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    # No tight cropping: the figure is included at the width it was drawn, so
    # cropping would rescale it and its text. constrained layout already fits
    # every label inside the figure.
    matplotlib, _ = _plt()
    # Font families resolve when the figure is drawn, so save under the paper's
    # font settings too; otherwise text falls back to the global default font.
    rc = {k: v for k, v in paper_rc().items() if k.startswith(("font.family", "font.sans-serif", "mathtext."))}
    rc["pdf.fonttype"] = 42  # embed TrueType, never Type 3
    with matplotlib.rc_context(rc):
        fig.savefig(path, format="pdf", **kwargs)
    return path


def save_all(figures: Iterable[tuple[str, object]], outdir: str | Path) -> list[Path]:
    """Save ``(name, fig)`` pairs as ``outdir/name.pdf``."""
    return [save_pdf(fig, Path(outdir) / name) for name, fig in figures]
