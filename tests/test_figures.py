"""Tests for the paper figures. Synthetic arrays only, Agg backend, no files outside tmp_path."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markov_lean import figures


# ------------------------------------------------ pure helpers (no matplotlib)
def test_nmi_identical_and_relabelled():
    a = [0, 0, 1, 1, 2, 2]
    assert figures.nmi(a, a) == pytest.approx(1.0)
    assert figures.nmi(a, ["x", "x", "y", "y", "z", "z"]) == pytest.approx(1.0)


def test_nmi_independent_is_zero():
    a = [0, 0, 1, 1]
    b = [0, 1, 0, 1]
    assert figures.nmi(a, b) == pytest.approx(0.0, abs=1e-12)


def test_nmi_hand_computed():
    # a = [0,0,1,1], b = [0,0,0,1]: I = H(b) - H(b|a) = h(1/4) - 0.5*ln2
    a, b = [0, 0, 1, 1], [0, 0, 0, 1]
    hb = -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
    mi = hb - 0.5 * np.log(2)
    expected = mi / (0.5 * (np.log(2) + hb))
    assert figures.nmi(a, b) == pytest.approx(expected)


def test_nmi_matches_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, 300)
    b = (a + rng.integers(0, 3, 300)) % 7
    assert figures.nmi(a, b) == pytest.approx(metrics.normalized_mutual_info_score(a, b))


def test_nmi_rejects_bad_input():
    with pytest.raises(ValueError):
        figures.nmi([0, 1], [0])
    with pytest.raises(ValueError):
        figures.nmi([], [])


def test_area_transitions_counts_moves_within_proofs():
    a = pd.DataFrame({
        "proof_id": ["p1"] * 4 + ["p2"] * 3,
        "step": [0, 1, 2, 3, 0, 1, 2],
        "macrostate": [0, 0, 1, 2, 2, 1, 1],
    })
    out = figures.area_transitions(a)
    got = {(r.from_area, r.to_area): r.count for r in out.itertuples()}
    assert got == {(0, 1): 1, (1, 2): 1, (2, 1): 1}
    # p1 ends in 2 and p2 starts in 2: that is not a move
    with_stays = figures.area_transitions(a, changes_only=False)
    assert int(with_stays["count"].sum()) == 5


def test_area_transitions_sorts_steps_and_checks_columns():
    a = pd.DataFrame({"proof_id": ["p"] * 3, "step": [2, 0, 1], "msc": ["26", "11", "11"]})
    out = figures.area_transitions(a, area_col="msc")
    assert list(zip(out.from_area, out.to_area)) == [("11", "26")]
    with pytest.raises(KeyError):
        figures.area_transitions(a)  # no macrostate column


def test_color_for_folds_past_eight():
    assert figures.color_for(0) == figures.CATEGORICAL[0]
    assert figures.color_for(8) == figures.OTHER


# ------------------------------------------------ figures (need matplotlib)
@pytest.fixture
def plt():
    mpl = pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def _bars(ax):
    from matplotlib.container import BarContainer

    return [b for c in ax.containers if isinstance(c, BarContainer) for b in c if np.isfinite(b.get_height())]


def _msm(m=3, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.random((m, m)) + 4 * np.eye(m)
    return T / T.sum(1, keepdims=True)


def test_implied_timescales(plt, tmp_path):
    lags = np.array([1, 2, 3, 5, 8])
    ts = np.column_stack([10 + lags, 4 + 0.5 * lags])
    fig = figures.implied_timescales(lags, ts, lower=ts * 0.9, upper=ts * 1.1)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    drawn = [ln for ln in ax.get_lines() if len(ln.get_xdata())]  # seaborn adds empty legend proxies
    assert len(drawn) == 3  # two timescales + the tau = t line
    path = figures.save_pdf(fig, tmp_path / "its.pdf")
    assert path.read_bytes()[:4] == b"%PDF"


def test_implied_timescales_shape_check(plt):
    with pytest.raises(ValueError):
        figures.implied_timescales([1, 2, 3], np.ones((2, 2)))


def test_chapman_kolmogorov(plt, tmp_path):
    T = _msm(3)
    ks = np.arange(1, 6)
    pred = np.stack([np.linalg.matrix_power(T, k) for k in ks])
    est = np.clip(pred + 0.01, 0, 1)
    fig = figures.chapman_kolmogorov(ks, pred, est, estimated_err=np.full_like(est, 0.02),
                                     labels=["A", "B", "C"])
    assert len(fig.axes) == 9
    figures.save_pdf(fig, tmp_path / "ck")
    assert (tmp_path / "ck.pdf").exists()


def test_chapman_kolmogorov_rejects_mismatch(plt):
    with pytest.raises(ValueError):
        figures.chapman_kolmogorov([1, 2], np.zeros((2, 2, 2)), np.zeros((2, 3, 3)))


def test_transition_heatmap(plt, tmp_path):
    T = _msm(4)
    fig = figures.transition_heatmap(T, labels=list("abcd"))
    assert len(fig.axes[0].texts) == 0  # no numbers by default: the colour scale reads them
    annotated = figures.transition_heatmap(T, labels=list("abcd"), annotate=True)
    texts = [t.get_text() for t in annotated.axes[0].texts]
    assert len(texts) == 16 and f"{T[0, 0]:.2f}" in texts
    figures.save_pdf(fig, tmp_path / "T.pdf")


def test_chapman_kolmogorov_diagonal_shows_stay_panels(plt, tmp_path):
    T = _msm(7)
    ks = np.arange(1, 5)
    pred = np.stack([np.linalg.matrix_power(T, k) for k in ks])
    fig = figures.chapman_kolmogorov_diagonal(ks, pred, pred, labels=[f"M{i + 1}" for i in range(7)])
    shown = [ax for ax in fig.axes if ax.get_visible()]
    assert len(shown) == 7  # 2 rows of 5, the last three hidden
    assert shown[6].get_title() == "M7$\\to$M7"
    assert np.allclose(shown[2].get_lines()[0].get_ydata(), pred[:, 2, 2])
    path = figures.save_pdf(fig, tmp_path / "ckd.pdf")
    assert fig.get_size_inches()[0] == pytest.approx(figures.COLUMN_WIDTH)
    assert b"/MediaBox [ 0 0 234" in path.read_bytes()  # 3.25 in: saved at the drawn size, not cropped


def test_chapman_kolmogorov_full_grid_is_full_width(plt):
    T = _msm(10)
    pred = np.stack([np.linalg.matrix_power(T, k) for k in (1, 2, 3)])
    fig = figures.chapman_kolmogorov([1, 2, 3], pred, pred)
    assert fig.get_size_inches()[0] == pytest.approx(figures.TEXT_WIDTH)
    assert len(fig.axes) == 100


def test_text_is_8_pt_with_7_pt_ticks():
    rc = figures.paper_rc()
    assert rc["axes.labelsize"] == 8 and rc["font.size"] == 8
    assert rc["xtick.labelsize"] == 7 and rc["ytick.labelsize"] == 7 and rc["legend.fontsize"] == 7


def test_transition_heatmap_requires_stochastic(plt):
    with pytest.raises(ValueError):
        figures.transition_heatmap(np.ones((2, 2)))


def test_judge_cluster_confusion_reports_nmi(plt, tmp_path):
    rng = np.random.default_rng(1)
    clusters = rng.integers(0, 4, 400)
    msc = np.array(["11", "26", "05", "15"])[clusters]
    msc[:40] = "03"  # some noise
    fig = figures.judge_cluster_confusion(clusters, msc)
    assert fig.nmi == pytest.approx(figures.nmi(clusters, msc))
    assert fig.axes[0].get_title(loc="left") == ""  # ICML: no titles inside figures
    figures.save_pdf(fig, tmp_path / "conf.pdf")
    shown = figures.judge_cluster_confusion(clusters, msc, show_nmi=True)
    assert f"NMI = {fig.nmi:.2f}" in shown.axes[0].get_title(loc="left")


def test_judge_cluster_confusion_drops_missing_and_validates(plt):
    fig = figures.judge_cluster_confusion([0, 0, 1, 1], ["11", None, "26", "26"], normalize=None)
    assert fig.nmi == pytest.approx(1.0)
    with pytest.raises(ValueError):
        figures.judge_cluster_confusion([0, 1], ["11"])
    with pytest.raises(ValueError):
        figures.judge_cluster_confusion([0, 1], ["11", "26"], normalize="col")


def test_area_flow(plt, tmp_path):
    rng = np.random.default_rng(2)
    a = pd.DataFrame({
        "proof_id": np.repeat([f"p{i}" for i in range(30)], 6),
        "step": np.tile(range(6), 30),
        "macrostate": rng.integers(0, 4, 180),
    })
    flows = figures.area_transitions(a)
    names = {0: "algebra", 1: "analysis", 2: "number theory", 3: "combinatorics"}
    fig = figures.area_flow(flows, top=5, area_names=names)
    ax = fig.axes[0]
    assert ax.get_title(loc="left") == ""
    assert fig.flows_shown == 5
    assert fig.moves_shown == int(flows.head(5)["count"].sum())
    assert fig.moves_total == int(flows["count"].sum())
    titled = figures.area_flow(flows, top=5, area_names=names, show_summary=True)
    assert titled.axes[0].get_title(loc="left").startswith("top 5 flows")
    labels = {t.get_text() for t in ax.texts}
    shown = set(flows.head(5).from_area) | set(flows.head(5).to_area)
    assert {names[a] for a in shown} <= labels
    figures.save_pdf(fig, tmp_path / "flow.pdf")


def test_msc_codes_get_short_names(plt):
    flows = pd.DataFrame({"from_area": ["28", "26"], "to_area": ["26", "54"], "count": [5, 3]})
    labels = {t.get_text() for t in figures.area_flow(flows).axes[0].texts}
    assert {"28 Measure, integration", "26 Real functions", "54 General topology"} <= labels
    fig = figures.judge_cluster_confusion(["M2", "M10", "M1", "M2"], ["28", "26", "11", "28"])
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["M1", "M2", "M10"]
    assert "11 Number theory" in [t.get_text() for t in ax.get_xticklabels()]


def test_area_flow_rejects_empty(plt):
    with pytest.raises(ValueError):
        figures.area_flow(pd.DataFrame({"from_area": [0], "to_area": [1], "count": [0]}))


def test_search_ablation(plt, tmp_path):
    res = pd.DataFrame({
        "benchmark": ["miniF2F", "miniF2F", "ProofNet", "ProofNet"],
        "method": ["baseline", "area prior", "baseline", "area prior"],
        "solved": [40.0, 44.5, 15.0, 17.0],
        "se": [1.0, 1.2, 0.8, 0.9],
    })
    fig = figures.search_ablation(res, group_col="benchmark", err_col="se")
    ax = fig.axes[0]
    assert len(_bars(ax)) == 4
    assert len([c for c in ax.containers if type(c).__name__ == "ErrorbarContainer"]) == 4
    assert [t.get_text() for t in ax.get_xticklabels()] == ["miniF2F", "ProofNet"]
    figures.save_pdf(fig, tmp_path / "ablation.pdf")


def test_search_ablation_err_col_needs_one_row_per_bar(plt):
    res = pd.DataFrame({"method": ["a", "a"], "solved": [1.0, 2.0], "se": [0.1, 0.1]})
    with pytest.raises(ValueError):
        figures.search_ablation(res, err_col="se")


def test_search_ablation_aggregates_seeds(plt):
    res = pd.DataFrame({
        "method": ["baseline"] * 3 + ["area prior"] * 3,
        "seed": [0, 1, 2] * 2,
        "solved": [40.0, 42.0, 41.0, 44.0, 46.0, 45.0],
    })
    fig = figures.search_ablation(res, errorbar="sd")
    ax = fig.axes[0]
    heights = sorted(b.get_height() for b in _bars(ax))
    assert heights == pytest.approx([41.0, 45.0])
    assert ax.get_legend() is not None


def test_search_ablation_err_col_with_missing_bar(plt):
    res = pd.DataFrame({
        "benchmark": ["miniF2F", "miniF2F", "ProofNet"],
        "method": ["baseline", "area prior", "baseline"],
        "solved": [40.0, 44.5, 15.0],
        "se": [1.0, 1.2, 0.8],
    })
    fig = figures.search_ablation(res, group_col="benchmark", err_col="se")
    assert len(_bars(fig.axes[0])) == 3


# ------------------------------------------------ style
def test_style_is_sans_serif_and_does_not_leak(plt):
    import matplotlib

    before = dict(matplotlib.rcParams)
    fig = figures.transition_heatmap(_msm())
    assert dict(matplotlib.rcParams) == before
    ax = fig.axes[0]
    family = ax.xaxis.label.get_fontfamily()
    assert family == ["sans-serif"]
    rc = figures.paper_rc()
    assert rc["font.sans-serif"][0] == "Arial"
    assert rc["axes.spines.top"] is False


def test_pdf_embeds_truetype_not_type3(plt, tmp_path):
    fig = figures.implied_timescales([1, 2, 4], np.array([[10.0], [11.0], [11.5]]))
    data = figures.save_pdf(fig, tmp_path / "its.pdf").read_bytes()
    assert b"/Type3" not in data
    assert b"/FontFile2" in data


def test_saved_pdf_uses_the_paper_font_not_the_global_default(plt, tmp_path):
    """Families resolve at draw time, so save_pdf must apply the paper fonts too."""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    if not available & {"Arial", "Helvetica", "Liberation Sans", "Nimbus Sans"}:
        pytest.skip("no sans-serif font from the paper stack is installed")
    fig = figures.implied_timescales([1, 2, 3], [[5.0], [6.0], [6.5]])
    out = figures.save_pdf(fig, tmp_path / "its.pdf")
    raw = out.read_bytes()
    assert b"DejaVuSans" not in raw.replace(b"DejaVuSans-", b"")  # text font, not math fallback
    assert any(name.encode() in raw for name in ("Arial", "Helvetica", "LiberationSans", "NimbusSans"))


def _paired_frames():
    rates = pd.DataFrame({"benchmark": ["miniF2F"] * 3, "method": ["LM only", "area", "random"],
                          "rate": [39.8, 39.3, 39.3], "lo": [33.6, 33.2, 33.2], "hi": [45.9, 45.5, 45.5]})
    diffs = pd.DataFrame({"benchmark": ["miniF2F"] * 2, "method": ["area", "random"], "diff": [-0.4, -0.4],
                          "lo": [-2.9, -2.5], "hi": [2.0, 1.6], "only_cond": [5, 4], "only_base": [6, 5]})
    return rates, diffs


def test_search_paired(plt, tmp_path):
    rates, diffs = _paired_frames()
    fig = figures.search_paired(rates, diffs, baseline="LM only")
    ax_r, ax_d = fig.axes[:2]
    assert [t.get_text() for t in ax_r.get_yticklabels()] == ["LM only", "area", "random"]
    lo, hi = ax_d.get_xlim()
    assert lo == pytest.approx(-hi) and hi >= 2.9  # symmetric around zero, every interval inside
    labels = [t.get_text() for t in ax_d.child_axes[0].get_yticklabels()]
    assert labels == ["", "+5 −6", "+4 −5"]
    assert fig.get_size_inches()[0] == pytest.approx(figures.COLUMN_WIDTH)
    raw = figures.save_pdf(fig, tmp_path / "paired.pdf").read_bytes()
    assert b"/MediaBox [ 0 0 234" in raw


def test_search_paired_rows_per_benchmark(plt):
    rates, diffs = _paired_frames()
    rates = pd.concat([rates, rates.assign(benchmark="PutnamBench")])
    diffs = pd.concat([diffs, diffs.assign(benchmark="PutnamBench")])
    fig = figures.search_paired(rates, diffs, baseline="LM only")
    assert fig.axes[0].get_ylabel() == "miniF2F" and fig.axes[2].get_ylabel() == "PutnamBench"


def test_search_paired_checks_inputs(plt):
    rates, diffs = _paired_frames()
    with pytest.raises(KeyError):
        figures.search_paired(rates.drop(columns="lo"), diffs, baseline="LM only")
    with pytest.raises(ValueError):
        figures.search_paired(rates, diffs, baseline="nope")
