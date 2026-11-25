"""search-report on the committed runs in tests/fixtures/search_runs (see tests/search_fixture.py).

The runs were written with ``search_eval.append_result``, so they have the
runner's exact layout and fields. Every expected number here follows from the
design in ``search_fixture.py``. No GPU, Lean or network.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import pytest

import search_fixture as fx
from markov_lean import search_report as sr

N = 12  # miniF2F problems in the fixture


@pytest.fixture(scope="module")
def df():
    return sr.load_summaries(fx.FIXTURE)


def _row(frame, **kw):
    m = np.ones(len(frame), bool)
    for k, v in kw.items():
        m &= (frame[k] == v).to_numpy()
    assert m.sum() == 1, kw
    return frame[m].iloc[0]


def test_committed_fixture_matches_generator(tmp_path):
    fresh = fx.write(tmp_path / "runs")
    committed = sorted(p.relative_to(fx.FIXTURE) for p in fx.FIXTURE.rglob("*.jsonl"))
    assert committed == sorted(p.relative_to(fresh) for p in fresh.rglob("*.jsonl"))
    for rel in committed:
        assert (fx.FIXTURE / rel).read_text() == (fresh / rel).read_text(), rel


def test_load_from_any_level(df):
    assert len(df) == (12 + 6) * len(fx.CONDITIONS) * len(fx.SEEDS)
    assert set(sr.COLUMNS) <= set(df.columns)
    bench_dirs = sorted(glob.glob(str(fx.FIXTURE / "*")))  # what `--runs results/search_eval/*` expands to
    assert len(sr.load_summaries(bench_dirs)) == len(df)
    one = sr.load_summaries(fx.FIXTURE / "minif2f" / "lm")
    assert set(one["condition"]) == {"lm"} and len(one) == N * 2
    seed = sr.load_summaries(fx.FIXTURE / "minif2f" / "lm" / "seed0" / "summary.jsonl")
    assert len(seed) == N and set(seed["seed"]) == {0}
    # the same run reached twice is read once
    assert len(sr.load_summaries([fx.FIXTURE, fx.FIXTURE / "putnam"])) == len(df)


def test_pass_and_paired(df):
    t = sr.pass_table(df, ks=(1, 2), n_boot=200)
    assert _row(t, benchmark="minif2f", condition="lm")["pass@1"] == pytest.approx(4.5 / N)
    assert _row(t, benchmark="minif2f", condition="lm")["pass@2"] == pytest.approx(5 / N)
    assert _row(t, benchmark="minif2f", condition="area_l0.3")["pass@1"] == pytest.approx(8 / N)
    assert _row(t, benchmark="putnam", condition="lm")["n_statement_errors"] == 1
    p = sr.paired_compare(df, n_boot=500)
    area = _row(p, benchmark="minif2f", condition="area_l0.3")
    assert area["diff"] == pytest.approx(3.5 / N) and area["diff_lo"] > 0
    assert (area["only_cond"], area["only_base"]) == (3, 0)
    rnd = _row(p, benchmark="minif2f", condition="random_l0.3")
    assert rnd["diff"] == 0 and rnd["p_mcnemar"] == 1.0


def test_area_gain(df):
    g = sr.area_gain_table(df, "root_area", n_boot=300)
    m = g[(g.benchmark == "minif2f") & (g.condition == "area_l0.3")].set_index("area")
    assert list(m.index) == ["0", "1", "2"] and m["n"].eq(4).all()
    np.testing.assert_allclose(m["diff"], [0.25, 0.375, 0.25])
    np.testing.assert_allclose(m["rate_base"], [0.5, 0.375, 0.25])
    assert (m["diff_lo"] <= m["diff"]).all() and (m["diff"] <= m["diff_hi"]).all()
    # area gains, weighted by problems, add up to the overall paired gain
    assert (m["diff"] * m["n"]).sum() / m["n"].sum() == pytest.approx(3.5 / N)
    rnd = g[(g.benchmark == "minif2f") & (g.condition == "random_l0.3")]
    assert rnd["diff"].eq(0).all()
    # PutnamBench has no root areas: one "unknown" group with all its problems
    put = g[(g.benchmark == "putnam") & (g.condition == "area_l0.3")]
    assert list(put["area"]) == ["unknown"] and put["n"].iloc[0] == 6
    assert sr.area_gain_table(df, "nope").empty
    assert sr.area_gain_table(df, "root_area", baseline="nope").empty
    names = sr.area_gain_table(df, "name_area", n_boot=0)
    assert {"algebra", "number_theory"} <= set(names["area"]) and names["diff_lo"].isna().all()


def test_search_stats(df):
    s = sr.search_stats(df)
    lm = _row(s, benchmark="minif2f", condition="lm")
    assert (lm["n_runs"], lm["n_solved"]) == (24, 9)
    assert lm["len_median"] == 2 and lm["len_mean"] == pytest.approx(16 / 9) and lm["len_max"] == 3
    assert lm["stop_budget"] == pytest.approx(14 / 24) and lm["stop_wall_time"] == pytest.approx(1 / 24)
    assert lm["stop_other"] == 0 and lm["nodes_per_expansion"] > 3
    area = _row(s, benchmark="minif2f", condition="area_l0.3")
    assert area["n_solved"] == 16 and area["exp_solved_median"] < lm["exp_solved_median"]
    put = _row(s, benchmark="putnam", condition="lm")
    assert put["stop_statement_error"] == pytest.approx(2 / 12)
    total = put[[c for c in s.columns if c.startswith("stop_")]].sum() + put["solve_rate"]
    assert total == pytest.approx(1.0)


def test_effort_compare(df):
    e = sr.effort_compare(df, n_boot=300)
    area = _row(e, benchmark="minif2f", condition="area_l0.3")
    assert (area["n_pairs"], area["n_problems"]) == (9, 5)
    assert area["exp_ratio"] == pytest.approx(0.5)
    assert area["exp_ratio_lo"] == pytest.approx(0.5) and area["exp_ratio_hi"] == pytest.approx(0.5)
    assert area["len_diff"] == pytest.approx(1.0)
    assert area["p_wilcoxon"] < 0.1
    rnd = _row(e, benchmark="minif2f", condition="random_l0.3")
    assert rnd["exp_ratio"] == pytest.approx(1.0) and rnd["p_wilcoxon"] == 1.0 and rnd["len_diff"] == 0
    assert sr.effort_compare(df, baseline="nope").empty


def test_budget_curves(df):
    c = sr.budget_curve(df, grid=[10, 64], n_boot=300)
    lm64 = _row(c, benchmark="minif2f", condition="lm", budget=64)
    assert lm64["solved_frac"] == pytest.approx(9 / 24)
    assert lm64["solved_lo"] <= lm64["solved_frac"] <= lm64["solved_hi"]
    assert _row(c, benchmark="minif2f", condition="lm", budget=10)["solved_frac"] == pytest.approx(1 / 24)
    assert _row(c, benchmark="minif2f", condition="area_l0.3", budget=10)["solved_frac"] == pytest.approx(9 / 24)
    assert sr.budget_curve(df, grid=[64], n_boot=0)["solved_lo"].isna().all()

    g = sr.budget_gain_curve(df, grid=[10, 64], n_boot=300)
    assert _row(g, benchmark="minif2f", condition="area_l0.3", budget=64)["diff"] == pytest.approx(3.5 / N)
    assert _row(g, benchmark="minif2f", condition="area_l0.3", budget=10)["diff"] == pytest.approx(4 / N)
    assert g.loc[g.condition == "random_l0.3", "diff"].eq(0).all()
    full = sr.budget_gain_curve(df, n_boot=0)
    assert full["budget"].max() == 64 and full["budget"].min() == 1


def test_latex_tables(df):
    conds = ["lm", "area_l0.3", "random_l0.3"]
    labels = sr._Labels({"lm": r"LM only ($\lambda = 0$)"})
    t = sr.pass_table(df, ks=(1, 2), n_boot=100)
    tex = sr._latex_table(t, conds, labels=labels)
    assert r"LM only ($\lambda = 0$)" in tex and "miniF2F & PutnamBench" in tex
    assert r"\textbf{66.7}" in tex and "area\\_l0.3" in tex
    multi = sr._latex_pass_multi(t, conds, [1, 2], labels)
    assert r"\multicolumn{2}{c}{miniF2F}" in multi and r"\cmidrule(lr){2-3}" in multi and "$k=2$" in multi
    paired = sr._latex_paired(sr.paired_compare(df, n_boot=100), labels)
    assert "+29.2 [" in paired and paired.count(r"\midrule") == 2
    gains = sr._latex_area_gain(sr.area_gain_table(df, "root_area", n_boot=100), conds, labels)
    assert r"\emph{miniF2F}" in gains and "+37.5" in gains and "unknown" in gains
    stats = sr._latex_stats(sr.search_stats(df), sr.effort_compare(df, n_boot=100), labels)
    assert "0.50 [0.50, 0.50]" in stats and "1.00" in stats
    for text in (tex, multi, paired, gains, stats):
        assert text.count(r"\begin{tabular}") == 1 and r"\bottomrule" in text
        body = [l for l in text.splitlines() if l.endswith(r"\\")]
        cols = {l.count("&") for l in body if r"\multicolumn" not in l}
        assert len(cols) == 1, text  # every row has the same number of cells


def test_cli_on_glob(tmp_path, capsys):
    mpl = pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    mpl.use("Agg", force=True)
    from markov_lean import cli

    out = tmp_path / "report"
    runs = sorted(glob.glob(str(fx.FIXTURE / "*")))
    assert cli.main(["search-report", "--runs", *runs, "--out", str(out), "--n-boot", "100",
                     "--label", r"area_l0.3=Area prior ($\lambda = 0.3$)", "--bench-label", "putnam=Putnam"]) == 0
    printed = capsys.readouterr().out
    assert "area_l0.3" in printed
    expected = ["pass_at_k.csv", "paired.csv", "by_area.csv", "by_area_gain.csv", "budget.csv", "budget_gain.csv",
                "search_stats.csv", "effort.csv", "report.md", "pass_at_k.tex", "pass_at_k_all.tex", "paired.tex",
                "by_area_gain.tex", "search_stats.tex", "fig_pass_at_1.pdf", "fig_budget.pdf",
                "fig_budget_gain.pdf", "fig_effort.pdf", "fig_by_area_minif2f.pdf", "fig_area_gain_minif2f.pdf",
                "fig_area_gain_putnam.pdf"]
    for name in expected:
        assert (out / name).stat().st_size > 0, name
    assert (out / "fig_area_gain_minif2f.pdf").read_bytes()[:4] == b"%PDF"
    assert r"Area prior ($\lambda = 0.3$)" in (out / "pass_at_k.tex").read_text()
    assert "Putnam &" in (out / "paired.tex").read_text()
    md = (out / "report.md").read_text()
    for head in ("## Gain over `lm` by area", "## Proof length and search effort", "## Effort on runs both solved"):
        assert head in md
    assert pd.read_csv(out / "by_area_gain.csv")["benchmark"].tolist()[0] == "minif2f"


def test_default_out_is_common_parent(tmp_path):
    runs = fx.write(tmp_path / "search_eval")
    benches = sorted(str(p) for p in runs.iterdir())
    assert sr.main(["--runs", *benches, "--n-boot", "10", "--no-figures"]) == 0
    assert (runs / "report" / "report.md").exists()
    # a second run over the same glob sees the report folder and skips it
    assert sr.main(["--runs", *sorted(str(p) for p in runs.iterdir()), "--n-boot", "10", "--no-figures"]) == 0
