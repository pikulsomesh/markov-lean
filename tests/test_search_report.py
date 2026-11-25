"""Tests for the search report. Synthetic summaries in tmp_path only; no GPU, Lean or network."""
from __future__ import annotations

import itertools
import json
from math import comb

import numpy as np
import pandas as pd
import pytest

from markov_lean import search_report as sr

N_PROBLEMS = 30
SEEDS = (0, 1, 2)
# problems solved per condition (deterministic): the area prior adds problems 10..17
SOLVES = {"lm": 10, "area_l0.3": 18, "diversity": 11}


def _line(bench, cond, seed, i, solved, stop=None, expansions=None):
    return {
        "proof_id": f"{bench}_{i}", "theorem": f"t{i}", "source": bench, "split": "test",
        "solved": solved, "stop_reason": stop or ("solved" if solved else "budget"),
        "n_tactics": 3 if solved else 0, "expansions": expansions if expansions is not None else (5 + i if solved else 64),
        "nodes": 10, "elapsed": 1.0 + i, "proof": "simp" if solved else "", "benchmark": bench,
        "condition": cond, "seed": seed, "max_expansions": 64, "wall_time": 60.0, "root_area": i % 3,
        "name_area": None,
    }


def _write(root, bench, cond, seed, lines):
    d = root / bench / cond / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "summary.jsonl", "a") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    for bench in ("minif2f", "putnam"):
        for cond, n_solved in SOLVES.items():
            for seed in SEEDS:
                lines = []
                for i in range(N_PROBLEMS):
                    solved = i < n_solved or (i == 25 and seed == 0)  # one seed-dependent solve
                    if bench == "putnam" and cond == "diversity" and i == 29:
                        lines.append(_line(bench, cond, seed, i, False, stop="statement error: unknown id"))
                        continue
                    lines.append(_line(bench, cond, seed, i, solved))
                if bench == "minif2f" and cond == "lm" and seed == 0:
                    # resumed run: problem 0 first logged as a wall-time stop, then re-run and solved
                    lines.insert(0, _line(bench, cond, seed, 0, False, stop="wall_time"))
                _write(root, bench, cond, seed, lines)
    with open(root / "putnam" / "lm" / "seed1" / "summary.jsonl", "a") as f:
        f.write('{"proof_id": "trunc')  # crash mid-write
    return root


# ------------------------------------------------------------ estimators
def test_pass_at_k_matches_enumeration():
    for n in range(1, 7):
        for c in range(n + 1):
            items = [1] * c + [0] * (n - c)
            for k in range(1, n + 1):
                subsets = list(itertools.combinations(items, k))
                brute = sum(any(s) for s in subsets) / len(subsets)
                assert sr.pass_at_k(n, c, k) == pytest.approx(brute)
                assert sr.pass_at_k(n, c, k) == pytest.approx(1 - comb(n - c, k) / comb(n, k))
    assert 0 < sr.pass_at_k(10_000, 3, 100) < 1
    with pytest.raises(ValueError):
        sr.pass_at_k(2, 1, 3)


def test_pass_table_by_hand():
    rows = [  # p1 solved in 1 of 2 seeds, p2 in none (statement error in seed 1), p3 only in seed 0
        ("p1", 0, True, "solved", 4), ("p1", 1, False, "budget", 8),
        ("p2", 0, False, "budget", 8), ("p2", 1, False, "statement error: x", 0),
        ("p3", 0, True, "solved", 2),
    ]
    df = pd.DataFrame([{"benchmark": "b", "condition": "lm", "proof_id": p, "seed": s, "solved": v,
                        "stop_reason": r, "expansions": e, "elapsed": float(e)} for p, s, v, r, e in rows])
    t = sr.pass_table(df, ks=(1, 2, 4), n_boot=200)
    r = t.iloc[0]
    assert r["n_problems"] == 2 and r["n_seeds"] == 2
    assert r["pass@1"] == pytest.approx(0.25)
    assert r["pass@2"] == pytest.approx(0.5)
    assert "pass@4" not in t.columns
    assert r["pass@1_lo"] <= r["pass@1"] <= r["pass@1_hi"]
    assert r["n_statement_errors"] == 1
    assert r["mean_expansions_solved"] == pytest.approx(4)
    assert r["median_elapsed"] == pytest.approx(6)  # median of 4, 8, 8, 0


def test_holm():
    np.testing.assert_allclose(sr.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(sr.holm([0.5, 0.5]), [1.0, 1.0])
    assert len(sr.holm([])) == 0


def _paired_df(only_cond, only_base, both, neither):
    rows, i = [], 0
    for cond_s, base_s, n in ((1, 0, only_cond), (0, 1, only_base), (1, 1, both), (0, 0, neither)):
        for _ in range(n):
            rows += [{"benchmark": "b", "condition": "x", "proof_id": f"p{i}", "seed": 0, "solved": bool(cond_s)},
                     {"benchmark": "b", "condition": "lm", "proof_id": f"p{i}", "seed": 0, "solved": bool(base_s)}]
            i += 1
    return pd.DataFrame(rows)


def test_mcnemar_hand_computed():
    out = sr.paired_compare(_paired_df(6, 1, 3, 10), n_boot=500)
    r = out.iloc[0]
    assert (r["only_cond"], r["only_base"], r["both"], r["neither"], r["n_common"]) == (6, 1, 3, 10, 20)
    assert r["p_mcnemar"] == pytest.approx(2 * (1 + 7) / 2**7)  # two-sided binomial, 1 of 7 discordant
    assert r["diff"] == pytest.approx(5 / 20)
    assert r["diff_lo"] <= r["diff"] <= r["diff_hi"]
    assert r["p_holm"] == pytest.approx(r["p_mcnemar"])
    assert sr.paired_compare(_paired_df(0, 0, 2, 2), n_boot=50).iloc[0]["p_mcnemar"] == 1.0
    assert sr.paired_compare(_paired_df(1, 0, 1, 1), baseline="nope").empty


# ------------------------------------------------------------ on synthetic runs
def test_load_dedupes_resumed_lines(runs):
    df = sr.load_summaries(runs)
    assert len(df) == 2 * len(SOLVES) * len(SEEDS) * N_PROBLEMS
    assert not df.duplicated(sr.KEY).any()
    r = df[(df.benchmark == "minif2f") & (df.condition == "lm") & (df.seed == 0) & (df.proof_id == "minif2f_0")]
    assert len(r) == 1 and bool(r["solved"].iloc[0]) and r["stop_reason"].iloc[0] == "solved"
    assert set(sr.COLUMNS) <= set(df.columns)
    assert sr.load_summaries(runs / "missing").empty


def test_tables_on_runs(runs):
    df = sr.load_summaries(runs)
    t = sr.pass_table(df, ks=(1, 2, 3, 4), n_boot=200).set_index(["benchmark", "condition"])
    assert "pass@4" not in t.columns
    lm = t.loc[("minif2f", "lm")]
    assert lm["n_problems"] == N_PROBLEMS and lm["n_seeds"] == 3
    assert lm["pass@1"] == pytest.approx((10 + 1 / 3) / N_PROBLEMS)
    assert lm["pass@3"] == pytest.approx(11 / N_PROBLEMS)
    div = t.loc[("putnam", "diversity")]
    assert div["n_statement_errors"] == 1 and div["n_problems"] == N_PROBLEMS

    p = sr.paired_compare(df, n_boot=300).set_index(["benchmark", "condition"])
    area = p.loc[("minif2f", "area_l0.3")]
    assert area["n_common"] == N_PROBLEMS and area["diff"] == pytest.approx(8 / N_PROBLEMS)
    assert area["only_cond"] == 8 and area["only_base"] == 0 and area["diff_lo"] > 0
    assert area["p_mcnemar"] == pytest.approx(2 / 2**8) and area["p_boot"] < 0.05
    assert (p["p_holm"] >= p["p_mcnemar"] - 1e-12).all()

    a = sr.area_table(df, "root_area", n_boot=100)
    assert set(a["area"]) == {"0", "1", "2"}
    assert a.groupby(["benchmark", "condition"])["n"].sum().eq(N_PROBLEMS).all()
    assert sr.area_table(df, "no_such_col").empty
    nulls = sr.area_table(df.assign(root_area=None), "root_area", n_boot=10)
    assert set(nulls["area"]) == {"unknown"}


def test_budget_curve_monotone(runs):
    df = sr.load_summaries(runs)
    c = sr.budget_curve(df)
    assert c["budget"].max() == 64 and c["budget"].min() == 1
    for (_, _), g in c.groupby(["benchmark", "condition"]):
        g = g.sort_values("budget")
        assert (np.diff(g["solved_frac"].to_numpy()) >= 0).all()
    last = c[(c.benchmark == "minif2f") & (c.condition == "area_l0.3") & (c.budget == 64)]
    assert last["solved_frac"].iloc[0] == pytest.approx((18 * 3 + 1) / (N_PROBLEMS * 3))
    assert sr.budget_curve(df, grid=[10, 5]).groupby("condition").size().eq(4).all()


def test_write_report_and_main(runs, tmp_path, capsys):
    mpl = pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    mpl.use("Agg", force=True)
    out = tmp_path / "rep"
    paths = sr.write_report(runs, out, n_boot=50)
    expected = {"pass_at_k", "paired", "by_area", "budget", "report", "latex", "fig_pass_at_1", "fig_budget",
                "fig_by_area_minif2f", "fig_by_area_putnam"}
    assert expected <= set(paths)
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    assert paths["fig_budget"].read_bytes()[:4] == b"%PDF"
    table = pd.read_csv(paths["pass_at_k"])
    assert {"pass@1", "pass@2"} <= set(table.columns) and "pass@4" not in table.columns
    assert list(dict.fromkeys(table["condition"]))[0] == "lm"
    md = paths["report"].read_text()
    assert "## pass@k" in md and "area\\_l0.3" in paths["latex"].read_text() and "\\toprule" in paths["latex"].read_text()
    assert "area_l0.3" in md and "p Holm" in md

    assert sr.main(["--runs", str(runs), "--n-boot", "20", "--no-figures", "--k", "1"]) == 0
    printed = capsys.readouterr().out
    assert str(runs / "report" / "report.md") in printed and "area_l0.3" in printed
    assert not (runs / "report" / "fig_budget.pdf").exists()
