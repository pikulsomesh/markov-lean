"""Markov state model pipeline on synthetic proofs with a known hidden area."""

import time

import numpy as np
import pandas as pd
import pytest

from markov_lean import evaluate as ev
from markov_lean.featurize import lean_tokens, make_featurizer
from markov_lean.msm import (
    ASSIGNMENT_COLUMNS,
    MSMConfig,
    choose_lag,
    bootstrap_timescales,
    count_model,
    map_microstates,
    run,
    sort_steps,
    split_trajectories,
)
from markov_lean.synthetic import generate

# Threshold for recovering the hidden area at the default synthetic settings
# (6 areas, p_stay 0.85, noise 0.45, 400 proofs).  Over seeds 0-2 the automatic
# pipeline scored NMI 0.87-0.91 and ARI 0.90-0.94; chance is about 0.  The
# folder baseline scores about 0.12.  0.75 leaves room for sklearn/deeptime
# version drift while still failing if the kinetics or clustering break.
NMI_THRESHOLD = 0.75


@pytest.fixture(scope="module")
def synthetic():
    steps, T = generate(n_proofs=400, seed=0)
    return steps, T


@pytest.fixture(scope="module")
def result(synthetic):
    steps, _ = synthetic
    t0 = time.perf_counter()
    res = run(steps, MSMConfig(seed=0, error_samples=20))
    res.report["_wall"] = time.perf_counter() - t0
    return res


def test_generator_is_seeded_and_matches_schema():
    a, Ta = generate(n_proofs=20, seed=3)
    b, Tb = generate(n_proofs=20, seed=3)
    pd.testing.assert_frame_equal(a, b)
    np.testing.assert_allclose(Ta, Tb)
    for col in ["proof_id", "step", "source", "split", "theorem", "file_path", "tactic",
                "state_before", "state_after", "true_area"]:
        assert col in a.columns
    assert set(a["source"]) == {"synthetic"}
    np.testing.assert_allclose(Ta.sum(1), 1.0)
    assert a.groupby("proof_id")["step"].apply(lambda s: list(s) == list(range(len(s)))).all()


def test_lean_tokens_keep_namespaces_and_parts():
    toks = lean_tokens("h : Nat.Prime.two_le p\n⊢ x = 2")
    assert {"Nat", "Nat.Prime", "Nat.Prime.two_le", "prime", "two", "le"} <= set(toks)
    assert "x" not in toks and "h" not in toks


def test_featurizer_registry():
    f = make_featurizer("tfidf_svd", n_components=4)
    X = f.fit_transform(["Nat.Prime p", "Nat.gcd a b", "Real.exp x", "Real.log y", "Nat.Prime q"])
    assert X.shape[0] == 5 and X.shape[1] <= 4
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)
    with pytest.raises(ValueError):
        make_featurizer("nope")


def test_counts_never_cross_proofs():
    # proof A visits 0 -> 1, proof B visits 2 -> 3.  Joining them would add 1 -> 2.
    steps = pd.DataFrame({"proof_id": ["A", "A", "B", "B"], "step": [0, 1, 0, 1], "state_before": list("wxyz")})
    df = sort_steps(steps)
    dtrajs = split_trajectories(df, np.array([0, 1, 2, 3]))
    assert [d.tolist() for d in dtrajs] == [[0, 1], [2, 3]]
    C = count_model(dtrajs, 1, 4).count_matrix
    assert C[1, 2] == 0 and C[0, 1] == 1 and C[2, 3] == 1 and C.sum() == 2


def test_rows_out_of_order_are_sorted_by_proof_then_step(synthetic):
    steps, _ = synthetic
    shuffled = steps.sample(frac=1.0, random_state=0)
    df = sort_steps(shuffled)
    assert (df.groupby("proof_id")["step"].diff().dropna() == 1).all()


def test_choose_lag_picks_first_converged():
    its = pd.DataFrame({"lag": [1, 2, 3, 4], "index": [1] * 4, "timescale": [2.0, 4.0, 4.2, 4.3]})
    assert choose_lag(its, 1, tol=0.1, max_lag=4) == (2, True)
    assert choose_lag(its, 1, tol=0.01, max_lag=4)[1] is False


def test_assignments_table(result, synthetic):
    steps, _ = synthetic
    a = result.assignments
    assert list(a.columns) == ASSIGNMENT_COLUMNS
    assert len(a) == len(steps)
    assert not a.duplicated(["proof_id", "step"]).any()
    assert a["macrostate"].between(0, result.report["n_macrostates"] - 1).all()
    assert a["membership"].between(0, 1).all()
    assert str(a["step"].dtype) == "int64"


def test_recovers_hidden_areas(result, synthetic):
    steps, _ = synthetic
    df = sort_steps(steps)
    score = ev.agreement(result.assignments["macrostate"], df["true_area"])
    assert score["nmi"] > NMI_THRESHOLD, score
    folder = ev.folder_baseline(df, "true_area")
    assert score["nmi"] > folder["nmi"] + 0.3


def test_automatic_choices_and_checks(result):
    rep = result.report
    assert rep["time_unit"] == "proof steps"
    assert rep["n_macrostates_chosen_by"] == "spectral_gap"
    assert rep["n_macrostates"] == 6          # six hidden areas
    assert rep["n_microstates_chosen_by"] == "vamp2_cv"
    assert rep["lag_chosen_by"] == "its_convergence"
    assert set(result.microstate_cv["n_microstates"]) == {25, 50, 100, 200}
    expected = {"connectivity", "sampling", "lag_vs_proof_length", "its_converged", "timescales_resolved",
                "spectral_gap", "pcca_crisp", "chapman_kolmogorov", "its_within_errors"}
    assert expected <= set(rep["checks"])
    failed = [k for k, c in rep["checks"].items() if not c["passed"]]
    assert rep["all_checks_passed"], failed
    # timescales are in proof steps: the slowest should be a few steps, longer than the lag
    assert rep["timescales_steps"][0] > rep["lag"]
    assert not result.timescales.empty and not result.ck.empty


def test_checks_flag_a_model_without_areas():
    # two barely separated areas forced into four macrostates: the extra sets have no
    # slow process behind them, so convergence, resolution and spectral-gap checks fail
    steps, _ = generate(n_proofs=150, areas=["11", "26"], p_stay=0.5, noise=0.9, leak=0.0, seed=1)
    res = run(steps, MSMConfig(n_microstates=20, n_macrostates=4, error_samples=0, seed=1))
    failed = [k for k, c in res.report["checks"].items() if not c["passed"]]
    assert failed, res.report["checks"]
    assert not res.report["all_checks_passed"]
    assert {"spectral_gap", "timescales_resolved"} <= set(failed)


def test_map_microstates_drop_and_nearest():
    # microstates 0..3; 2 is outside the connected set and sits next to 3
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.1], [5.0, 5.0]])
    connected = np.array([0, 1, 3])
    pcca_assign = np.array([0, 0, 1])
    memberships = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
    macro, member, moved = map_microstates(4, connected, pcca_assign, memberships, centers, "drop")
    assert macro.tolist() == [0, 0, -1, 1] and np.isnan(member[2]) and moved == []
    macro, member, moved = map_microstates(4, connected, pcca_assign, memberships, centers, "nearest")
    assert macro.tolist() == [0, 0, 1, 1] and member[2] == pytest.approx(0.7) and moved == [2]
    with pytest.raises(ValueError):
        map_microstates(4, connected, pcca_assign, memberships, centers, "other")


def test_largest_connected_set_excludes_one_way_states():
    # state 3 is only ever left, never entered, so it is not strongly connected
    dtrajs = [np.array([3, 0, 1, 0, 1, 2, 1, 2, 0])]
    kept = count_model(dtrajs, 1, 4).submodel_largest(directed=True).state_symbols
    assert sorted(kept.tolist()) == [0, 1, 2]


def test_drop_mode_marks_disconnected(synthetic):
    steps, _ = synthetic
    small = steps[steps["proof_id"].isin(steps["proof_id"].unique()[:40])]
    res = run(small, MSMConfig(n_microstates=60, lag=1, n_macrostates=3, disconnected="drop", error_samples=0))
    dropped = res.report["connected_set"]["dropped_microstates"]
    a = res.assignments
    # whether k-means leaves a sparse cluster outside the connected set depends on the
    # scikit-learn version, so only the invariant is checked here; the mapping itself is
    # tested deterministically in test_map_microstates_drop_and_nearest
    assert (a.loc[a["microstate"].isin(dropped), "macrostate"] == -1).all()
    assert (a.loc[~a["microstate"].isin(dropped), "macrostate"] >= 0).all()
    assert res.report["connected_set"]["n_kept"] + len(dropped) == res.report["n_microstates"]


def test_checks_are_reproducible(synthetic):
    # the same inputs and seed must give the same checks, error bars included
    steps, _ = synthetic
    small = steps[steps["proof_id"].isin(steps["proof_id"].unique()[:150])]
    cfg = MSMConfig(n_microstates=25, lag=1, n_macrostates=6, error_samples=10, seed=0)
    a, b = run(small, cfg).report, run(small, cfg).report
    assert a["timescale_errors_steps"] == b["timescale_errors_steps"]
    assert a["checks"] == b["checks"]


def test_bootstrap_interval_brackets_mean():
    dtrajs = [np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1]) for _ in range(10)] + [np.array([1, 0, 0, 1, 1, 1, 0])] * 5
    e = bootstrap_timescales(dtrajs, 1, 2, 1, 30, seed=0)
    assert e["method"] == "bootstrap" and e["lower"][0] <= e["mean"][0] <= e["upper"][0]
    assert e == bootstrap_timescales(dtrajs, 1, 2, 1, 30, seed=0)


def test_deprecated_bayesian_samples_alias(synthetic):
    steps, _ = synthetic
    small = steps[steps["proof_id"].isin(steps["proof_id"].unique()[:60])]
    rep = run(small, MSMConfig(n_microstates=20, lag=1, n_macrostates=3, bayesian_samples=0)).report
    assert rep["timescale_errors_steps"] is None and "its_within_errors" not in rep["checks"]


def test_save_writes_outputs(result, tmp_path):
    result.save(tmp_path)
    back = pd.read_parquet(tmp_path / "assignments.parquet")
    assert list(back.columns) == ASSIGNMENT_COLUMNS
    for name in ["implied_timescales.csv", "ck_test.csv", "microstate_cv.csv", "msm_report.json",
                 "coarse_T.npy", "micro_T.npy"]:
        assert (tmp_path / name).exists()


def test_runs_fast(result):
    assert result.report["_wall"] < 60
