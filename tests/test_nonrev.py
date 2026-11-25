"""Reversible vs non-reversible MSMs: GPCCA, the absorbing end state, and the comparison command."""

import json

import numpy as np
import pandas as pd
import pytest
from deeptime.markov import pcca as deeptime_pcca
from sklearn.metrics import adjusted_rand_score

from markov_lean import cli
from markov_lean import nonrev as nr
from markov_lean.msm import MSMConfig, run
from markov_lean.synthetic import generate


def block_matrix(n_blocks=3, size=4, inside=0.95, cyclic=False, seed=0):
    """Metastable blocks; ``cyclic`` sends all between-block mass one way (block b -> b+1)."""
    rng = np.random.default_rng(seed)
    n = n_blocks * size
    blk = np.repeat(np.arange(n_blocks), size)
    if not cyclic:
        K = rng.random((n, n)) * np.where(blk[:, None] == blk[None, :], 1.0, 0.02)
        K = K + K.T
        T = K / K.sum(1, keepdims=True)
        return T, K.sum(1) / K.sum(), blk
    T = np.zeros((n, n))
    for i in range(n):
        same = np.flatnonzero(blk == blk[i])
        nxt = np.flatnonzero(blk == (blk[i] + 1) % n_blocks)
        T[i, same] = inside * rng.dirichlet(np.ones(size))
        T[i, nxt] = (1 - inside) * rng.dirichlet(np.ones(size))
    return T, None, blk


def sample(T, n_traj, length, rng, start=None, exit_prob=None):
    """Trajectories of a chain; with ``exit_prob`` each step ends the trajectory with that state-dependent chance."""
    n = len(T)
    out = []
    for _ in range(n_traj):
        x = int(rng.integers(n)) if start is None else int(rng.choice(n, p=start))
        d = [x]
        for _ in range(length - 1):
            if exit_prob is not None and rng.random() < exit_prob[x]:
                break
            x = int(rng.choice(n, p=T[x]))
            d.append(x)
        out.append(np.array(d, dtype=np.int64))
    return out


# ---------------------------------------------------------------- GPCCA


def test_gpcca_reproduces_pcca_on_a_reversible_matrix():
    T, pi, blk = block_matrix()
    ref = deeptime_pcca(T, 3, pi)
    g = nr.gpcca(T, 3, pi)
    assert adjusted_rand_score(blk, g.assignments) == 1.0
    assert adjusted_rand_score(np.argmax(ref.memberships, 1), g.assignments) == 1.0
    # same memberships up to the order of the sets
    perm = [int(np.argmax(np.bincount(g.assignments[np.argmax(ref.memberships, 1) == j], minlength=3)))
            for j in range(3)]
    np.testing.assert_allclose(np.asarray(ref.memberships), g.memberships[:, perm], atol=0.05)


def test_schur_basis_is_eta_orthonormal_invariant_and_starts_constant():
    T, _, _ = block_matrix(cyclic=True)
    eta = np.random.default_rng(1).random(len(T)) + 0.5
    eta /= eta.sum()
    X, ev = nr.schur_basis(T, 3, eta)
    np.testing.assert_allclose(X[:, 0], 1.0, atol=1e-10)
    np.testing.assert_allclose(X.T @ (eta[:, None] * X), np.eye(3), atol=1e-10)
    # invariant subspace: T X = X R for some R
    R = np.linalg.lstsq(X, T @ X, rcond=None)[0]
    np.testing.assert_allclose(T @ X, X @ R, atol=1e-10)
    assert np.isclose(np.max(np.abs(ev)), 1.0)


def test_gpcca_finds_one_way_blocks_and_keeps_their_direction():
    T, _, blk = block_matrix(cyclic=True)
    g = nr.gpcca(T, 3)
    assert g.n_sets == 3
    assert adjusted_rand_score(blk, g.assignments) == 1.0
    assert np.all(g.memberships >= 0)
    np.testing.assert_allclose(g.memberships.sum(1), 1.0)
    # coarse chain: forward mass well above backward
    order = [int(np.bincount(g.assignments[blk == b]).argmax()) for b in range(3)]
    Pc = g.coarse_T[np.ix_(order, order)]
    for b in range(3):
        assert Pc[b, (b + 1) % 3] > 0.03 > abs(Pc[(b + 1) % 3, b])


def test_complex_pair_is_never_split():
    # a pure one-way 3-cycle with self loops: eigenvalues 1 and a conjugate pair
    T = nr.directed_area_matrix(3, 0.7, 1.0)
    with pytest.raises(nr.SplitPairError):
        nr.schur_basis(T, 2, np.full(3, 1 / 3))
    g = nr.gpcca(T, 2)
    assert g.n_sets == 3 and g.n_sets_requested == 2
    assert nr.choose_n_sets(block_matrix(cyclic=True)[0], max_sets=6) == 3


# ---------------------------------------------------------------- absorbing end state


def test_absorbing_counts_send_every_step_to_a_successor_or_done():
    d = [np.array([0, 1, 1, 2]), np.array([2, 0])]
    C = nr.absorbing_counts(d, 2, 3)
    done = 3
    assert C.sum() == 6 + 2  # one count per step plus done -> done once per proof
    assert C[0, 1] == 1 and C[1, 2] == 1  # 0 -> 1 (t=0 -> 2), 1 -> 2 (t=1 -> 3)
    assert C[1, done] == 1 and C[2, done] == 2 and C[0, done] == 1
    assert C[done, done] == 2


def test_absorbing_view_matches_its_empirical_values_on_a_markov_chain():
    # two areas of three microstates; leaving is state-dependent and memoryless, so the model must agree
    rng = np.random.default_rng(0)
    T, _, blk = block_matrix(n_blocks=2, size=3, cyclic=True, inside=0.85, seed=2)
    exit_prob = np.array([0.05, 0.1, 0.15, 0.08, 0.12, 0.2])
    dtrajs = sample(T, 4000, 200, rng, exit_prob=exit_prob)
    ab = nr.absorbing_analysis(dtrajs, 6, blk.astype(np.int64), 2)
    assert abs(ab["switch_model"] - ab["switch_empirical"]) < 0.03
    assert ab["max_abs_hit_error"] < 0.03
    assert ab["max_abs_stay_error"] < 0.03
    # and against the true chain: expected steps to no goals from each state
    Q = (1 - exit_prob)[:, None] * T
    true_left = np.linalg.solve(np.eye(6) - Q, np.ones(6))
    w = np.bincount(np.concatenate(dtrajs), minlength=6)
    for A in range(2):
        m = blk == A
        assert ab["steps_left_model"][A] == pytest.approx(np.sum(w[m] * true_left[m]) / w[m].sum(), rel=0.08)


def test_chapman_kolmogorov_is_small_for_sampled_markov_chains():
    rng = np.random.default_rng(3)
    T, _, _ = block_matrix(cyclic=True, inside=0.8)
    dtrajs = sample(T, 400, 40, rng)
    fits = nr.fit_all(dtrajs, len(T), 1, 3)
    err = {}
    for name in nr.FITS:
        ck = nr.ck_table(fits[name], dtrajs, len(T), 4)
        err[name] = (ck["predicted"] - ck["estimated"]).abs().max()
    assert err["nonreversible"] < 0.03 and err["absorbing"] < 0.03
    assert err["reversible"] > 2 * err["nonreversible"]  # detailed balance cannot follow a one-way cycle


# ---------------------------------------------------------------- comparison on synthetic proofs

MCFG = dict(n_microstates=40, lag=1, n_macrostates=6, error_samples=0, its_lags=[1, 2, 3], ck_mlags=3, seed=0)


@pytest.fixture(scope="module")
def directed():
    steps, T = nr.generate_directed(n_proofs=200, forward=1.0, seed=0)
    res, _ = nr.compare_steps(steps, MSMConfig(**MCFG), label_col="true_area")
    return res, T


@pytest.fixture(scope="module")
def symmetric():
    steps, T = nr.generate_directed(n_proofs=200, forward=0.5, seed=0)
    res, _ = nr.compare_steps(steps, MSMConfig(**MCFG), label_col="true_area")
    return res, T


def _coarse_error(res, T_true, fit):
    """Frobenius error of a fit's coarse matrix against the true area chain, sets matched by majority label."""
    from markov_lean.synthetic import AREA_VOCAB

    areas = list(AREA_VOCAB)
    a = res.assignments.assign(area=res.report["_labels"])
    col = f"macro_{fit}"
    to_area = a[a[col] >= 0].groupby(col)["area"].agg(lambda s: s.value_counts().index[0])
    idx = [areas.index(to_area[i]) for i in range(len(to_area))]
    assert sorted(idx) == list(range(len(areas)))
    Pc = np.asarray(res.report["fits"][fit]["coarse_T"])
    return float(np.linalg.norm(Pc - T_true[np.ix_(idx, idx)]))


def _attach_labels(res, steps):
    k = res.assignments[["proof_id", "step"]]
    lab = steps.astype({"proof_id": str}).set_index(["proof_id", "step"])["true_area"]
    res.report["_labels"] = lab.reindex(pd.MultiIndex.from_frame(k)).to_numpy()


def test_directed_proofs_reject_detailed_balance_and_favor_the_nonreversible_fit(directed):
    res, T = directed
    steps, _ = nr.generate_directed(n_proofs=200, forward=1.0, seed=0)
    _attach_labels(res, steps)
    r = res.report
    irr = r["irreversibility"]["macrostate_nonreversible"]
    assert irr["p_value"] < 1e-6
    assert irr["cv_nonrev_minus_rev_per_transition"] > 0.02
    assert irr["entropy_production_per_lag"] > 0.05
    ck = {n: r["fits"][n]["checks"]["chapman_kolmogorov"]["value"] for n in nr.FITS}
    assert ck["nonreversible"] < ck["reversible"]
    assert r["vs_labels"]["nonreversible"]["nmi"] > 0.8
    assert _coarse_error(res, T, "nonreversible") < 0.5 * _coarse_error(res, T, "reversible")
    # the same areas either way: direction changes the kinetics, not the partition
    assert r["agreement"]["reversible_vs_nonreversible"]["ari"] > 0.9
    assert r["agreement"]["pcca_vs_gpcca_on_reversible"]["ari"] > 0.95


def test_symmetric_proofs_do_not_reject_detailed_balance(symmetric):
    res, _ = symmetric
    irr = res.report["irreversibility"]["macrostate_nonreversible"]
    assert irr["p_value"] > 1e-3
    assert abs(irr["cv_nonrev_minus_rev_per_transition"]) < 0.01
    assert irr["entropy_production_per_lag"] < 0.02
    assert res.report["agreement"]["reversible_vs_nonreversible"]["ari"] > 0.9


def test_absorbing_fit_gives_the_end_state_a_set_and_keeps_every_step(directed):
    res, _ = directed
    f = res.report["fits"]["absorbing"]
    assert f["n_sets"] == res.report["n_macrostates"] + 1
    assert (res.assignments["macro_absorbing"] >= 0).all()
    assert f["n_states_active"] == res.report["n_microstates"] + 1


def test_report_and_outputs(directed, tmp_path):
    res, _ = directed
    out = res.save(tmp_path)
    paths = nr.figures(res, out)
    rep = json.loads((out / "nonrev_report.json").read_text())
    for key in ("fits", "agreement", "irreversibility", "absorbing_view", "checks", "vs_labels"):
        assert key in rep
    for n in nr.FITS:
        assert {"spectral_gap", "chapman_kolmogorov", "crisp"} <= set(rep["fits"][n]["checks"])
    assert set(pd.read_csv(out / "implied_timescales.csv")["fit"]) == set(nr.FITS)
    assert [p.name for p in paths] == ["nonrev_comparison.pdf", "nonrev_absorbing.pdf"]
    assert all(p.stat().st_size > 1000 for p in paths)
    assert "Non-reversible + GPCCA" in nr.summary_md(res)


def test_saved_model_round_trip_reproduces_the_reversible_partition(tmp_path):
    steps, _ = generate(n_proofs=120, seed=1)
    fit = run(steps, MSMConfig(n_microstates=30, lag=1, n_macrostates=4, error_samples=0, its_lags=[1, 2, 3]))
    fit.save(tmp_path / "msm")
    res = nr.compare_saved(tmp_path / "msm")
    saved = fit.assignments.sort_values(["proof_id", "step"]).reset_index(drop=True)
    assert adjusted_rand_score(saved["macrostate"], res.assignments["macro_reversible"]) == 1.0
    assert res.report["lag"] == 1 and res.report["n_macrostates"] == 4


def test_cli_synthetic(tmp_path):
    out = tmp_path / "nr"
    assert cli.main(["nonrev-check", "--synthetic", "directed", "--n-proofs", "100", "--out", str(out)]) == 0
    for name in ("nonrev_report.json", "SUMMARY.md", "nonrev_comparison.pdf", "nonrev_absorbing.pdf",
                 "assignments.parquet", "ck_test.csv", "occupancy.csv"):
        assert (out / name).exists(), name
