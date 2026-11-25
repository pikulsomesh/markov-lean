"""The giant-state fixes: featurizer views, proof-weighted k-means and the jump-chain MSM."""

import pickle

import numpy as np
import pytest

from markov_lean.featurize import make_featurizer, state_view
from markov_lean.msm import MSMConfig, collapse_repeats, estimate_msm, run, step_weights
from markov_lean.synthetic import generate

STATE = """case h.inl
f g : MulRingNorm ℚ
hfp : ∀ (n : ℕ),
    f ↑n ≤ 1
⊢ f ↑n ^ t⁻¹ =
    Real.exp t

case b
x : ℕ
⊢ 0 < Nat.succ x"""


def test_state_view_typed_drops_binders_and_case_tags():
    typed = state_view(STATE, "typed")
    assert "hfp" not in typed and "case" not in typed and "inl" not in typed
    assert "MulRingNorm ℚ" in typed and "f ↑n ≤ 1" in typed  # continuation line kept
    assert "Real.exp t" in typed and "Nat.succ x" in typed and "⊢" not in typed


def test_state_view_goal_keeps_only_goals():
    goal = state_view(STATE, "goal")
    assert "MulRingNorm" not in goal and "f ↑n ≤ 1" not in goal
    assert "Real.exp t" in goal and "Nat.succ x" in goal
    assert state_view(STATE, "full") == STATE
    with pytest.raises(ValueError):
        state_view(STATE, "bogus")


def test_typed_featurizer_vocabulary_and_pickle():
    texts = [STATE, STATE.replace("hfp", "hbound"), "hx : 0 < x\n⊢ Real.exp x ≠ 0"] * 3
    full = make_featurizer("tfidf_svd", n_components=4, min_df=1)
    typed = make_featurizer("tfidf_svd", n_components=4, min_df=1, view="typed")
    full.fit_transform(texts)
    X = typed.fit_transform(texts)
    assert "hfp" in full.vectorizer.vocabulary_
    assert not {"hfp", "hbound", "hx"} & set(typed.vectorizer.vocabulary_)
    again = pickle.loads(pickle.dumps(typed))  # full.py ships the featurizer to worker processes
    np.testing.assert_allclose(again.transform(texts), X, atol=1e-6)
    with pytest.raises(ValueError):
        make_featurizer("tfidf_svd", view="bogus")


def test_step_weights():
    codes = np.array([0, 0, 0, 0, 1, 1, 2])
    assert step_weights(codes, "step") is None
    w = step_weights(codes, "proof")
    assert w.mean() == pytest.approx(1.0)
    # each proof gets the same total weight
    np.testing.assert_allclose(np.bincount(codes, weights=w), len(codes) / 3)
    with pytest.raises(ValueError):
        step_weights(codes, "bogus")


def test_collapse_repeats():
    out = collapse_repeats([np.array([3, 3, 1, 1, 1, 3]), np.array([2]), np.array([], dtype=np.int64)])
    assert [d.tolist() for d in out] == [[3, 1, 3], [2], []]


def _sticky_area_trajectories(seed=0):
    """Two areas of 5 microstates each, plus microstate 10 in area A that a few long proofs sit in.

    Proofs hop between microstates of their area (switching area with probability 0.05 per jump)
    and dwell in each microstate for a few steps; the sticky proofs stay in microstate 10.
    """
    rng = np.random.default_rng(seed)
    areas = [np.arange(5), np.arange(5, 10)]
    trajs = []
    for _ in range(300):
        a = int(rng.integers(2))
        d = []
        for _ in range(int(rng.integers(4, 10))):
            if rng.random() < 0.05:
                a = 1 - a
            d += [int(rng.choice(areas[a]))] * int(rng.integers(1, 4))
        trajs.append(np.array(d))
    for _ in range(15):  # sticky proofs: enter microstate 10 from area A and stay a long time
        trajs.append(np.array([0, 1] + [10] * 60 + [2]))
    return trajs


def _two_sets(dtrajs):
    msm, _ = estimate_msm(dtrajs, 1, 11)
    sets = msm.pcca(2).assignments
    return {int(s): int(m) for s, m in zip(msm.count_model.state_symbols, sets)}


def test_jump_chain_finds_areas_where_step_counts_isolate_a_sticky_microstate():
    trajs = _sticky_area_trajectories()
    by_step = _two_sets(trajs)
    # counted per step, the sticky microstate is the slowest process and gets a set to itself
    assert [s for s in by_step if by_step[s] == by_step[10]] == [10]
    by_jump = _two_sets(collapse_repeats(trajs))
    # counted per jump, the two sets are the two areas
    a, b = {by_jump[s] for s in range(5)}, {by_jump[s] for s in range(5, 10)}
    assert len(a) == 1 and len(b) == 1 and a != b


def test_run_with_jump_chain_reports_jumps():
    steps, _ = generate(n_proofs=150, seed=1)
    res = run(steps, MSMConfig(seed=0, error_samples=0, n_microstates=20, lag=1, n_macrostates=3,
                               jump_chain=True, kmeans_weight="proof",
                               featurizer_kwargs={"n_components": 16, "view": "typed"}))
    assert res.report["time_unit"] == "microstate jumps"
    assert res.report["config"]["jump_chain"] is True
    assert 0 <= res.report["trajectory_length"]["fraction_one_state"] <= 1
    assert res.report["trajectory_length"]["median"] <= res.report["proof_length"]["median"]
    assert set(res.assignments["macrostate"]) <= {0, 1, 2}
