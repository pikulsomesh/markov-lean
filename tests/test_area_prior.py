"""Tests for the area prior, on a toy two-area Markov model. Pure numpy, no I/O."""
from __future__ import annotations

import numpy as np
import pytest

from markov_lean import area_prior as ap

# Toy MSM: 4 microstates in 2-D, two metastable areas.
#   micro 0, 1 near (1, 0) -> area 0 ("algebra")
#   micro 2, 3 near (0, 1) -> area 1 ("analysis")
CENTROIDS = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
MEMBERSHIP = np.array([[0.95, 0.05], [0.9, 0.1], [0.05, 0.95], [0.1, 0.9]])
STICKY_T = np.array([[0.9, 0.1], [0.1, 0.9]])   # areas tend to persist
SWITCH_T = np.array([[0.1, 0.9], [0.9, 0.1]])   # areas tend to alternate


def featurize(text: str) -> np.ndarray:
    v = np.zeros(2)
    if "ring" in text:
        v[0] += 1.0
    if "limit" in text:
        v[1] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class BatchFeaturizer:
    """Shaped like featurize.TfidfSvdFeaturizer: transform(list[str]) -> (n, d)."""

    def transform(self, texts):
        return np.stack([featurize(t) for t in texts])


ALG, ANA = "⊢ ring x = y", "⊢ limit f = 0"


# ---------------------------------------------------------------- area_distribution

def test_area_distribution_picks_nearest_area():
    p = ap.area_distribution([1.0, 0.0], CENTROIDS, MEMBERSHIP, temperature=0.05)
    assert p.shape == (2,) and p.sum() == pytest.approx(1.0)
    assert p[0] > 0.85
    q = ap.area_distribution([0.0, 1.0], CENTROIDS, MEMBERSHIP, temperature=0.05)
    assert q[1] > 0.85


def test_area_distribution_zero_and_nonfinite_vectors_are_uniform():
    for v in ([0.0, 0.0], [np.nan, 1.0], [np.inf, 0.0]):
        np.testing.assert_allclose(ap.area_distribution(v, CENTROIDS, MEMBERSHIP), [0.5, 0.5])


def test_area_distribution_hard_assignment_and_top_k():
    hard = ap.area_distribution([1.0, 0.0], CENTROIDS, MEMBERSHIP, temperature=0)
    np.testing.assert_allclose(hard, MEMBERSHIP[0] / MEMBERSHIP[0].sum())
    # a hot softmax over all 4 microstates is near uniform; top-2 keeps it in area 0
    hot = ap.area_distribution([1.0, 0.0], CENTROIDS, MEMBERSHIP, temperature=100)
    top2 = ap.area_distribution([1.0, 0.0], CENTROIDS, MEMBERSHIP, temperature=100, k=2)
    assert abs(hot[0] - 0.5) < 0.05
    assert top2[0] > 0.9


def test_area_distribution_nan_membership_rows_spread_evenly():
    M = MEMBERSHIP.copy()
    M[0] = np.nan
    p = ap.area_distribution([1.0, 0.0], CENTROIDS, M, temperature=0)
    np.testing.assert_allclose(p, [0.5, 0.5])


def test_area_distribution_shape_errors():
    with pytest.raises(ValueError):
        ap.area_distribution([1.0, 0.0, 0.0], CENTROIDS, MEMBERSHIP)
    with pytest.raises(ValueError):
        ap.area_distribution([1.0, 0.0], CENTROIDS, MEMBERSHIP[:3])


# ---------------------------------------------------------------- reweight_prior

def test_reweight_prior_endpoints_and_mix():
    p_lm = np.array([0.7, 0.2, 0.1])
    scores = np.array([0.0, 1.0, 3.0])
    np.testing.assert_allclose(ap.reweight_prior(p_lm, scores, 0.0), p_lm)
    np.testing.assert_allclose(ap.reweight_prior(p_lm, scores, 1.0), [0.0, 0.25, 0.75])
    mid = ap.reweight_prior(p_lm, scores, 0.5)
    np.testing.assert_allclose(mid, 0.5 * p_lm + 0.5 * np.array([0.0, 0.25, 0.75]))
    assert mid.sum() == pytest.approx(1.0)


def test_reweight_prior_renormalizes_and_handles_zeros():
    out = ap.reweight_prior([2.0, 2.0], [0.0, 0.0], 0.5)      # unnormalized LM, zero scores
    np.testing.assert_allclose(out, [0.5, 0.5])
    out = ap.reweight_prior([1.0, 0.0], [-1.0, 1.0], 1.0)      # negative score clips to 0
    np.testing.assert_allclose(out, [0.0, 1.0])
    assert ap.reweight_prior([], [], 0.3).size == 0


def test_reweight_prior_rejects_bad_input():
    with pytest.raises(ValueError):
        ap.reweight_prior([0.5, 0.5], [1.0, 1.0], 1.5)
    with pytest.raises(ValueError):
        ap.reweight_prior([0.5, 0.5], [1.0, 1.0], -0.1)
    with pytest.raises(ValueError):
        ap.reweight_prior([0.5, 0.5], [1.0], 0.5)
    with pytest.raises(ValueError):
        ap.reweight_prior([np.nan, 0.5], [1.0, 1.0], 0.5)


def test_logprobs_to_probs():
    np.testing.assert_allclose(ap.logprobs_to_probs(np.log([0.2, 0.8])), [0.2, 0.8])
    np.testing.assert_allclose(ap.logprobs_to_probs([-1000.0, -1000.0]), [0.5, 0.5])
    np.testing.assert_allclose(ap.logprobs_to_probs([-np.inf, 0.0]), [0.0, 1.0])
    np.testing.assert_allclose(ap.logprobs_to_probs([-np.inf, -np.inf]), [0.5, 0.5])


# ---------------------------------------------------------------- diversity_bonus

def test_diversity_bonus():
    np.testing.assert_allclose(ap.diversity_bonus([0, 0, 0]), [1, 1, 1])
    b = ap.diversity_bonus([0, 3, 8])
    np.testing.assert_allclose(b, [1.0, 0.5, 1 / 3])
    assert np.all(np.diff(b) < 0)
    np.testing.assert_allclose(ap.diversity_bonus([5, 1], alpha=0), [1, 1])
    assert ap.diversity_bonus([]).size == 0
    with pytest.raises(ValueError):
        ap.diversity_bonus([-1, 0])
    with pytest.raises(ValueError):
        ap.diversity_bonus([1, 0], alpha=-1)


# ---------------------------------------------------------------- stuck_escape

def test_stuck_escape_no_bias_without_enough_history():
    np.testing.assert_allclose(ap.stuck_escape([], window=3, threshold=1, n_areas=2), [1, 1])
    hist = [(0, 0.0), (0, 0.0)]
    np.testing.assert_allclose(ap.stuck_escape(hist, window=3, threshold=1, n_areas=2), [1, 1])


def test_stuck_escape_penalizes_stuck_area_only():
    hist = [(1, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)]
    b = ap.stuck_escape(hist, window=3, threshold=1.0, n_areas=2, strength=2.0)
    np.testing.assert_allclose(b, [np.exp(-2.0), 1.0])


def test_stuck_escape_off_when_progress_made():
    hist = [(0, 0.0), (0, 1.0), (0, 0.5)]
    np.testing.assert_allclose(ap.stuck_escape(hist, window=3, threshold=1.0, n_areas=2), [1, 1])


def test_stuck_escape_soft_areas_and_errors():
    hist = [([0.5, 0.5], 0.0), ([1.0, 0.0], 0.0)]
    b = ap.stuck_escape(hist, window=2, threshold=1.0, n_areas=2, strength=1.0)
    np.testing.assert_allclose(b, np.exp(-np.array([0.75, 0.25])))
    with pytest.raises(ValueError):
        ap.stuck_escape(hist, window=0, threshold=1.0, n_areas=2)
    with pytest.raises(ValueError):
        ap.stuck_escape([(5, 0.0)], window=1, threshold=1.0, n_areas=2)


# ---------------------------------------------------------------- make_prior

def _prior(**kw):
    kw.setdefault("temperature", 0.05)
    return ap.make_prior(featurize, CENTROIDS, MEMBERSHIP, **kw)


def test_prior_signature_and_normalization():
    prior = _prior(lam=0.5)
    cands = [("ring_nf", ALG), ("simp", ANA), ("norm_num", "no goals")]
    w = prior(ALG, cands)
    assert w.shape == (3,) and w.sum() == pytest.approx(1.0) and np.all(w >= 0)
    assert prior(ALG, []).size == 0


def test_prior_lam_zero_returns_lm_and_lam_one_ignores_it():
    cands = [{"tactic": "a", "state_after": ALG, "logprob": np.log(0.9)},
             {"tactic": "b", "state_after": ANA, "logprob": np.log(0.1)}]
    np.testing.assert_allclose(_prior(lam=0.0, coarse_T=SWITCH_T)(ALG, cands), [0.9, 0.1])
    w = _prior(lam=1.0, coarse_T=SWITCH_T)(ALG, cands)
    assert w[1] > w[0]      # area term alone prefers the switch the MSM predicts


def test_prior_follows_coarse_transition_matrix():
    cands = [("x", ALG), ("y", ANA)]
    stay = _prior(lam=1.0, coarse_T=STICKY_T, diversity_alpha=0)(ALG, cands)
    move = _prior(lam=1.0, coarse_T=SWITCH_T, diversity_alpha=0)(ALG, cands)
    assert stay[0] > stay[1]
    assert move[1] > move[0]


def test_prior_closed_goal_is_never_penalized_and_unknown_is_neutral():
    prior = _prior(lam=1.0, coarse_T=STICKY_T)
    w = prior(ALG, [("done", "no goals"), ("x", ALG), ("y", ANA), "bare_tactic"])
    assert w[0] == pytest.approx(w.max())
    assert w[2] < w[3] < w[1]       # unknown result sits between the two areas


def test_prior_diversity_and_stuck_escape_push_out_of_area():
    cands = [("x", ALG), ("y", ANA)]
    prior = _prior(lam=1.0, coarse_T=STICKY_T, window=3, threshold=1.0, escape_strength=5.0)
    before = prior(ALG, cands)
    assert before[0] > before[1]
    for _ in range(3):
        prior.record(ALG, progress=0.0)
    after = prior(ALG, cands)
    assert after[1] > after[0]      # stuck in area 0 for 3 steps -> prefer area 1
    prior.reset()
    np.testing.assert_allclose(prior(ALG, cands), before)


def test_prior_accepts_batch_featurizer_and_zero_vector_states():
    prior = ap.make_prior(BatchFeaturizer(), CENTROIDS, MEMBERSHIP, lam=1.0, temperature=0.05)
    w = prior("⊢ True", [("x", "⊢ True"), ("y", "⊢ 1 = 1")])   # both featurize to zero
    np.testing.assert_allclose(w, [0.5, 0.5])


def test_prior_custom_hooks():
    seen = []

    def next_state(state, cand):
        seen.append((state, cand))
        return ANA if cand == "go-analysis" else ALG

    prior = _prior(lam=1.0, coarse_T=SWITCH_T, next_state=next_state,
                   lm_logprob=lambda c: 0.0)
    w = prior(ALG, ["stay", "go-analysis"])
    assert w[1] > w[0] and len(seen) == 2


def test_make_prior_validates():
    with pytest.raises(ValueError):
        _prior(lam=2.0)
    with pytest.raises(ValueError):
        _prior(coarse_T=np.eye(3))
    with pytest.raises(ValueError):
        ap.make_prior(featurize, CENTROIDS[:3], MEMBERSHIP)
