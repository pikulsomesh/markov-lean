"""Saved Markov model -> area prior for search, end to end on synthetic proofs."""

import numpy as np
import pytest

from markov_lean.area_prior import AreaPrior
from markov_lean.msm import MSMConfig, run
from markov_lean.prior_loader import build_area_prior, load_msm_artifacts
from markov_lean.search.runner import load_prior
from markov_lean.synthetic import generate


@pytest.fixture(scope="module")
def msm_dir(tmp_path_factory):
    steps, _ = generate(n_proofs=200, seed=5)
    res = run(steps, MSMConfig(n_microstates=40, lag=1, n_macrostates=6, bayesian_samples=0,
                               featurizer_kwargs={"n_components": 32}))
    out = tmp_path_factory.mktemp("msm")
    res.save(out)
    return out, res, steps


def test_artifacts_round_trip(msm_dir):
    out, res, steps = msm_dir
    art = load_msm_artifacts(out)
    assert art.centers.shape == res.centers.shape
    assert art.memberships.shape == (res.centers.shape[0], 6)
    assert art.coarse_T.shape == (6, 6)
    kept = art.memberships[res.connected]
    assert np.allclose(kept.sum(axis=1), 1.0)
    texts = steps["state_before"].head(5).tolist()
    assert np.allclose(art.featurizer.transform(texts), res.featurizer.transform(texts))


def test_prior_places_states_in_their_area(msm_dir):
    out, res, steps = msm_dir
    prior = build_area_prior(out, lam=0.5)
    assert isinstance(prior, AreaPrior) and prior.n_areas == 6
    # a training state's area distribution should favor the macrostate it was assigned
    a = res.assignments.merge(steps[["proof_id", "step", "state_before"]], on=["proof_id", "step"])
    a = a[a["macrostate"] >= 0].head(50)
    hits = [np.argmax(prior.area_of(t)) == m for t, m in zip(a["state_before"], a["macrostate"])]
    assert np.mean(hits) > 0.7


def test_prior_scores_candidates(msm_dir):
    out, _, steps = msm_dir
    prior = build_area_prior(out, lam=0.3)
    s = steps["state_before"].tolist()
    cands = [{"tactic": "simp", "state_after": s[1], "logprob": -1.0},
             {"tactic": "ring", "state_after": "no goals", "logprob": -3.0},
             {"tactic": "rw [h]", "state_after": s[40], "logprob": -0.5}]
    w = np.asarray(prior(s[0], cands))
    assert w.shape == (3,) and np.isclose(w.sum(), 1.0) and (w >= 0).all()
    prior.record(s[0], 0.0)
    prior.reset()
    assert prior.history == []


def test_search_runner_builds_it_from_config(msm_dir):
    out, _, _ = msm_dir
    prior = load_prior({"factory": "markov_lean.prior_loader:build_area_prior",
                         "kwargs": {"msm_dir": str(out), "lam": 0.3}})
    assert isinstance(prior, AreaPrior)


def test_old_run_without_artifacts_says_rerun(tmp_path):
    (tmp_path / "coarse_T.npy").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="Rerun `markov-lean pilot`"):
        load_msm_artifacts(tmp_path)
