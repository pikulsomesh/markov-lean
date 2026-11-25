"""Multi-judge agreement (src/markov_lean/agreement.py): pairwise and Fleiss' kappa, NMI per judge."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score, normalized_mutual_info_score

from markov_lean import agreement as ag
from markov_lean import cli, judge, pilot
from markov_lean import significance as sig
from markov_lean.synthetic import generate

FAST_MSM = {"n_microstates": 40, "lag": 1, "bayesian_samples": 0, "featurizer_kwargs": {"n_components": 32},
            "n_macrostates": 4}
H, S, L = "claude-haiku-5-5", "claude-sonnet-5-5", "ollama/qwen2.5:7b-instruct"


def test_fleiss_matches_the_published_example():
    # Fleiss (1971) as worked on Wikipedia: 10 subjects, 14 raters, 5 categories, kappa = 0.210
    counts = np.array([
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ], dtype=float)
    n = 14
    p_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    k = ag.fleiss_from_sums(np.array([10]), np.array([p_i.sum()]), counts.sum(0)[None], n)[0]
    assert k == pytest.approx(0.210, abs=5e-4)
    ratings = np.array([[c for c, m in enumerate(row) for _ in range(int(m))] for row in counts])
    assert ag.fleiss_kappa(ratings) == pytest.approx(k)


def test_fleiss_edge_cases():
    assert ag.fleiss_kappa(np.array([["a", "a", "a"], ["b", "b", "b"]])) == pytest.approx(1.0)
    assert ag.fleiss_kappa(np.array([["a", "a", "a"]])) == pytest.approx(1.0)
    assert np.isnan(ag.fleiss_kappa(np.empty((0, 3))))


def _three_judges(n_proofs=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_proofs):
        truth = rng.choice(["11", "26", "05", "15"], size=6)
        for step in [-1, *range(6)]:
            t = truth[0] if step == -1 else truth[step]
            for j, noise in ((H, 0.05), (S, 0.15), (L, 0.35)):
                lab = t if rng.random() > noise else rng.choice(["11", "26", "05", "15", "03"])
                rows.append({"proof_id": f"p{p}", "step": step, "judge": j, "msc": lab, "confidence": 0.8})
    df = judge.labels_frame(rows)
    # the local judge skipped a few proofs, so the overlaps differ
    return df[~((df.judge == L) & df.proof_id.isin(["p0", "p1", "p2"]))].reset_index(drop=True)


def test_pairwise_and_fleiss():
    labels = _three_judges()
    rep = ag.judge_agreement(labels, n_boot=200, seed=1)
    assert rep["judges"] == [H, S, L]
    c = rep["step"]["common"]
    assert c["n"] == 77 * 6 and c["n_proofs"] == 77
    w = ag.wide_labels(labels[labels.step >= 0], [H, S, L])
    pairs = {(p["a"], p["b"]): p for p in c["pairwise"]}
    assert set(pairs) == {(H, S), (H, L), (S, L)}
    for (a, b), p in pairs.items():
        assert p["kappa"] == pytest.approx(cohen_kappa_score(w[a], w[b]), abs=1e-10)
        lo, hi = p["kappa_ci95"]
        assert lo <= p["kappa"] <= hi
    assert pairs[(H, S)]["kappa"] > pairs[(S, L)]["kappa"]  # noisier judge agrees less
    assert c["fleiss_kappa"] == pytest.approx(ag.fleiss_kappa(w[[H, S, L]].to_numpy()), abs=1e-10)
    assert c["fleiss_kappa_ci95"][0] <= c["fleiss_kappa"] <= c["fleiss_kappa_ci95"][1]
    assert 0 < c["all_agree"] < 1
    # pairwise on each pair's own overlap: Haiku vs Sonnet keeps the three proofs the local judge skipped
    ov = {(o["a"], o["b"]): o for o in rep["step"]["pair_overlap"]}
    assert ov[(H, S)]["n"] == 80 * 6 and ov[(H, L)]["n"] == 77 * 6
    assert rep["proof"]["common"]["n"] == 77


def test_two_judges_and_unknown_judges():
    labels = _three_judges()
    rep = ag.judge_agreement(labels, [H, S], n_boot=0)
    assert "fleiss_kappa" not in rep["step"]["common"] and len(rep["step"]["common"]["pairwise"]) == 1
    assert np.isnan(rep["step"]["common"]["pairwise"][0]["kappa_ci95"][0])
    with pytest.raises(ValueError, match="not in the labels"):
        ag.judge_agreement(labels, [H, "gpt"], n_boot=0)


def test_judge_cli_agree_takes_three_judges(tmp_path, capsys):
    path = tmp_path / "labels.parquet"
    _three_judges().to_parquet(path, index=False)
    assert judge.main(["agree", "--labels", str(path), "--judges", H, S, L]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["step"]["common"]["pairwise"]) == 3 and "fleiss_kappa" in out["step"]["common"]


# --------------------------------------------------------------------------- on a pilot run


@pytest.fixture(scope="module")
def judged_run(tmp_path_factory):
    run = tmp_path_factory.mktemp("judged")
    steps, _ = generate(n_proofs=120, seed=2)
    cfg = {"data": {"source": "leandojo", "seed": 0}, "msm": FAST_MSM, "judge": {"sample_proofs": 60},
           "eval": {"gate_nmi": 0.15, "macrostate_sweep": [2, 4]}}
    plain = steps.drop(columns=["true_area"]).assign(source="leandojo")
    pilot.run(cfg, run, steps=plain, verbose=False)
    sample = pd.read_parquet(run / "judge" / "sample_steps.parquet")
    lab = sample[["proof_id", "step"]].merge(steps[["proof_id", "step", "true_area"]], on=["proof_id", "step"])
    rng = np.random.default_rng(1)
    noisy = lab["true_area"].where(rng.random(len(lab)) < 0.8, "Other")
    api = pd.concat([lab.assign(judge=H, msc=lab["true_area"], confidence=0.9),
                     lab.assign(judge=S, msc=noisy, confidence=0.8)])[judge.LABEL_COLUMNS]
    api.to_parquet(run / "judge" / "labels.parquet", index=False)
    pilot.run(cfg, run, steps=plain, verbose=False)
    # the local judge labels a third of the proofs, with more noise
    keep = sorted(lab["proof_id"].unique())[::3]
    lo = lab[lab.proof_id.isin(keep)]
    local = lo.assign(judge=L, msc=lo["true_area"].where(rng.random(len(lo)) < 0.6, "Other"),
                      confidence=0.7)[judge.LABEL_COLUMNS]
    (run / "judge" / "local").mkdir()
    local.to_parquet(run / "judge" / "local" / "labels.parquet", index=False)
    return run


def test_nmi_per_judge_on_a_run(judged_run, tmp_path):
    api = pd.read_parquet(judged_run / "judge" / "labels.parquet")
    local = pd.read_parquet(judged_run / "judge" / "local" / "labels.parquet")
    labels = pd.concat([api, local])
    rep = ag.report(labels, run=judged_run, n_boot=100, seed=0)
    nmi = rep["nmi"]
    assert set(nmi["common_steps"]) == {H, S, L, "Mathlib folder"}
    n_common = nmi["common_steps"][H]["n_steps"]
    assert n_common == nmi["common_steps"][L]["n_steps"] == nmi["common_steps"]["Mathlib folder"]["n_steps"] > 0
    assert nmi["own_steps"][H]["n_steps"] > nmi["own_steps"][L]["n_steps"]
    # the clean judge tracks the macrostates best, and the numbers are scikit-learn's
    assert nmi["common_steps"][H]["nmi"] > nmi["common_steps"][L]["nmi"]
    res = sig.load_msm(judged_run / "msm")
    steps = pd.read_parquet(judged_run / "judge" / "sample_steps.parquet")
    t, _ = sig.scored_table(res, steps, api[api.judge == H], "msc", [])
    assert nmi["own_steps"][H]["nmi"] == pytest.approx(normalized_mutual_info_score(t["label"], t["macrostate"]),
                                                       abs=1e-9)
    for j in (H, S, L):
        lo, hi = nmi["common_steps"][j]["nmi_ci95"]
        assert lo <= nmi["common_steps"][j]["nmi"] <= hi
    md = ag.markdown(rep)
    assert "Fleiss" in md and "NMI with the gate macrostates" in md and L in md


def test_cli_judge_agreement_writes_the_report(judged_run, tmp_path):
    out = tmp_path / "agree"
    assert cli.main(["judge-agreement", "--run", str(judged_run), "--labels",
                     str(judged_run / "judge" / "labels.parquet"),
                     str(judged_run / "judge" / "local" / "labels.parquet"), "--n-boot", "50",
                     "--out", str(out)]) == 0
    rep = json.loads((out / "judge_agreement.json").read_text())
    assert rep["judges"] == [H, S, L] and "nmi" in rep
    assert "Fleiss' kappa (3 judges)" in (out / "judge_agreement.md").read_text()


def test_significance_extra_labels_leave_the_gate_alone(judged_run, tmp_path):
    base = sig.run(judged_run, tmp_path / "a", n_boot=100, n_perm=20, verbose=False)
    extra = sig.run(judged_run, tmp_path / "b", n_boot=100, n_perm=20, verbose=False,
                    extra_labels=[judged_run / "judge" / "local" / "labels.parquet"])
    assert base["cross_family"] is None
    assert extra["gate1"] == base["gate1"] and extra["judge_agreement"] == base["judge_agreement"]
    cf = extra["cross_family"]
    assert cf["judges"] == [H, S, L] and "fleiss_kappa" in cf["step"]["common"]
    assert L in cf["nmi"]["common_steps"]
    md = (tmp_path / "b" / "significance.md").read_text()
    assert "## Judge agreement across model families" in md and "### Step labels" in md
    assert cli.main(["significance", "--run", str(judged_run), "--out", str(tmp_path / "c"), "--n-boot", "50",
                     "--n-perm", "10", "--extra-labels",
                     str(judged_run / "judge" / "local" / "labels.parquet")]) == 0
