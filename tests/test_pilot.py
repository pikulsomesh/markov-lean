"""End-to-end pilot on synthetic data, and the two-pass judge flow on a fake LeanDojo table."""

import json

import pandas as pd
import pytest

from markov_lean import pilot
from markov_lean.synthetic import generate

FAST_MSM = {"n_microstates": 40, "lag": 1, "bayesian_samples": 0, "featurizer_kwargs": {"n_components": 32}}


def test_synthetic_end_to_end(tmp_path):
    cfg = {"data": {"source": "synthetic", "synthetic": {"n_proofs": 200, "seed": 1}},
           "msm": FAST_MSM, "eval": {"gate_nmi": 0.15}}
    code = pilot.run(cfg, tmp_path, verbose=False)
    rep = json.loads((tmp_path / "pilot_report.json").read_text())
    assert code == 0
    assert rep["eval"]["gate1"]["passed"]
    assert rep["eval"]["macrostates"]["nmi"] > rep["eval"]["folder_baseline"]["nmi"]
    assert (tmp_path / "msm" / "assignments.parquet").exists()


def _as_leandojo(steps):
    return steps.drop(columns=["true_area"]).assign(source="leandojo")


def test_two_pass_judge_flow(tmp_path):
    steps, _ = generate(n_proofs=150, seed=2)
    truth = steps[["proof_id", "step", "true_area"]]
    cfg = {"data": {"source": "leandojo", "seed": 0}, "msm": FAST_MSM,
           "judge": {"sample_proofs": 40}, "eval": {"gate_nmi": 0.15}}

    # pass 1: no labels yet, so the sample and the next steps are written and nothing is scored
    assert pilot.run(cfg, tmp_path, steps=_as_leandojo(steps), verbose=False) == 0
    rep = json.loads((tmp_path / "pilot_report.json").read_text())
    assert rep["gate1"] == "waiting for judge labels"
    sample = pd.read_parquet(tmp_path / "judge" / "sample_steps.parquet")
    assert sample["proof_id"].nunique() == 40
    assert "ANTHROPIC_API_KEY" in (tmp_path / "judge" / "NEXT.md").read_text()
    assert "$" in rep["judge"]["estimate"] or "could not" not in rep["judge"]["estimate"]

    # pass 2: two fake judges that agree on the hidden area
    lab = sample[["proof_id", "step"]].merge(truth, on=["proof_id", "step"])
    labels = pd.concat([
        lab.assign(judge=j, msc=lab["true_area"], confidence=0.9) for j in ("judge-a", "judge-b")
    ])[["proof_id", "step", "judge", "msc", "confidence"]]
    labels.to_parquet(tmp_path / "judge" / "labels.parquet", index=False)
    code = pilot.run(cfg, tmp_path, steps=_as_leandojo(steps), verbose=False)
    rep = json.loads((tmp_path / "pilot_report.json").read_text())
    assert rep["eval"]["n_labelled_steps"] == len(lab)
    assert rep["judge_agreement"]["judges"] == ["judge-a", "judge-b"]
    assert code == (0 if rep["eval"]["gate1"]["passed"] else 1)
    assert rep["eval"]["gate1"]["passed"]


def test_whole_proof_labels_broadcast(tmp_path):
    steps, _ = generate(n_proofs=120, seed=3)
    cfg = {"data": {"source": "leandojo"}, "msm": FAST_MSM, "judge": {"sample_proofs": 30}}
    pilot.run(cfg, tmp_path, steps=_as_leandojo(steps), verbose=False)
    sample = pd.read_parquet(tmp_path / "judge" / "sample_steps.parquet")
    first = sample.sort_values("step").groupby("proof_id").head(1)
    labels = first.assign(step=-1, judge="consensus", msc=first["true_area"] if "true_area" in first else "26",
                          confidence=1.0)[["proof_id", "step", "judge", "msc", "confidence"]]
    labels.to_parquet(tmp_path / "judge" / "labels.parquet", index=False)
    pilot.run(cfg, tmp_path, steps=_as_leandojo(steps), verbose=False)
    rep = json.loads((tmp_path / "pilot_report.json").read_text())
    assert rep["eval"]["n_labelled_steps"] == len(sample)


def test_unknown_msm_key_is_refused():
    with pytest.raises(ValueError, match="unknown msm settings"):
        pilot.msm_config({"n_micro": 5})


def test_missing_processed_data_says_how_to_fetch(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_data.sh"):
        pilot.load_steps({"source": "leandojo", "processed_dir": str(tmp_path)})
