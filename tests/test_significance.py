"""Gate 1 significance tests (src/markov_lean/significance.py) on synthetic fixtures."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score

from markov_lean import cli, pilot
from markov_lean import significance as sig
from markov_lean.synthetic import generate

FAST_MSM = {"n_microstates": 40, "lag": 1, "bayesian_samples": 0, "featurizer_kwargs": {"n_components": 32},
            "n_macrostates": 4}
FAST = {"n_boot": 200, "n_perm": 50, "verbose": False}


def _table(x, y):
    x, _ = sig._codes(pd.Series(x))
    y, _ = sig._codes(pd.Series(y))
    C = np.zeros((x.max() + 1, y.max() + 1))
    np.add.at(C, (x, y), 1)
    return C


@pytest.mark.parametrize("seed", range(6))
def test_contingency_metrics_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 300))
    x = rng.integers(0, int(rng.integers(1, 8)), n)
    y = np.where(rng.random(n) < 0.6, x, rng.integers(0, 5, n))
    nmi, ari = sig.nmi_ari_from_tables(_table(x, y))
    assert nmi[0] == pytest.approx(normalized_mutual_info_score(y, x), abs=1e-10)
    assert ari[0] == pytest.approx(adjusted_rand_score(y, x), abs=1e-10)
    k, _ = sig.kappa_from_tables(_square(x, y))
    assert k[0] == pytest.approx(cohen_kappa_score(x, y), abs=1e-10)


def _square(x, y):
    K = int(max(x.max(), y.max())) + 1
    C = np.zeros((K, K))
    np.add.at(C, (x, y), 1)
    return C


def test_degenerate_partitions_match_sklearn():
    for x, y in [([0, 0, 0], [0, 0, 0]), ([0, 1, 2], [0, 1, 2]), ([0, 0, 0], [0, 1, 1]), ([0, 1, 2], [0, 0, 0])]:
        nmi, ari = sig.nmi_ari_from_tables(_table(x, y))
        assert nmi[0] == pytest.approx(normalized_mutual_info_score(y, x))
        assert ari[0] == pytest.approx(adjusted_rand_score(y, x))


def test_weighted_tables_equal_duplicated_rows():
    """A bootstrap weight of w on a proof scores like w copies of its steps."""
    rng = np.random.default_rng(0)
    proof = np.repeat(np.arange(30), 5)
    x = rng.integers(0, 4, len(proof))
    y = (x + (rng.random(len(proof)) < 0.3)) % 4
    M = sig.proof_tables(proof, 30, x, y, 4, 4)
    W = rng.multinomial(30, np.full(30, 1 / 30), size=3).astype(float)
    nmi, ari = sig.nmi_ari_from_tables(sig._resampled_tables(M, W, (4, 4)))
    for b in range(3):
        rows = np.repeat(np.arange(len(proof)), W[b][proof].astype(int))
        assert nmi[b] == pytest.approx(normalized_mutual_info_score(y[rows], x[rows]), abs=1e-10)
        assert ari[b] == pytest.approx(adjusted_rand_score(y[rows], x[rows]), abs=1e-10)


def test_holm_and_bh():
    p = [0.01, 0.04, 0.03, 0.5]
    assert sig.holm(p) == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert sig.benjamini_hochberg(p) == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])
    assert all(h >= q for h, q in zip(sig.holm(p), p))


def test_bootstrap_pvalues():
    rng = np.random.default_rng(0)
    clear = sig.bootstrap_pvalues(0.2, rng.normal(0.2, 0.02, 2000))
    assert clear["p_greater"] < 0.01 and clear["p_two_sided"] < 0.01
    null = sig.bootstrap_pvalues(0.0, rng.normal(0.0, 0.02, 2000))
    assert null["p_greater"] > 0.3


# --------------------------------------------------------------------------- end to end on pilot runs


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("pilot_synth")
    cfg = {"data": {"source": "synthetic", "synthetic": {"n_proofs": 160, "seed": 1}},
           "msm": FAST_MSM, "eval": {"gate_nmi": 0.15, "macrostate_sweep": [2, 3, 4]}}
    pilot.run(cfg, out, verbose=False)
    (out / "config.yaml").write_text(json.dumps(cfg))  # JSON is valid YAML
    return out


def test_reloaded_model_reproduces_the_sweep(synthetic_run):
    cfg = json.loads((synthetic_run / "config.yaml").read_text())
    steps = pilot.sample_proofs(pilot.load_steps(cfg["data"]), None, 2, 0)
    fitted = pilot.run_msm(steps, pilot.msm_config(cfg["msm"]))
    reloaded = sig.load_msm(synthetic_run / "msm")
    a, b = pilot.macrostate_sweep(fitted, [2, 3, 4]), pilot.macrostate_sweep(reloaded, [2, 3, 4])
    for k in a:
        assert adjusted_rand_score(a[k], b[k]) == pytest.approx(1.0)


def test_synthetic_run_end_to_end(synthetic_run, tmp_path):
    rep = sig.run(synthetic_run, tmp_path, config_path=synthetic_run / "config.yaml", seed=3, **FAST)
    pilot_rep = json.loads((synthetic_run / "pilot_report.json").read_text())

    # point estimates come from pilot.score and agree with the paired table
    assert rep["pilot_score"]["macrostates"]["nmi"] == pytest.approx(pilot_rep["eval"]["macrostates"]["nmi"])
    assert rep["partitions"]["msm"]["nmi"]["value"] == pytest.approx(rep["pilot_score"]["macrostates"]["nmi"], abs=1e-9)
    assert rep["partitions"]["folder"]["nmi"]["value"] == pytest.approx(
        rep["pilot_score"]["folder_baseline"]["nmi"], abs=1e-9)
    for k, v in rep["pilot_score"]["macrostate_sweep"].items():
        assert rep["partitions"][f"k={k}"]["nmi"]["value"] == pytest.approx(v["nmi"], abs=1e-9)

    # synthetic macrostates track the hidden areas step by step, so everything is significant
    d = rep["vs_folder"]["msm"]["nmi"]
    assert d["difference"] > 0 and d["ci95"][0] > 0 and d["p_greater"] < 0.05
    assert rep["gate1"]["significant"]
    assert not rep["labels_constant_within_proofs"]
    assert rep["shuffle_nulls"]["msm"]["within_proof"]["p"] < 0.05
    assert rep["shuffle_nulls"]["msm"]["global"]["null_mean"] < 0.05
    for test in rep["corrected"].values():
        for v in test.values():
            assert v["p"] <= v["bh"] <= v["holm"] <= 1
    assert rep["judge_agreement"] is None

    md = (tmp_path / "significance.md").read_text()
    assert "Gate 1 significance" in md and "Holm" in md and "within-proof" in md
    assert json.loads((tmp_path / "significance.json").read_text())["settings"]["resampling_unit"] == "proof"


def test_seeded(synthetic_run, tmp_path):
    a = sig.run(synthetic_run, tmp_path / "a", config_path=synthetic_run / "config.yaml", seed=7, **FAST)
    b = sig.run(synthetic_run, tmp_path / "b", config_path=synthetic_run / "config.yaml", seed=7, **FAST)
    assert a["vs_folder"] == b["vs_folder"] and a["shuffle_nulls"] == b["shuffle_nulls"]


def test_no_signal_is_not_significant(synthetic_run, tmp_path):
    """Labels unrelated to the steps: the MSM should not beat the folder significantly."""
    cfg = json.loads((synthetic_run / "config.yaml").read_text())
    steps = pilot.sample_proofs(pilot.load_steps(cfg["data"]), None, 2, 0)
    rng = np.random.default_rng(0)
    noise = steps.assign(true_area=rng.choice(list("ABCDE"), len(steps)))
    path = tmp_path / "noise.parquet"
    noise.to_parquet(path, index=False)
    rep = sig.run(synthetic_run, tmp_path / "out", steps_path=path, seed=0, **FAST)
    assert rep["vs_folder"]["msm"]["nmi"]["ci95"][0] < 0.02
    assert rep["shuffle_nulls"]["msm"]["within_proof"]["p"] > 0.01
    assert not all(v["holm"] < 0.05 for v in rep["corrected"]["nmi_within_proof_null"].values())


def _judge_run(tmp_path, whole_proof=False):
    steps, _ = generate(n_proofs=120, seed=2)
    cfg = {"data": {"source": "leandojo", "seed": 0}, "msm": FAST_MSM, "judge": {"sample_proofs": 60},
           "eval": {"gate_nmi": 0.15, "macrostate_sweep": [2, 4]}}
    plain = steps.drop(columns=["true_area"]).assign(source="leandojo")
    pilot.run(cfg, tmp_path, steps=plain, verbose=False)
    sample = pd.read_parquet(tmp_path / "judge" / "sample_steps.parquet")
    lab = sample[["proof_id", "step"]].merge(steps[["proof_id", "step", "true_area"]], on=["proof_id", "step"])
    if whole_proof:
        lab = lab.groupby("proof_id", as_index=False)["true_area"].first().assign(step=-1)
    rng = np.random.default_rng(1)
    noisy = lab["true_area"].where(rng.random(len(lab)) < 0.8, "Other")
    labels = pd.concat([
        lab.assign(judge="claude-haiku-5-5", msc=lab["true_area"], confidence=0.9),
        lab.assign(judge="claude-sonnet-5-5", msc=noisy, confidence=0.8),
    ])[["proof_id", "step", "judge", "msc", "confidence"]]
    labels.to_parquet(tmp_path / "judge" / "labels.parquet", index=False)
    pilot.run(cfg, tmp_path, steps=plain, verbose=False)
    return labels


def test_judge_run_with_kappa(tmp_path):
    labels = _judge_run(tmp_path)
    assert cli.main(["significance", "--run", str(tmp_path), "--n-boot", "200", "--n-perm", "30"]) == 0
    rep = json.loads((tmp_path / "significance" / "significance.json").read_text())
    jt = rep["judge_agreement"]
    assert jt["judges"] == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    a = labels[labels.judge == "claude-haiku-5-5"].sort_values(["proof_id", "step"])
    b = labels[labels.judge == "claude-sonnet-5-5"].sort_values(["proof_id", "step"])
    assert jt["step"]["kappa"] == pytest.approx(cohen_kappa_score(a.msc, b.msc), abs=1e-10)
    lo, hi = jt["step"]["kappa_ci95"]
    assert lo <= jt["step"]["kappa"] <= hi and hi - lo < 0.3
    assert jt["proof"]["n"] == 0
    # consensus labels are what gets scored, as in the pilot
    pilot_rep = json.loads((tmp_path / "pilot_report.json").read_text())
    assert rep["pilot_score"]["n_labelled_steps"] == pilot_rep["eval"]["n_labelled_steps"]
    assert "Judge agreement" in (tmp_path / "significance" / "significance.md").read_text()


def test_whole_proof_labels_flag_the_degenerate_null(tmp_path):
    _judge_run(tmp_path, whole_proof=True)
    rep = sig.run(tmp_path, seed=0, **FAST)
    assert rep["labels_constant_within_proofs"]
    assert rep["shuffle_nulls"]["msm"]["within_proof"]["p"] == pytest.approx(1.0)
    assert rep["judge_agreement"]["proof"]["n"] > 0 and rep["judge_agreement"]["step"]["n"] == 0
    assert "cannot move NMI" in (tmp_path / "significance" / "significance.md").read_text()


def test_cli_help_lists_significance(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "significance" in capsys.readouterr().out
