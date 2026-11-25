"""Gate 1 robustness checks on synthetic data: fast, offline, no API calls."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from markov_lean import cli
from markov_lean import robustness as rb

TINY = dict(lags=[1, 2], macrostates=[2, 4, 6], microstates=[20, 30], featurizers=["tfidf_svd:16"],
            n_permutations=30, n_bootstrap=30, memory_folds=3, memory_betas=[1.0, 10.0, 100.0])


# ------------------------------------------------------------------ metrics


def test_table_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 6, 500)
    b = (a + rng.integers(0, 3, 500)) % 8
    t = rb.table(a, 6, b, 8)
    assert rb.nmi_table(t) == pytest.approx(normalized_mutual_info_score(b, a))
    assert rb.ari_table(t) == pytest.approx(adjusted_rand_score(b, a))


def test_codes_marks_missing():
    c, n = rb.codes(pd.Series(["11", None, "", "26", "11"]))
    assert n == 2
    assert list(c) == [0, -1, -1, 1, 0]
    c, n = rb.codes(pd.Series([3, -1, 3]))
    assert list(c) == [0, -1, 0] and n == 1


def test_static_groupings():
    assert rb.folder_path("Mathlib/NumberTheory/Padics/Basic.lean", 1) == "NumberTheory"
    assert rb.folder_path("Mathlib/NumberTheory/Padics/Basic.lean", 2) == "NumberTheory/Padics"
    assert rb.folder_path("Mathlib/Order/Basic.lean", 2) == "Order/Basic"
    assert rb.folder_path("Other/X.lean") is None
    assert rb.namespace("Nat.Prime.two_le", 1) == "Nat"
    assert rb.namespace("Nat.Prime.two_le", 2) == "Nat.Prime"
    assert rb.namespace("foo", 1) == "_root_"


def test_expand_labels_prefers_step_labels():
    rows = pd.DataFrame({"proof_id": ["p", "p", "q"], "step": [0, 1, 0]})
    lab = pd.DataFrame({"proof_id": ["p", "p", "q"], "step": [-1, 1, 5], "msc": ["11", "26", "05"]})
    assert list(rb.expand_labels(rows, lab).fillna("na")) == ["11", "26", "na"]
    assert list(rb.proof_labels(rows, lab).fillna("na")) == ["11", "11", "na"]


def test_permutation_null_and_bootstrap():
    rng = np.random.default_rng(1)
    proof = np.repeat(np.arange(60), 8)
    lab = np.repeat(rng.integers(0, 4, 60), 8)
    good = lab.copy()
    bad = rng.integers(0, 4, len(lab))
    null_good = rb.permutation_null(good, 4, lab, 4, proof, 50, 0)
    assert null_good["proof"]["p_nmi"] < 0.05 and null_good["row"]["p_nmi"] < 0.05
    # an unrelated partition is not significant on average (p is uniform under the null)
    ps = [rb.permutation_null(np.random.default_rng(s).integers(0, 4, len(lab)), 4, lab, 4, proof, 40, s)
          ["proof"]["p_nmi"] for s in range(10)]
    assert np.mean(ps) > 0.2
    boot = rb.bootstrap_proofs({"good": (good, 4), "bad": (bad, 4)}, lab, 4, proof, 50, 0, reference="good")
    lo, hi = boot["ci"]["good"]["nmi"]
    assert lo <= 1.0 <= hi + 1e-9
    assert boot["paired"]["bad"]["nmi_diff_ci"][0] > 0


# ------------------------------------------------------------------ memory


def _chain(order, n_runs, length, S, seed):
    rng = np.random.default_rng(seed)
    T = rng.dirichlet(np.ones(S) * 0.3, size=S ** order)
    runs = []
    for _ in range(n_runs):
        x = list(rng.integers(S, size=order))
        while len(x) < length:
            ctx = 0
            for v in x[-order:]:
                ctx = ctx * S + v
            x.append(int(rng.choice(S, p=T[ctx])))
        runs.append(np.array(x))
    return runs


def test_backoff_large_beta_equals_lower_order():
    runs = _chain(1, 40, 12, 4, 0)
    lo = rb.BackoffChain(1, 4, [3.0]).fit(runs).loglik(runs, 2)[0]
    hi = rb.BackoffChain(2, 4, [3.0, 1e12]).fit(runs).loglik(runs, 2)[0]
    assert hi == pytest.approx(lo, rel=1e-6)


@pytest.mark.parametrize("order,expect_gain", [(1, False), (2, True)])
def test_memory_test_detects_second_order(order, expect_gain):
    runs = _chain(order, 300, 14, 3, order)
    res = rb.memory_test(runs, np.arange(len(runs)), 3, [0, 1, 2], 3, [0.3, 3.0, 30.0, 300.0], 100, 0)
    g = res["gain_2_vs_1"]
    if expect_gain:
        assert g["ci"][0] > 0
    else:
        assert g["mean"] < 0.02


# ------------------------------------------------------------------ end to end


@pytest.fixture(scope="module")
def pilot_dir(tmp_path_factory):
    return rb.save_synthetic_pilot(tmp_path_factory.mktemp("pilot"), n_proofs=120, seed=0)


def test_run_writes_tables(pilot_dir, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "rob"
    s = rb.run(pilot_dir, out, rb.RobustnessConfig(**TINY), verbose=False)
    for f in ("baselines.csv", "sweep.csv", "memory.csv", "kmeans_by_k.csv", "robustness.json", "robustness.md"):
        assert (out / f).exists(), f
    assert (out / "figures" / "nmi_by_k.pdf").exists()
    base = pd.read_csv(out / "baselines.csv").set_index("grouping")
    for g in ("macrostate", "folder", "kmeans_steps", "kmeans_microstates", "random", "proof_label"):
        assert g in base.index
    # the synthetic areas are learnable: macrostates beat the folder and the random partition
    assert base.loc["macrostate", "nmi"] > base.loc["folder", "nmi"]
    assert base.loc["macrostate", "nmi"] > base.loc["random", "nmi"] + 0.3
    sweep = pd.read_csv(out / "sweep.csv")
    assert set(sweep["lag"]) == {1, 2} and set(sweep["n_microstates"]) == {20, 30}
    assert {"ck_max_error", "ck_passed", "its_converged", "nmi", "ami"} <= set(sweep.columns)
    # the sweep's cell at the pilot's own setting refits the same model
    assert s["sweep_summary"]["pilot_setting_nmi"] == pytest.approx(base.loc["macrostate", "nmi"], abs=1e-9)
    rep = json.loads((out / "robustness.json").read_text())
    assert rep["judge_ceiling"]["judges"] == ["judge-a", "judge-b"]
    assert {m["sequence"] for m in rep["memory"]} >= {"macrostates", "microstates"}
    assert "robust_pass" in rep["verdict"]


def test_ck_reuses_pcca_like_deeptime(pilot_dir, tmp_path):
    """The sweep's CK error (one PCCA+ per k) equals deeptime's own ck_test on the same model."""
    from markov_lean.msm import cluster_microstates, estimate_msm

    inp = rb.load_pilot(pilot_dir)
    X = rb.featurize(inp.steps["state_before"].astype(str).tolist(), "tfidf_svd:16")
    proof, _ = rb.codes(inp.assignments["proof_id"])
    km = cluster_microstates(X, 20, 0)
    np.save(tmp_path / "m.npy", km.labels_.astype(np.int64))
    np.save(tmp_path / "c.npy", km.cluster_centers_)
    lab, n_lab = rb.codes(rb.scored_frame(inp)["label"])
    rows = rb._lag_cell({"micro_path": tmp_path / "m.npy", "centers_path": tmp_path / "c.npy", "spec": "x",
                         "lag": 1, "ks": [3], "ck_mlags": 3, "proof": proof, "lab": lab, "n_lab": n_lab})
    d = [np.asarray(t) for t in np.split(km.labels_.astype(np.int64), np.flatnonzero(np.diff(proof)) + 1)]
    models = [estimate_msm(d, L, 20)[0] for L in (1, 2, 3)]
    ck = models[0].ck_test(models, 3, include_lag0=False)
    want = float(np.abs(np.asarray(ck.predictions) - np.asarray(ck.estimates)).max())
    assert rows[0]["ck_max_error"] == pytest.approx(want)


def test_cli_parallel_no_figures(pilot_dir, tmp_path):
    out = tmp_path / "cli"
    code = cli.main(["robustness", "--pilot", str(pilot_dir), "--out", str(out), "--jobs", "2", "--no-figures",
                     "--lags", "1", "--microstates", "20", "30", "--macrostates", "3", "6",
                     "--featurizers", "tfidf_svd:16", "--permutations", "20", "--bootstrap", "20"])
    assert code in (0, 1)
    assert len(pd.read_csv(out / "sweep.csv")) == 4  # 2 microstate counts x 1 lag x 2 k
    assert not (out / "figures").exists()


def test_missing_labels_explains(tmp_path):
    rb.save_synthetic_pilot(tmp_path, n_proofs=40)
    (tmp_path / "judge" / "labels.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="labels"):
        rb.load_pilot(tmp_path)
