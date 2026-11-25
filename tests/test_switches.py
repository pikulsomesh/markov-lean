"""Within-proof area-switch test (src/markov_lean/switches.py) on synthetic data with known switching."""

import json

import numpy as np
import pandas as pd
import pytest

from markov_lean import cli, pilot
from markov_lean import switches as sw
from markov_lean.synthetic import generate

FAST = {"n_perm": 200, "n_boot": 300}


def _areas(seqs: dict[str, list], folder: dict[str, str] | None = None) -> pd.DataFrame:
    rows = [{"proof_id": p, "step": i, "area": str(a)} for p, s in seqs.items() for i, a in enumerate(s)]
    return pd.DataFrame(rows)


def _seq(seqs):
    return sw.Sequences(_areas(seqs))


def _overall(r):
    return sw._overall(r)["statistics"]


def _iid(n_proofs, K, rng, lo=6, hi=30):
    return {f"p{i:04d}": list(rng.integers(0, K, int(rng.integers(lo, hi + 1)))) for i in range(n_proofs)}


def _sticky(n_proofs, K, p_stay, rng, lo=6, hi=30):
    out = {}
    for i in range(n_proofs):
        a = int(rng.integers(K))
        s = []
        for _ in range(int(rng.integers(lo, hi + 1))):
            s.append(a)
            if rng.random() > p_stay:
                a = int((a + rng.integers(1, K)) % K)
        out[f"p{i:04d}"] = s
    return out


# --------------------------------------------------------------------------- statistics


def test_per_proof_statistics_by_hand():
    seq = _seq({"a": [0, 0, 1, 1, 1, 0], "b": [2, 2, 2], "c": [3, 1]})
    st = {k: v[0] for k, v in sw.per_proof_stats(seq, seq.y[None]).items()}
    assert st["n_switches"].tolist() == [2, 0, 1]
    assert st["has_switch"].tolist() == [1, 0, 1]
    assert st["switch_rate"] == pytest.approx([2 / 5, 0, 1])
    assert st["mean_dwell"] == pytest.approx([6 / 3, 3, 1])
    assert st["off_mode"] == pytest.approx([1 - 3 / 6, 0, 1 / 2])
    assert sorted(sw.run_lengths(seq, seq.y).tolist()) == [1, 1, 1, 2, 3, 3]


def test_short_proofs_are_dropped_and_steps_ordered():
    a = _areas({"a": ["11"], "b": ["05", "11", "05"]}).sample(frac=1, random_state=0)
    seq = sw.Sequences(a)
    assert seq.n_dropped == 1 and seq.proofs == ["b"]
    assert [seq.areas[i] for i in seq.y] == ["05", "11", "05"]


def test_shuffle_null_matches_its_exact_expectation():
    rng = np.random.default_rng(0)
    seq = _seq(_sticky(120, 4, 0.8, rng))
    Y = sw.simulate_shuffle(seq, 400, rng)
    # composition is kept exactly
    for r in range(3):
        assert (seq.counts(Y[r]) == seq.counts()).all()
    sim = sw.per_proof_stats(seq, Y)["switch_rate"].mean(0)
    assert sim.mean() == pytest.approx(sw.expected_shuffle_rate(seq).mean(), abs=0.01)


def test_pooled_null_keeps_folder_composition():
    rng = np.random.default_rng(1)
    seq = _seq(_sticky(60, 3, 0.9, rng))
    block = np.arange(seq.P) % 2
    Y = sw.simulate_pooled(seq, block, 5, rng)
    for b in (0, 1):
        pos = block[seq.pidx] == b
        for r in range(5):
            assert sorted(Y[r, pos]) == sorted(seq.y[pos])
    assert (sw.per_proof_stats(seq, Y)["n_switches"].sum(0) > 0).any()


def test_noise_null_rate():
    rng = np.random.default_rng(2)
    seq = _seq({f"p{i}": [i % 5] * 40 for i in range(200)})
    Y = sw.simulate_noise(seq, 0.2, 20, rng)
    assert (Y != seq.y[None]).mean() == pytest.approx(0.2, abs=0.01)


def test_markov_null_recovers_the_chain():
    rng = np.random.default_rng(3)
    seq = _seq(_sticky(400, 3, 0.7, rng, lo=20, hi=40))
    g = np.zeros(seq.P, np.int64)
    T = sw.fit_markov(seq, g)
    assert np.diagonal(T[0]) == pytest.approx([0.7] * 3, abs=0.03)
    Y = sw.simulate_markov(seq, T, g, 50, rng)
    assert (Y[:, seq.start] == seq.y[seq.start]).all()
    rate = sw.per_proof_stats(seq, Y)["switch_rate"].mean()
    assert rate == pytest.approx(0.3, abs=0.02)


# --------------------------------------------------------------------------- the nulls on data with known switching


def test_sticky_segments_beat_the_shuffle():
    """Areas in runs: fewer switches and longer dwell than a within-proof shuffle."""
    seq = _seq(_sticky(300, 4, 0.85, np.random.default_rng(4)))
    r = sw.analyse_source(seq, np.full(seq.P, None, object), ["shuffle"], seed=0, **FAST)
    st = _overall(r)
    d = st["switch_rate"]["nulls"]["shuffle"]
    assert d["difference"] < -0.2 and d["difference_ci95"][1] < 0 and d["p_less"] < 0.01
    assert st["mean_dwell"]["nulls"]["shuffle"]["p_greater"] < 0.01
    assert st["off_mode"]["nulls"]["shuffle"]["fixed_by_null"]
    assert st["has_switch"]["nulls"]["shuffle"]["fixed_by_null"]


def test_iid_labels_do_not_beat_the_shuffle():
    seq = _seq(_iid(300, 4, np.random.default_rng(5)))
    r = sw.analyse_source(seq, np.full(seq.P, None, object), ["shuffle", "markov"], seed=1, **FAST)
    st = _overall(r)
    for n in ("shuffle", "markov"):
        d = st["switch_rate"]["nulls"][n]
        assert d["p_two_sided"] > 0.05
        assert d["difference_ci95"][0] < 0 < d["difference_ci95"][1]


@pytest.mark.parametrize("seed", [1, 2])
def test_markov_null_is_calibrated_on_a_markov_chain(seed):
    """The generator is a first-order chain, so the default Markov null should not reject it."""
    steps, _ = generate(n_proofs=300, p_stay=0.85, seed=seed)
    seq = sw.Sequences(steps.rename(columns={"true_area": "area"}))
    r = sw.analyse_source(seq, sw._folders(seq, steps), ["markov"], seed=seed, **FAST)
    for s in sw.STATS:
        assert _overall(r)[s]["nulls"]["markov"]["p_two_sided"] > 0.01, s


def test_mode_grouping_is_biased():
    """Grouping the chain by each proof's main area conditions on the outcome (documented, not the default)."""
    steps, _ = generate(n_proofs=400, p_stay=0.85, seed=1)
    seq = sw.Sequences(steps.rename(columns={"true_area": "area"}))
    r = sw.analyse_source(seq, sw._folders(seq, steps), ["markov"], seed=0, markov_group="mode", **FAST)
    assert _overall(r)["off_mode"]["nulls"]["markov"]["p_two_sided"] < 0.05


def test_markov_null_sees_switching_concentrated_in_some_proofs():
    """Half the proofs never move and half move a lot: a single chain spreads switches too evenly."""
    rng = np.random.default_rng(6)
    stay = {f"s{i:04d}": [int(rng.integers(4))] * int(rng.integers(8, 25)) for i in range(250)}
    move = {f"m{k}": v for k, v in _sticky(250, 4, 0.6, rng, 8, 25).items()}
    seq = _seq({**stay, **move})
    r = sw.analyse_source(seq, np.full(seq.P, None, object), ["markov"], seed=2, **FAST)
    d = _overall(r)["has_switch"]["nulls"]["markov"]
    assert d["difference"] < -0.2 and d["p_less"] < 0.01


def test_noise_floor_separates_movement_from_flicker():
    rng = np.random.default_rng(7)
    eps = 0.1
    flicker = {}
    for i in range(300):
        home = int(rng.integers(5))
        s = [home if rng.random() >= eps else int((home + rng.integers(1, 5)) % 5) for _ in range(rng.integers(8, 30))]
        flicker[f"p{i:04d}"] = s
    seq = _seq(flicker)
    r = sw.analyse_source(seq, np.full(seq.P, None, object), ["noise"], noise_rate=eps, seed=3, **FAST)
    assert _overall(r)["off_mode"]["nulls"]["noise"]["p_two_sided"] > 0.01

    # real two-segment proofs: far more of the proof spent off the main area than noise gives
    seg = {f"p{i:04d}": [i % 5] * 10 + [(i + 1) % 5] * 6 for i in range(300)}
    seq = _seq(seg)
    r = sw.analyse_source(seq, np.full(seq.P, None, object), ["noise", "shuffle"], noise_rate=eps, seed=4, **FAST)
    st = _overall(r)
    assert st["off_mode"]["nulls"]["noise"]["p_greater"] < 0.01
    assert st["switch_rate"]["nulls"]["shuffle"]["p_less"] < 0.01


def test_noise_null_is_skipped_without_a_rate():
    seq = _seq(_iid(50, 3, np.random.default_rng(8)))
    r = sw.analyse_source(seq, np.full(seq.P, None, object), list(sw.NULLS), n_perm=20, n_boot=20)
    assert "noise" not in r["nulls"]


def test_noise_rate_from_two_judges():
    raw = pd.DataFrame({"proof_id": ["a"] * 8, "step": list(range(4)) * 2, "judge": ["j1"] * 4 + ["j2"] * 4,
                        "msc": ["11", "11", "05", "11", "11", "11", "11", "11"], "confidence": 0.9})
    eps, note = sw.estimate_noise(raw)
    assert eps == pytest.approx(1 - np.sqrt(0.75))
    assert "j1" in note
    assert sw.estimate_noise(raw[raw.judge == "j1"])[0] is None


def test_strata_and_corrections():
    rng = np.random.default_rng(9)
    seq = _seq(_sticky(200, 3, 0.85, rng, 2, 40))
    folder = np.where(np.arange(seq.P) % 3 == 0, "Topology", np.where(np.arange(seq.P) % 3 == 1, "Analysis", None))
    r = sw.analyse_source(seq, folder.astype(object), ["shuffle"], min_proofs=30, **FAST)
    kinds = {(e["kind"], e["stratum"]): e for e in r["strata"]}
    assert ("folder", "Topology") in kinds and ("folder", "Analysis") in kinds and ("folder", "other") in kinds
    lens = [e for e in r["strata"] if e["kind"] == "length"]
    assert sum(e["n_proofs"] for e in lens) == seq.P
    for e in lens:
        d = e["statistics"]["switch_rate"]["nulls"]["shuffle"]
        assert d["holm"] >= d["p_two_sided"] - 1e-12 and d["bh"] <= d["holm"] + 1e-12


def test_length_bins():
    assert sw.length_bin(np.array([2, 3, 4, 7, 8, 31, 32, 400])).tolist() == \
        ["2-3", "2-3", "4-7", "4-7", "8-15", "16-31", "32+", "32+"]


# --------------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("pilot_switch")
    cfg = {"data": {"source": "synthetic", "synthetic": {"n_proofs": 160, "seed": 1}},
           "msm": {"n_microstates": 40, "lag": 1, "bayesian_samples": 0, "featurizer_kwargs": {"n_components": 32},
                   "n_macrostates": 4},
           "eval": {"gate_nmi": 0.15, "macrostate_sweep": [2, 3, 4]}}
    pilot.run(cfg, out, verbose=False)
    (out / "config.yaml").write_text(json.dumps(cfg))
    return out


def test_cli_end_to_end(synthetic_run, tmp_path):
    out = tmp_path / "sw"
    rc = cli.main(["switches", "--run", str(synthetic_run), "--config", str(synthetic_run / "config.yaml"),
                   "--out", str(out), "--n-perm", "60", "--n-boot", "100", "--noise-rate", "0.1",
                   "--min-proofs", "10"])
    assert rc == 0
    for f in ("switches.json", "switches.csv", "proofs.csv", "dwell.csv", "REPORT.md", "table_switches.tex",
              "switches.pdf", "switches.png"):
        assert (out / f).exists(), f
    rep = json.loads((out / "switches.json").read_text())
    assert set(rep["sources"]) == {"true_area", "macrostate"}
    # the planted areas are sticky, so they come in runs
    h = rep["headline"]["true_area"]
    assert h["switch_rate"] < h["switch_rate_shuffle"] and h["p_switch_rate_shuffle"] < 0.05
    assert "folder" in {e["kind"] for e in rep["sources"]["true_area"]["strata"]}
    tex = (out / "table_switches.tex").read_text()
    assert r"\begin{tabular}" in tex and "Macrostate" in tex and "True area" in tex
    long = pd.read_csv(out / "switches.csv")
    assert {"source", "kind", "stratum", "statistic", "null", "p_two_sided", "holm"} <= set(long.columns)


def test_judge_labels_are_used_when_present(synthetic_run, tmp_path):
    steps = pilot.sample_proofs(pilot.load_steps(json.loads((synthetic_run / "config.yaml").read_text())["data"]),
                                None, 2, 0)
    lab = steps[["proof_id", "step", "true_area"]].rename(columns={"true_area": "msc"})
    raw = pd.concat([lab.assign(judge="a", confidence=0.9), lab.assign(judge="b", confidence=0.8)])
    run = tmp_path / "run"
    (run / "judge").mkdir(parents=True)
    raw.to_parquet(run / "judge" / "labels.parquet")
    steps.to_parquet(run / "judge" / "sample_steps.parquet")
    rep = sw.run(run, tmp_path / "out", sources=["judge"], n_perm=30, n_boot=30, figures=False, verbose=False)
    assert list(rep["sources"]) == ["judge"]
    assert rep["noise_rate_source"]["judge"].startswith("1 - sqrt")
    assert rep["sources"]["judge"]["noise_rate"] == pytest.approx(0.0)


def test_missing_inputs_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        sw.run(tmp_path, verbose=False)
