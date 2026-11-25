"""Full-scale run and cross-dataset projection, on small synthetic data so CI stays fast."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from markov_lean import project as pj

# --------------------------------------------------------------------------- statistics (numpy and pandas only)


def _placed():
    # proof a: 0 0 1 1 0 (two switches in four moves), proof b: 2 2 2, proof c: one statement
    return pd.DataFrame({
        "proof_id": ["a"] * 5 + ["b"] * 3 + ["c"],
        "step": [0, 1, 2, 3, 4, 0, 1, 2, -1],
        "microstate": 0,
        "macrostate": [0, 0, 1, 1, 0, 2, 2, 2, 1],
        "membership": 1.0,
    })


def test_proof_tables_by_hand():
    t = pj.proof_tables(_placed(), n_macro=3, lag=1)
    assert t.n_steps.tolist() == [5, 3, 1]
    assert t.n_moves.tolist() == [4, 2, 0]
    assert t.n_switches.tolist() == [2, 0, 0]
    np.testing.assert_array_equal(t.transitions.sum(0), [[1, 1, 0], [1, 1, 0], [0, 0, 2]])
    runs = t.runs
    assert runs["dwell"].tolist() == [2, 2, 1, 3, 1]
    assert runs["left_censored"].tolist() == [True, False, False, True, True]
    assert runs["right_censored"].tolist() == [False, False, True, True, True]

    s = pj.statistics(t)
    assert s["pooled_switch_rate"] == pytest.approx(2 / 6)
    assert s["mean_switch_rate"] == pytest.approx(0.25)          # (0.5 + 0) / 2 proofs with moves
    assert s["mean_switches_per_proof"] == pytest.approx(1.0)    # statements are left out
    assert s["mean_dwell"] == pytest.approx(8 / 4)               # runs of proofs a and b
    assert s["mean_dwell_uncensored"] == pytest.approx(2.0)      # the one run inside proof a
    np.testing.assert_allclose(s["occupancy"], [3 / 9, 3 / 9, 3 / 9])
    np.testing.assert_allclose(s["stay_probability"], [0.5, 0.5, 1.0])


def test_lag_two_skips_one_step():
    t = pj.proof_tables(_placed(), n_macro=3, lag=2)
    assert t.n_moves.tolist() == [3, 1, 0]
    # proof a is 0 0 1 1 0, so the lag-2 pairs are 0->1, 0->1, 1->0
    np.testing.assert_array_equal(t.transitions[0], [[0, 2, 0], [1, 0, 0], [0, 0, 0]])
    assert t.n_switches.tolist() == [3, 0, 0]


def test_bootstrap_is_seeded_and_brackets_the_estimate():
    rng = np.random.default_rng(3)
    rows = []
    for p in range(60):
        n = int(rng.integers(3, 15))
        m = rng.integers(0, 4, size=n)
        rows += [{"proof_id": f"p{p}", "step": s, "microstate": 0, "macrostate": int(x), "membership": 1.0}
                 for s, x in enumerate(m)]
    placed = pd.DataFrame(rows)
    a = pj.analyze("d", placed, 4, 1, n_boot=300, seed=5)
    b = pj.analyze("d", placed, 4, 1, n_boot=300, seed=5)
    np.testing.assert_array_equal(a.draws["pooled_switch_rate"], b.draws["pooled_switch_rate"])
    row = pj.summary_row(a)
    for k in pj.SCALARS:
        assert row[f"{k}_lo"] <= row[k] + 1e-12 <= row[f"{k}_hi"] + 2e-12
    tr = pj.transition_rows(a)
    assert np.allclose(tr.groupby("from_macrostate")["probability"].sum(), 1.0)
    assert (tr["lo"] <= tr["probability"] + 1e-12).all() and (tr["probability"] <= tr["hi"] + 1e-12).all()


def test_statement_only_dataset_has_occupancy_but_no_moves():
    placed = pd.DataFrame({"proof_id": [f"t{i}" for i in range(10)], "step": -1, "microstate": 0,
                           "macrostate": [0, 1, 1, 2, 2, 2, 0, 1, 2, 2], "membership": 1.0})
    r = pj.analyze("stmts", placed, 3, 1, n_boot=50, seed=0)
    assert not r.has_transitions
    assert np.isnan(r.point["pooled_switch_rate"]) and np.isnan(r.point["mean_dwell"])
    np.testing.assert_allclose(r.point["occupancy"], [0.2, 0.3, 0.5])
    occ = pj.occupancy_rows(r)
    assert (occ["lo"] <= occ["share"]).all() and (occ["share"] <= occ["hi"]).all()


def test_differences_against_reference():
    placed = _placed()
    ref = pj.analyze("ref", placed, 3, 1, n_boot=100, seed=0)
    other = pj.analyze("other", placed[placed["proof_id"] == "a"], 3, 1, n_boot=100, seed=1)
    d = pj.differences([ref, other], "ref")
    row = d[d["statistic"] == "pooled_switch_rate"].iloc[0]
    assert row["dataset"] == "other" and row["difference"] == pytest.approx(0.5 - 2 / 6)
    assert row["lo"] <= row["difference"] <= row["hi"]
    assert pj.differences([ref, other], "missing").empty


def test_nearest_center_and_assign():
    centers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], np.float32)
    X = np.array([[0.9, 0.1], [0.1, 0.8], [-2.0, 0.3], [0.2, 0.1]], np.float32)
    assert pj.nearest_center(X, centers, chunk=3).tolist() == [0, 1, 2, 0]
    mem = np.array([[0.9, 0.1], [0.2, 0.8], [np.nan, np.nan]])
    model = pj.FittedModel(None, centers, np.array([0, 1, -1]), mem, None, 1, 2)
    micro, macro, m = pj.assign(X, model)
    assert macro.tolist() == [0, 1, -1, 0]
    assert m[0] == pytest.approx(0.9) and m[1] == pytest.approx(0.8) and np.isnan(m[2])


# --------------------------------------------------------------------------- featurizing in parallel


def test_parallel_transform_matches_serial():
    from markov_lean.featurize import make_featurizer
    from markov_lean.full import ParallelTransform
    from markov_lean.synthetic import generate

    steps, _ = generate(n_proofs=40, seed=4)
    texts = steps["state_before"].tolist()
    feat = make_featurizer("tfidf_svd", n_components=16)
    feat.fit_transform(texts)
    serial = feat.transform(texts)
    with ParallelTransform(feat, n_jobs=2, chunk_rows=37) as pt:
        par = pt(texts)
    np.testing.assert_allclose(par, serial, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- the model on precomputed features


FAST_MSM = {"n_microstates": 20, "lag": 1, "n_macrostates": 4, "error_samples": 0,
            "featurizer_kwargs": {"n_components": 16}}


def test_msm_run_accepts_precomputed_features():
    pytest.importorskip("deeptime")
    from markov_lean.featurize import make_featurizer
    from markov_lean.msm import run, sort_steps
    from markov_lean.pilot import msm_config

    steps, _ = generate_steps(120, 6)
    cfg = msm_config(FAST_MSM)
    a = run(steps, cfg)
    feat = make_featurizer("tfidf_svd", n_components=16, seed=cfg.seed)
    X = feat.fit_transform(sort_steps(steps)["state_before"].tolist())
    b = run(steps[["proof_id", "step"]].assign(state_before=""), cfg, features=X, featurizer=feat)
    pd.testing.assert_frame_equal(a.assignments, b.assignments)
    assert b.featurizer is feat
    with pytest.raises(ValueError, match="rows"):
        run(steps, cfg, features=X[:-1], featurizer=feat)


def generate_steps(n_proofs, seed):
    from markov_lean.synthetic import generate

    return generate(n_proofs=n_proofs, seed=seed)


# --------------------------------------------------------------------------- end to end


@pytest.fixture
def processed(tmp_path):
    pytest.importorskip("pyarrow")
    from markov_lean.data import write_steps

    d = tmp_path / "processed"
    train, _ = generate_steps(240, 11)
    write_steps(train, d, "leandojo", "train")
    test, _ = generate_steps(60, 12)
    write_steps(test, d, "leandojo", "test")
    first = test[test["step"] == 0].head(25).copy()
    first["step"] = -1
    first["proof_id"] = first["proof_id"].str.replace("synthetic:", "minif2f:")
    write_steps(first, d, "minif2f", "test")
    return d, train


def _config(processed_dir, tmp_path, n_jobs=1):
    return {
        "data": {"processed_dir": str(processed_dir), "train": "leandojo_train", "min_steps": 8, "seed": 0},
        "features": {"fit_proofs": 100, "batch_rows": 700, "chunk_rows": 250, "n_jobs": n_jobs},
        "msm": FAST_MSM,
        "project": {
            "bootstrap": 60, "seed": 0, "reference": "leandojo_test",
            "datasets": [
                {"name": "leandojo_test", "path": str(processed_dir / "leandojo_test.parquet"), "min_steps": 8},
                {"name": "leandojo_train_short", "path": str(processed_dir / "leandojo_train.parquet"),
                 "min_steps": 2, "max_steps": 7},
                {"name": "minif2f_test", "path": str(processed_dir / "minif2f_test.parquet")},
                {"name": "minif2f_search", "path": str(tmp_path / "no" / "steps.parquet"), "min_steps": 2},
            ],
        },
    }


def test_full_run_end_to_end(processed, tmp_path):
    pytest.importorskip("deeptime")
    from markov_lean import full
    from markov_lean.prior_loader import load_msm_artifacts

    d, train = processed
    out = tmp_path / "out"
    assert full.run(_config(d, tmp_path, n_jobs=2), out, verbose=False) == 0

    sizes = train.groupby("proof_id").size()
    rep = json.loads((out / "full_report.json").read_text())
    assert rep["data"]["n_proofs"] == int((sizes >= 8).sum())
    assert rep["data"]["n_steps"] == int(sizes[sizes >= 8].sum())
    assert rep["msm"]["n_macrostates"] == 4 and rep["msm"]["lag"] == 1
    assert [s["dataset"] for s in rep["skipped"]] == ["minif2f_search"]
    assert rep["peak_memory_gb"] > 0

    # the saved model is what the search's area prior loads
    art = load_msm_artifacts(out / "msm")
    assert art.memberships.shape == (20, 4)

    # the streamed features are the featurizer's output, row for row in sorted order
    X = np.load(out / "cache" / "features.npy")
    keys = full.select_keys(train, 8)
    texts = keys.merge(train[["proof_id", "step", "state_before"]], on=["proof_id", "step"])["state_before"]
    np.testing.assert_allclose(X, art.featurizer.transform(texts.tolist()), rtol=1e-5, atol=1e-6)

    summary = json.loads((out / "project" / "summary.json").read_text())
    ds = summary["datasets"]
    assert set(ds) == {"leandojo_train", "leandojo_test", "leandojo_train_short", "minif2f_test"}
    assert ds["leandojo_train"]["has_transitions"] and ds["leandojo_test"]["has_transitions"]
    assert not ds["minif2f_test"]["has_transitions"] and ds["minif2f_test"]["pooled_switch_rate"] is None
    assert ds["leandojo_train_short"]["median_proof_length"] <= 7
    for name in ("leandojo_train", "leandojo_test"):
        e = ds[name]
        assert e["pooled_switch_rate_lo"] <= e["pooled_switch_rate"] <= e["pooled_switch_rate_hi"]
        rows = np.nansum(np.array(e["transition_matrix"], dtype=float), 1)
        assert np.allclose(rows[rows > 0], 1)
    assert ds["minif2f_test"]["n_proofs"] == 25 and np.isclose(sum(ds["minif2f_test"]["occupancy"]), 1)

    for f in ("summary.parquet", "transitions.parquet", "occupancy.parquet", "dwell_hist.parquet",
              "differences.parquet", "leandojo_test/placed.parquet", "leandojo_test/proof_stats.parquet",
              "leandojo_test/dwell_runs.parquet"):
        assert (out / "project" / f).exists(), f
    diff = pd.read_parquet(out / "project" / "differences.parquet")
    assert set(diff["dataset"]) == {"leandojo_train", "leandojo_train_short"}
    assert "leandojo_test" in (out / "RUN.md").read_text()
    if rep["figures"]:  # the figures extra is installed in CI
        names = {p.split("/")[-1] for p in rep["figures"]}
        assert {"implied_timescales.pdf", "switch_rates.pdf", "occupancy.pdf", "dwell_times.pdf"} <= names

    # a rerun reuses the cached features and gives the same model
    before = pd.read_parquet(out / "msm" / "assignments.parquet")
    assert full.run(_config(d, tmp_path), out, verbose=False) == 0
    rep2 = json.loads((out / "full_report.json").read_text())
    assert rep2["features"]["reused"] is True
    pd.testing.assert_frame_equal(before, pd.read_parquet(out / "msm" / "assignments.parquet"))


def test_project_command_on_a_saved_model(processed, tmp_path):
    pytest.importorskip("deeptime")
    from markov_lean import cli, full

    d, _ = processed
    cfg = _config(d, tmp_path)
    out = tmp_path / "out"
    full.run(cfg, out, skip_projection=True, verbose=False)
    assert not (out / "project").exists()

    # a search run's steps.parquet, given on the command line
    from markov_lean.data import write_steps

    found, _ = generate_steps(15, 21)
    write_steps(found, tmp_path / "search", "found", "x")
    cfg_path = tmp_path / "full.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    code = cli.main(["project", "--config", str(cfg_path), "--msm", str(out / "msm"), "--out",
                     str(tmp_path / "proj"), "--dataset", f"search={tmp_path / 'search' / 'found_x.parquet'}"])
    assert code == 0
    ds = json.loads((tmp_path / "proj" / "summary.json").read_text())["datasets"]
    assert {"leandojo_train", "leandojo_test", "search"} <= set(ds)
    assert ds["search"]["n_proofs"] == 15 and ds["search"]["has_transitions"]


def test_project_command_takes_paths_relative_to_where_it_runs(processed, tmp_path, monkeypatch):
    """`markov-lean project --msm results/full/msm` from a folder other than the repo root
    used to look in <repo>/results/full/msm; --msm and --dataset now follow the working directory."""
    pytest.importorskip("deeptime")
    from markov_lean import cli, full
    from markov_lean.data import write_steps

    d, _ = processed
    cfg = _config(d, tmp_path)
    full.run(cfg, tmp_path / "results" / "full", skip_projection=True, verbose=False)
    found, _ = generate_steps(15, 21)
    write_steps(found, tmp_path / "search", "found", "x")
    cfg_path = tmp_path / "full.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    code = cli.main(["project", "--config", str(cfg_path), "--msm", "results/full/msm", "--out", "proj",
                     "--dataset", "search=search/found_x.parquet"])
    assert code == 0
    report = json.loads((tmp_path / "proj" / "project_report.json").read_text())
    assert report["msm_dir"] == str(tmp_path / "results" / "full" / "msm")
    assert "search" in report["projected"]


def test_cli_path_falls_back_to_the_repo_root(tmp_path, monkeypatch):
    from markov_lean import full

    monkeypatch.chdir(tmp_path)
    assert full._cli_path("configs") == full.ROOT / "configs"          # only in the repo
    (tmp_path / "configs").mkdir()
    assert full._cli_path("configs") == tmp_path / "configs"           # here wins
    assert full._cli_path("nowhere/msm") == tmp_path / "nowhere" / "msm"  # neither: say where it was looked for


def test_cli_lists_full_and_project(capsys):
    from markov_lean import cli

    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "full" in out and "project" in out


def test_glob_reads_every_search_run(tmp_path):
    pytest.importorskip("pyarrow")
    from markov_lean import full

    found, _ = generate_steps(6, 31)
    for cond in ("lm", "area_l0.3"):
        d = tmp_path / "search_eval" / "minif2f" / cond / "seed0"
        d.mkdir(parents=True)
        found.to_parquet(d / "steps.parquet", index=False)
    spec = {"name": "s", "path": str(tmp_path / "search_eval" / "minif2f" / "*" / "*" / "steps.parquet"),
            "min_steps": 2}
    df = full.load_dataset(spec)
    assert df["proof_id"].nunique() == 12
    assert df["proof_id"].str.startswith(("lm/seed0/", "area_l0.3/seed0/")).all()
    assert full.load_dataset({**spec, "path": str(tmp_path / "none" / "*" / "steps.parquet")}) is None
