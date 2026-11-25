"""scripts/reproduce.sh and scripts/paper_assets.py on a small synthetic pilot run (no network, no API)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reproduce.sh"
STAGES = "setup data synthetic pilot judge score full search report paper release".split()

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def sh(*args: str, env: dict | None = None, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    e = {**os.environ, "PY": sys.executable, **(env or {})}
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=cwd, env=e, capture_output=True, text=True,
                          timeout=600)


def test_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_dry_run_lists_every_stage_and_runs_nothing(tmp_path):
    r = sh("--dry-run", "--pilot", str(tmp_path / "p"), "--release", str(tmp_path / "rel"))
    assert r.returncode == 0, r.stderr
    for stage in ("setup", "data", "synthetic", "pilot pass 1", "judge labels", "pilot pass 2", "full fit",
                  "proof search", "search report", "paper", "dataset release"):
        assert f"== {stage}" in r.stdout, stage
    assert "+ bash scripts/fetch_data.sh" in r.stdout
    assert "export-dataset" in r.stdout and "check-release" in r.stdout
    assert not (tmp_path / "p").exists() and not (tmp_path / "rel").exists()
    assert "judge submit" not in r.stdout  # no API call without --run-judges


def test_dry_run_with_judges_prints_estimate_then_submit(tmp_path):
    r = sh("--dry-run", "--stage", "judge", "--run-judges", "--pilot", str(tmp_path / "p"))
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert out.index("judge.yaml estimate") < out.index("judge.yaml submit") < out.index("judge.yaml collect")
    assert "--max-dollars 85" in out


def test_from_and_unknown_stages():
    r = sh("--dry-run", "--from", "paper")
    assert r.returncode == 0
    assert "== paper" in r.stdout and "== full fit" not in r.stdout and "== dataset release" in r.stdout
    assert sh("--stage", "nope").returncode == 2
    assert sh("--from", "nope").returncode == 2
    assert sh("--bogus").returncode == 2


def test_judge_stage_refuses_without_a_key(tmp_path):
    pilot = tmp_path / "p"
    (pilot / "judge").mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    r = subprocess.run(["bash", str(SCRIPT), "--stage", "judge", "--run-judges", "--pilot", str(pilot)],
                       cwd=ROOT, env={**env, "PY": sys.executable, "ANTHROPIC_API_KEY": ""},
                       capture_output=True, text=True, timeout=120, input="")
    # the estimate fails first (no sample in this empty run) or the key check stops it; either way nothing is sent
    assert r.returncode != 0
    assert not (pilot / "judge" / "run" / "batches.json").exists()


# --------------------------------------------------------------------------- end to end on synthetic data


@pytest.fixture(scope="module")
def synthetic_pilot(tmp_path_factory):
    """A pilot run after pass 1 with judge labels in place: real MSM on synthetic data, fake judges."""
    pytest.importorskip("deeptime")
    pytest.importorskip("pyarrow")
    from markov_lean.synthetic import generate

    root = tmp_path_factory.mktemp("repro")
    out = root / "pilot"
    steps, _ = generate(n_proofs=120, min_steps=8, max_steps=20, seed=1)
    processed = root / "processed"
    processed.mkdir()
    steps.assign(source="synth", true_area=None).to_parquet(processed / "synth_train.parquet", index=False)
    cfg = {"data": {"source": "synth", "split": "train", "processed_dir": str(processed), "min_steps": 8,
                    "seed": 0},
           "msm": {"featurizer": "tfidf_svd", "featurizer_kwargs": {"n_components": 32}, "n_microstates": 25,
                   "lag": 1, "n_macrostates": 6, "error_samples": 0},
           "judge": {"config": str(ROOT / "configs" / "judge.yaml"), "sample_proofs": 1000},
           "eval": {"topk": [1], "macrostate_sweep": [], "gate_nmi": 0.15}}
    config = root / "pilot.yaml"
    config.write_text(yaml.safe_dump(cfg))
    r = sh("--stage", "pilot", "--pilot", str(out), env={"PILOT_CONFIG": str(config)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "judge" / "sample_steps.parquet").exists()
    assert "eval" not in json.loads((out / "pilot_report.json").read_text())
    again = sh("--stage", "pilot", "--pilot", str(out), env={"PILOT_CONFIG": str(config)})
    assert "already done" in again.stdout
    rng = np.random.default_rng(0)
    codes = sorted(steps["true_area"].unique())
    rows = []
    for judge, noise in (("judge-a", 0.2), ("judge-b", 0.1)):
        for r in steps.itertuples(index=False):
            c = r.true_area if rng.random() > noise else str(rng.choice(codes))
            rows.append((r.proof_id, int(r.step), judge, c, 0.8))
        for pid, g in steps.groupby("proof_id"):
            rows.append((pid, -1, judge, g["true_area"].mode()[0], 0.9))
    labels = pd.DataFrame(rows, columns=["proof_id", "step", "judge", "msc", "confidence"])
    path = root / "labels_from_elsewhere.parquet"
    labels.to_parquet(path, index=False)
    syn = root / "pilot_synthetic"
    syn.mkdir()
    (syn / "pilot_report.json").write_text(json.dumps({
        "data": {"n_proofs": 400, "n_steps": 7790}, "eval": {"macrostates": {"nmi": 0.863, "ari": 0.896}}}))
    return root, out, path, syn, config


def test_reproduce_from_cached_labels_to_release(synthetic_pilot):
    pytest.importorskip("matplotlib")
    root, pilot_dir, labels, _, config = synthetic_pilot
    rel, paper = root / "release", root / "paper_out"
    r = sh("--stage", "judge", "--stage", "score", "--stage", "paper", "--stage", "release",
           "--labels", str(labels), "--pilot", str(pilot_dir), "--release", str(rel),
           env={"PAPER_OUT": str(paper), "SEARCH_OUT": str(root / "no_search"), "PILOT_CONFIG": str(config)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert (pilot_dir / "judge" / "labels.parquet").exists()
    rep = json.loads((pilot_dir / "pilot_report.json").read_text())
    assert "eval" in rep and rep["eval"]["macrostates"]["nmi"] > 0.15
    assert f"{rel}: OK" in r.stdout
    for name in ("implied_timescales", "chapman_kolmogorov", "transition_matrix", "judge_confusion", "area_flow"):
        assert (paper / "figures" / f"{name}.pdf").stat().st_size > 1000, name
    assert (paper / "tables" / "areas.tex").exists() and (paper / "tables" / "data.tex").exists()
    assert "skipped tables/search.tex" in r.stdout and "skipped figures/search_ablation.pdf" in r.stdout


def test_paper_assets_tables(synthetic_pilot):
    pytest.importorskip("sklearn")
    root, pilot_dir, labels, syn, _ = synthetic_pilot
    if not (pilot_dir / "judge" / "labels.parquet").exists():
        shutil.copyfile(labels, pilot_dir / "judge" / "labels.parquet")
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import paper_assets
    finally:
        sys.path.pop(0)
    out = root / "tables_only"
    search = root / "search_report"
    search.mkdir(exist_ok=True)
    (search / "pass_at_k.tex").write_text("% pass@k\n")
    built, skipped = paper_assets.build(pilot_dir, pilot_dir / "msm", syn, search, root / "no_data", out,
                                        figures=False)
    assert set(built) == {"tables/data.tex", "tables/areas.tex", "tables/search.tex"}
    assert skipped == []
    areas = (out / "tables" / "areas.tex").read_text()
    assert "Macrostates ($m = 6$)" in areas and "0.863" in areas and "Random partition" in areas
    data = (out / "tables" / "data.tex").read_text()
    assert "LeanDojo Benchmark 4" in data and "120" in data and r"\tbd" in data  # no processed files here


def test_release_check_via_make(synthetic_pilot, tmp_path):
    if shutil.which("make") is None:
        pytest.skip("make not installed")
    root, pilot_dir, labels, _, _ = synthetic_pilot
    if not (pilot_dir / "judge" / "labels.parquet").exists():
        shutil.copyfile(labels, pilot_dir / "judge" / "labels.parquet")
    rel = tmp_path / "rel"
    env = {**os.environ, "PY": sys.executable}
    r = subprocess.run(["make", "release", f"RUN={pilot_dir}", f"RELEASE={rel}", f"PY={sys.executable}"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run(["make", "release-check", f"RELEASE={rel}", f"PY={sys.executable}"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr
    (rel / "data" / "steps.parquet").unlink()
    r = subprocess.run(["make", "release-check", f"RELEASE={rel}", f"PY={sys.executable}"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "missing data/steps.parquet" in r.stdout
