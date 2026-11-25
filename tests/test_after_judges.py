"""scripts/after_judges.sh end to end on a synthetic pilot, with a fake Batches client (no network, no API)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "after_judges.sh"
TESTS = Path(__file__).resolve().parent

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def sh(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    e = {**base, "PY": sys.executable, "PYTHONPATH": os.pathsep.join([str(TESTS), base.get("PYTHONPATH", "")]),
         **(env or {})}
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, env=e, capture_output=True, text=True,
                          timeout=900)


def test_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_read_only_client_blocks_everything_but_status_and_results(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(TESTS))
    try:
        import collect_batches as cb
        import fake_batches
    finally:
        sys.path[:2] = []
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"results": []}))
    monkeypatch.setenv("FAKE_BATCHES", str(spec))
    c = cb.ReadOnlyClient(fake_batches.client())
    assert c.messages.batches.retrieve("b1").processing_status == "ended"
    assert list(c.messages.batches.results("b1")) == []
    for bad in (lambda: c.messages.batches.create(requests=[]), lambda: c.messages.batches.cancel("b1"),
                lambda: c.messages.create(model="x"), lambda: c.beta):
        with pytest.raises(cb.SpendBlocked):
            bad()


@pytest.fixture(scope="module")
def submitted(tmp_path_factory):
    """A pilot after pass 1, with batches 'submitted' for two judges and a fake client that has their results."""
    pytest.importorskip("deeptime")
    pytest.importorskip("pyarrow")
    from markov_lean.synthetic import generate

    root = tmp_path_factory.mktemp("after")
    pilot = root / "pilot"
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
    r = subprocess.run([sys.executable, "-m", "markov_lean.cli", "pilot", "--config", str(config), "--out",
                        str(pilot)], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (pilot / "judge" / "sample_steps.parquet").exists()

    # what `judge submit` leaves behind: manifest.json and batches.json, two batches (one per judge)
    rng = np.random.default_rng(0)
    codes = sorted(steps["true_area"].unique())
    manifest, results = {}, []
    for b, (judge, noise) in enumerate((("claude-haiku-5-5", 0.2), ("claude-sonnet-5-5", 0.1))):
        for pid, g in steps.groupby("proof_id"):
            cid = f"{judge}-{len(manifest)}"
            manifest[cid] = {"proof_id": str(pid), "judge": judge, "steps": [int(s) for s in g["step"]]}
            lab = [{"step": int(s.step), "msc": s.true_area if rng.random() > noise else str(rng.choice(codes)),
                    "confidence": 0.8} for s in g.itertuples(index=False)]
            results.append({"batch": f"batch_{b}", "custom_id": cid, "proof_msc": g["true_area"].mode()[0],
                            "steps": lab})
    run = pilot / "judge" / "run"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "batches.json").write_text(json.dumps({"batch_ids": ["batch_0", "batch_1"], "estimated_dollars": 1.0}))
    spec = root / "fake_batches.json"
    spec.write_text(json.dumps({"status": "ended", "results": results}))

    # a local judge's pilot labels (third judge) and gold-set labels for two judges
    local = pilot / "judge" / "local" / "qwen2.5_7b-instruct"
    local.mkdir(parents=True)
    rows = [(r.proof_id, int(r.step), "ollama/qwen2.5:7b-instruct",
             r.true_area if rng.random() > 0.4 else str(rng.choice(codes)), 0.6)
            for r in steps.itertuples(index=False)]
    pd.DataFrame(rows, columns=["proof_id", "step", "judge", "msc", "confidence"]).to_parquet(
        local / "labels.parquet", index=False)
    gold = pd.read_csv(ROOT / "data" / "gold" / "msc_gold.csv")
    strict = gold[gold.status == "strict"].head(40)
    gold_dir = root / "gold"
    for sub, judge in (("local/qwen2.5_7b-instruct", "ollama/qwen2.5:7b-instruct"), ("api", "claude-haiku-5-5")):
        (gold_dir / sub).mkdir(parents=True)
        pd.DataFrame({"proof_id": strict.id, "step": -1, "judge": judge, "msc": strict.msc.astype(str),
                      "confidence": 0.9}).to_parquet(gold_dir / sub / "labels.parquet", index=False)

    env = {"PILOT_CONFIG": str(config), "PAPER_OUT": str(root / "paper"), "RESULTS": str(root),
           "GOLD": str(gold_dir), "ZIP": str(root / "out.zip"), "FAKE_BATCHES": str(spec),
           "FAKE_BATCHES_CALLS": str(root / "calls.txt"),
           "AFTER_JUDGES_CLIENT_FACTORY": "fake_batches:client"}
    return root, pilot, env, spec


def test_dry_run_lists_every_step_and_runs_nothing(submitted):
    root, pilot, env, _ = submitted
    r = sh("--dry-run", "--pilot", str(pilot), env={**env, "AFTER_JUDGES_CLIENT_FACTORY": ""})
    assert r.returncode == 0, r.stdout + r.stderr
    for cmd in ("collect_batches.py --out", "cli pilot --config", "cli judge-agreement", "cli switches",
                "cli significance", "--extra-labels", "cli bridges", "cli cases", "cli judge-accuracy",
                "cli paper --results", "--accuracy"):
        assert cmd in r.stdout, cmd
    assert "judge submit" not in r.stdout and "estimate" not in r.stdout
    assert not (pilot / "judge" / "run" / "labels.parquet").exists()
    assert not (root / "out.zip").exists()


def test_stops_without_a_key_or_without_batches(submitted, tmp_path):
    _, pilot, env, _ = submitted
    r = sh("--pilot", str(pilot), env={**env, "AFTER_JUDGES_CLIENT_FACTORY": ""})
    assert r.returncode != 0 and "ANTHROPIC_API_KEY is not set" in r.stderr
    empty = tmp_path / "p"
    shutil.copytree(pilot, empty, ignore=shutil.ignore_patterns("batches.json"))
    r = sh("--pilot", str(empty), env=env)
    assert r.returncode != 0 and "batches.json is missing" in r.stderr
    r = sh("--pilot", str(tmp_path / "nothing"), env=env)
    assert r.returncode != 0 and "sample_steps.parquet is missing" in r.stderr


def test_stops_when_a_batch_has_not_ended(submitted, tmp_path):
    root, pilot, env, spec = submitted
    running = tmp_path / "running.json"
    running.write_text(json.dumps({**json.loads(spec.read_text()), "status": "in_progress"}))
    p = tmp_path / "p"
    shutil.copytree(pilot, p)
    r = sh("--pilot", str(p), env={**env, "FAKE_BATCHES": str(running), "ZIP": str(tmp_path / "z.zip")})
    assert r.returncode != 0 and "have not ended yet" in r.stdout + r.stderr
    assert not (p / "judge" / "run" / "labels.parquet").exists()
    assert not (p / "judge" / "labels.parquet").exists()


def test_end_to_end_then_resume(submitted):
    pytest.importorskip("matplotlib")
    root, pilot, env, _ = submitted
    r = sh("--pilot", str(pilot), env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    calls = set((root / "calls.txt").read_text().split())
    assert calls == {"retrieve", "results"}, calls  # nothing that could spend

    assert (pilot / "judge" / "labels.parquet").read_bytes() == (pilot / "judge" / "run" / "labels.parquet").read_bytes()
    rep = json.loads((pilot / "pilot_report.json").read_text())
    assert rep["eval"]["gate1"]["passed"]
    agree = json.loads((pilot / "agreement" / "judge_agreement.json").read_text())
    assert "ollama/qwen2.5:7b-instruct" in json.dumps(agree)
    assert (pilot / "switches" / "switches.json").exists()
    assert (pilot / "significance" / "significance.json").exists()
    assert (pilot / "bridges" / "summary.json").exists()
    assert (root / "paper" / "figures" / "case_studies_table.tex").exists()
    assert (root / "gold" / "accuracy" / "judge_accuracy.json").exists()
    macros = (root / "paper" / "figures" / "results_macros.tex").read_text()
    assert r"\newcommand{\resGoldAccHaiku}" in macros and r"\newcommand{\resMacroNMI}" in macros
    assert "GATE 1 PASSED" in r.stdout
    assert f"Drag {root / 'out.zip'} into the Claude Duet chat" in r.stdout

    names = set(zipfile.ZipFile(root / "out.zip").namelist())
    rel = lambda p: Path(p).as_posix().lstrip("/")  # noqa: E731
    for want in (pilot / "judge/run/labels.parquet", pilot / "judge/run/issues.json", pilot / "pilot_report.json",
                 pilot / "agreement/judge_agreement.json", pilot / "switches/switches.json",
                 pilot / "significance/significance.json", root / "paper/figures/results_macros.tex",
                 root / "gold/accuracy/judge_accuracy.json", pilot / ".after_judges/after_judges.log"):
        assert rel(want) in names, want
    assert not any("manifest.json" in n or "batches.json" in n or "data/raw" in n for n in names)

    # rerun: nothing is downloaded again and finished steps are skipped
    (root / "calls.txt").unlink()
    again = sh("--pilot", str(pilot), env=env)
    assert again.returncode == 0, again.stdout + again.stderr
    assert "already collected" in again.stdout
    assert again.stdout.count("already done on these labels") >= 6
    assert not (root / "calls.txt").exists()
    assert "GATE 1 PASSED" in again.stdout
