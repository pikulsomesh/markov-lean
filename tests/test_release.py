"""Release tooling on a synthetic pilot run: export, card, checksums and the completeness check."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("pyarrow")

from markov_lean import cli, release, release_card  # noqa: E402
from markov_lean.synthetic import generate  # noqa: E402

JUDGES = ["judge-small", "judge-large"]
FAKE_KEY = "sk-ant-api03-" + "x" * 40


def make_run(root: Path, n_proofs: int = 30, seed: int = 0) -> Path:
    """A pilot run folder shaped like results/pilot3 after both passes, from synthetic data."""
    rng = np.random.default_rng(seed)
    steps, _ = generate(n_proofs=n_proofs, min_steps=8, max_steps=14, seed=seed)
    run = root / "pilot_synth"
    (run / "judge" / "run").mkdir(parents=True)
    (run / "msm").mkdir()
    steps.to_parquet(run / "judge" / "sample_steps.parquet", index=False)

    areas = sorted(steps["true_area"].unique())
    macro = {a: i for i, a in enumerate(areas)}
    assign = steps[["proof_id", "step"]].copy()
    assign["macrostate"] = steps["true_area"].map(macro).astype(int)
    assign["microstate"] = assign["macrostate"] * 3 + rng.integers(0, 3, len(assign))
    assign["membership"] = rng.uniform(0.5, 1.0, len(assign))
    assign = assign[["proof_id", "step", "microstate", "macrostate", "membership"]]
    assign.to_parquet(run / "msm" / "assignments.parquet", index=False)

    rows = []
    for judge, noise in zip(JUDGES, (0.3, 0.1)):
        for r in steps.itertuples(index=False):
            code = r.true_area if rng.random() > noise else str(rng.choice(areas))
            rows.append((r.proof_id, int(r.step), judge, code, float(rng.uniform(0.4, 1.0))))
        for pid, g in steps.groupby("proof_id"):
            rows.append((pid, -1, judge, g["true_area"].mode()[0], 0.9))
    rows.append((steps["proof_id"].iloc[0], 0, "judge-small", "99", 0.5))  # not an MSC 2020 code (replaced below)
    rows.append(("synthetic:not.in.sample", 0, "judge-small", "05", 0.5))
    rows.append((steps["proof_id"].iloc[0], 999, "judge-large", "05", 0.5))
    labels = pd.DataFrame(rows, columns=["proof_id", "step", "judge", "msc", "confidence"])
    labels = labels.drop_duplicates(["proof_id", "step", "judge"], keep="last")
    labels.to_parquet(run / "judge" / "labels.parquet", index=False)

    local = "/Users/someone/markov-lean/results/pilot_synth"
    (run / "pilot_report.json").write_text(json.dumps({
        "data": {"source": "leandojo", "split": "train", "n_proofs": n_proofs, "n_steps": len(steps)},
        "msm": {"n_microstates": 18, "lag": 1, "n_macrostates": 6, "all_checks_passed": False},
        "msm_checks_failed": ["its_converged"],
        "judge_agreement": {"judges": JUDGES, "step": {"n": 300, "agreement": 0.71, "kappa": 0.66}},
        "eval": {"macrostates": {"nmi": 0.42, "ari": 0.3, "n": 300, "coverage": 1.0},
                 "folder_baseline": {"nmi": 0.12, "ari": 0.05, "n": 300, "coverage": 1.0},
                 "gate1": {"passed": True, "rule": "macrostate NMI >= 0.15 and above the folder baseline",
                           "k": 6, "macrostate_nmi": 0.42, "folder_nmi": 0.12, "threshold": 0.15}},
        "labels": f"{local}/judge/labels.parquet",
        "judge": {"sample_steps": f"{local}/judge/sample_steps.parquet",
                  "commands": [f"python -m markov_lean.judge submit --out {local}/judge/run"]},
    }))
    (run / "judge" / "run" / "issues.json").write_text(json.dumps([
        {"custom_id": "judge-small-000001", "proof_id": "x", "judge": "judge-small", "issue": "partial"},
        {"custom_id": "judge-large-000002", "proof_id": "y", "judge": "judge-large", "issue": "refusal"},
    ]))
    return run


@pytest.fixture()
def exported(tmp_path):
    run = make_run(tmp_path)
    out = tmp_path / "release"
    assert release.export(run, out, say=lambda *a: None) == 0
    return run, out


def _rewrite_sums(out: Path) -> None:
    summary = json.loads((out / "run_summary.json").read_text())
    release.write_manifest(out, summary)


# --------------------------------------------------------------------------- export


def test_export_writes_every_file_and_passes_check(exported):
    _, out = exported
    for rel in release.FILES_REQUIRED:
        assert (out / rel).is_file(), rel
    assert release.check(out) == []


def test_tables_match_schema_and_keys(exported):
    _, out = exported
    for t, cols in release.TABLES.items():
        df = pd.read_parquet(out / "data" / f"{t}.parquet")
        assert list(df.columns) == [c for c, *_ in cols]
    steps = pd.read_parquet(out / "data" / "steps.parquet")
    labels = pd.read_parquet(out / "data" / "labels.parquet")
    proofs = pd.read_parquet(out / "data" / "proofs.parquet")
    assert not steps.duplicated(["proof_id", "step"]).any()
    assert set(labels["judge"]) == set(JUDGES) | {"consensus"}
    assert proofs.set_index("proof_id")["n_steps"].sum() == len(steps)
    assert steps["macrostate"].notna().all()
    assert (proofs["n_switches"] >= 0).all()
    assert proofs["mathlib_folder"].notna().all()


def test_bad_labels_are_dropped_and_counted(exported):
    _, out = exported
    labels = pd.read_parquet(out / "data" / "labels.parquet")
    assert "99" not in set(labels["msc"])
    assert "synthetic:not.in.sample" not in set(labels["proof_id"])
    assert (labels["step"] != 999).all()
    s = json.loads((out / "run_summary.json").read_text())
    f = s["label_filtering"]
    assert f["labels_invalid_code"] == 1 and f["labels_outside_sample"] == 1 and f["labels_unknown_step"] == 1


def test_consensus_rows_are_where_both_judges_agree(exported):
    _, out = exported
    lab = pd.read_parquet(out / "data" / "labels.parquet")
    a = lab[lab.judge == JUDGES[0]].set_index(["proof_id", "step"])["msc"]
    b = lab[lab.judge == JUDGES[1]].set_index(["proof_id", "step"])["msc"]
    both = a.index.intersection(b.index)
    agree = {k for k in both if a[k] == b[k]}
    cons = lab[lab.judge == "consensus"]
    assert set(zip(cons.proof_id, cons.step)) == agree
    assert cons.set_index(["proof_id", "step"])["msc"].to_dict() == {k: a[k] for k in agree}


def test_no_paths_keys_or_commands_leak(exported):
    _, out = exported
    for p in out.rglob("*"):
        if p.is_file() and p.suffix != ".parquet" or p.name == "SHA256SUMS":
            text = p.read_text()
            assert "/Users/" not in text and "sk-ant" not in text, p
            assert "custom_id" not in text and "judge-small-000001" not in text, p
    s = json.loads((out / "run_summary.json").read_text())
    assert "labels" not in s and "judge" not in s  # the raw report's path-valued keys are not copied
    assert s["judge_issues"] == {"partial": 1, "refusal": 1}
    assert s["eval"]["macrostates"]["nmi"] == pytest.approx(0.42)


def test_sha256sums_verify_with_the_system_tool(exported):
    _, out = exported
    tool = shutil.which("sha256sum")
    if tool is None:
        pytest.skip("sha256sum not installed")
    r = subprocess.run([tool, "-c", "SHA256SUMS"], cwd=out, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_manifest_lists_rows_and_hashes(exported):
    _, out = exported
    m = json.loads((out / "MANIFEST.json").read_text())
    by = {e["path"]: e for e in m["files"]}
    assert by["data/steps.parquet"]["rows"] == len(pd.read_parquet(out / "data" / "steps.parquet"))
    for rel, e in by.items():
        assert e["sha256"] == hashlib.sha256((out / rel).read_bytes()).hexdigest()


def test_no_text_export_nulls_the_text(tmp_path):
    run = make_run(tmp_path)
    out = tmp_path / "rel"
    assert release.export(run, out, include_text=False, say=lambda *a: None) == 0
    steps = pd.read_parquet(out / "data" / "steps.parquet")
    assert steps[["tactic", "state_before", "state_after"]].isna().all().all()
    assert "--no-text" in (out / "README.md").read_text()


def test_export_refuses_a_key_in_the_data(tmp_path):
    run = make_run(tmp_path)
    p = run / "judge" / "sample_steps.parquet"
    s = pd.read_parquet(p)
    s.loc[3, "state_before"] = f"h : ANTHROPIC = {FAKE_KEY}"
    s.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="Anthropic API key"):
        release.export(run, tmp_path / "rel", say=lambda *a: None)


def test_export_strips_local_checkout_paths(tmp_path):
    run = make_run(tmp_path)
    p = run / "judge" / "sample_steps.parquet"
    s = pd.read_parquet(p)
    s["file_path"] = "/Users/someone/mathlib4/" + s["file_path"]
    s.to_parquet(p, index=False)
    out = tmp_path / "rel"
    assert release.export(run, out, say=lambda *a: None) == 0
    fp = pd.read_parquet(out / "data" / "steps.parquet")["file_path"]
    assert fp.str.startswith("Mathlib/").all()


def test_export_needs_labels_and_an_empty_folder(tmp_path):
    run = make_run(tmp_path)
    (run / "judge" / "labels.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="labels"):
        release.export(run, tmp_path / "rel", say=lambda *a: None)
    run2 = make_run(tmp_path / "b")
    out = tmp_path / "busy"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        release.export(run2, out, say=lambda *a: None)
    assert release.export(run2, out, force=True, say=lambda *a: None) == 0
    assert not (out / "keep.txt").exists()


def test_labels_from_the_run_subfolder_and_without_assignments(tmp_path):
    run = make_run(tmp_path)
    (run / "judge" / "labels.parquet").rename(run / "judge" / "run" / "labels.parquet")
    (run / "msm" / "assignments.parquet").unlink()
    out = tmp_path / "rel"
    assert release.export(run, out, say=lambda *a: None) == 0
    steps = pd.read_parquet(out / "data" / "steps.parquet")
    assert steps["macrostate"].isna().all()
    assert pd.read_parquet(out / "data" / "proofs.parquet")["n_switches"].isna().all()


# --------------------------------------------------------------------------- check


def test_check_catches_a_changed_file(exported):
    _, out = exported
    with open(out / "SCHEMA.md", "a") as f:
        f.write("\nedited\n")
    problems = release.check(out)
    assert any("checksum mismatch for SCHEMA.md" in p for p in problems)


def test_check_catches_missing_and_extra_files(exported):
    _, out = exported
    (out / "data" / "proofs.parquet").unlink()
    (out / "data" / "extra.parquet").write_bytes(b"x")
    problems = release.check(out)
    assert "missing data/proofs.parquet" in problems
    assert any("extra.parquet is not in SHA256SUMS" in p for p in problems)
    assert any("unexpected file data/extra.parquet" in p for p in problems)


def test_check_catches_a_key_even_with_fresh_checksums(exported):
    _, out = exported
    p = out / "data" / "labels.parquet"
    df = pd.read_parquet(p)
    df.loc[0, "judge"] = FAKE_KEY
    df.to_parquet(p, index=False)
    _rewrite_sums(out)
    assert any("Anthropic API key" in x for x in release.check(out))


def test_check_catches_a_local_path_in_a_text_file(exported):
    _, out = exported
    with open(out / "README.md", "a") as f:
        f.write("\nbuilt in /home/someone/work\n")
    _rewrite_sums(out)
    assert any("README.md contains a home directory path" in x for x in release.check(out))


def test_check_catches_schema_and_key_problems(exported):
    _, out = exported
    p = out / "data" / "labels.parquet"
    df = pd.read_parquet(p)
    df = pd.concat([df, df.iloc[[0]]])
    df.loc[df.index[-1], "confidence"] = 2.0
    df.to_parquet(p, index=False)
    s = pd.read_parquet(out / "data" / "steps.parquet").drop(columns=["membership"])
    s.to_parquet(out / "data" / "steps.parquet", index=False)
    _rewrite_sums(out)
    problems = release.check(out)
    assert any("steps.parquet columns" in x for x in problems)


def test_check_catches_duplicate_labels_and_bad_confidence(exported):
    _, out = exported
    p = out / "data" / "labels.parquet"
    df = pd.read_parquet(p)
    dup = df.iloc[[0]].copy()
    dup["confidence"] = 2.0
    pd.concat([df, dup], ignore_index=True).to_parquet(p, index=False)
    _rewrite_sums(out)
    problems = release.check(out)
    assert any("repeated (proof_id, step, judge)" in x for x in problems)
    assert any("confidence outside 0 to 1" in x for x in problems)


def test_check_catches_a_broken_card(exported):
    _, out = exported
    text = (out / "README.md").read_text()
    (out / "README.md").write_text(text.replace("pretty_name:", "pretty_nam:", 1) + "\nTODO: licence\n")
    _rewrite_sums(out)
    problems = release.check(out)
    assert "README.md YAML header has no pretty_name" in problems
    assert any("TODO" in x for x in problems)


def test_check_on_a_missing_folder(tmp_path):
    assert release.check(tmp_path / "nope") == [f"{tmp_path / 'nope'} is not a folder"]


# --------------------------------------------------------------------------- card


def test_card_header_follows_the_hub_spec(exported):
    _, out = exported
    text = (out / "README.md").read_text()
    meta = yaml.safe_load(text.split("---\n")[1])
    for k in release.CARD_KEYS_REQUIRED:
        assert k in meta
    assert meta["task_categories"] == ["text-classification"]
    assert meta["size_categories"] == ["n<1K"]
    paths = [f["path"] for c in meta["configs"] for f in c["data_files"]]
    assert paths == ["data/steps.parquet", "data/labels.parquet", "data/proofs.parquet"]
    assert "CC BY 2.0" in text and "Apache-2.0" in text and "yang2023leandojo" in text
    assert "judge-small" in text and "its_converged" in text


def test_card_ships_no_msc_titles(exported):
    from markov_lean import msc

    _, out = exported
    text = (out / "README.md").read_text() + (out / "SCHEMA.md").read_text()
    assert msc.MSC2020_TOP["26"] not in text


@pytest.mark.parametrize("n,cat", [(5, "n<1K"), (5000, "1K<n<10K"), (128_559, "100K<n<1M"), (2_000_000, "1M<n<10M")])
def test_size_category(n, cat):
    assert release_card.size_category(n) == cat


# --------------------------------------------------------------------------- helpers and CLI


@pytest.mark.parametrize("text,kind", [
    (FAKE_KEY, "Anthropic API key"),
    ("AKIA" + "A" * 16, "AWS access key id"),
    ("export ANTHROPIC_API_KEY=abcdefgh1234", "key assignment"),
    ("/Users/alex/repo", "home directory path"),
    ("/home/ubuntu/markov-lean", "home directory path"),
    ("C:\\Users\\alex\\x", "Windows user path"),
    ("~/markov-lean", "home shorthand path"),
])
def test_find_leaks(text, kind):
    assert kind in release.find_leaks(text)


@pytest.mark.parametrize("text", [
    "Mathlib/Algebra/Group/Basic.lean", "⊢ ∀ (x : ℝ), 0 ≤ x ^ 2", "h : a / b = c", "s.card ≤ 2 ^ n",
    "Finset.sum_range_succ", "no goals",
])
def test_find_leaks_ignores_lean_text(text):
    assert release.find_leaks(text) == []


def test_relative_file_path():
    assert release.relative_file_path("Mathlib/A/B.lean") == "Mathlib/A/B.lean"
    assert release.relative_file_path("/Users/x/.lake/packages/mathlib/Mathlib/A/B.lean") == "Mathlib/A/B.lean"
    assert release.relative_file_path("/home/x/other/File.lean") == ""
    assert release.relative_file_path(None) == ""


def test_scrub_drops_paths_and_commands():
    out = release.scrub({"a": "/Users/x/y", "b": 1, "commands": ["ls"], "c": {"d": "~/z", "e": "ok"},
                         "f": ["/tmp/q", "fine"]})
    assert out == {"b": 1, "c": {"e": "ok"}, "f": ["fine"]}


def test_cli_export_and_check(tmp_path, capsys):
    run = make_run(tmp_path)
    out = tmp_path / "rel"
    assert cli.main(["export-dataset", "--run", str(run), "--out", str(out)]) == 0
    assert cli.main(["check-release", str(out)]) == 0
    assert "OK" in capsys.readouterr().out
    (out / "SCHEMA.md").write_text("changed")
    assert cli.main(["check-release", str(out)]) == 1
    assert cli.main(["export-dataset", "--run", str(tmp_path / "missing"), "--out", str(tmp_path / "o2")]) == 2
