"""The static results explorer, on small synthetic runs, whole and partial."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from markov_lean import cli, explore, pilot
from markov_lean.robustness import save_synthetic_pilot

HERE = Path(__file__).parent
FAST_MSM = {"n_microstates": 30, "lag": 1, "n_macrostates": 4, "error_samples": 0,
            "featurizer_kwargs": {"n_components": 16}}


def page_data(page: Path) -> dict:
    html = page.read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    assert m, "data block missing"
    return json.loads(m.group(1))


@pytest.fixture(scope="module")
def judged_run(tmp_path_factory):
    """A pilot folder with an MSM, the judged sample and two noisy judges (steps and whole proofs)."""
    return save_synthetic_pilot(tmp_path_factory.mktemp("judged"), n_proofs=60, seed=0, noise=0.2)


def test_full_run_page(judged_run, tmp_path):
    page = explore.run(judged_run, tmp_path / "site", verbose=False)
    html = page.read_text(encoding="utf-8")
    d = page_data(page)
    m = d["meta"]
    assert m["n_proofs_shown"] == m["n_proofs_total"] == 60
    assert m["area_source"].startswith("consensus of 2 judges")
    assert m["has_text"] and m["has_model"]
    # every proof has its steps in order, with tactic, goal, area and macrostate
    p = d["proofs"][0]
    assert [r[0] for r in p["st"]] == sorted(r[0] for r in p["st"])
    assert all(r[1] and r[2] for r in p["st"])
    assert any(r[3] for r in p["st"]) and all(r[6] is not None for r in p["st"])
    assert p["pl"]  # whole-proof label
    # noisy judges disagree somewhere, and that step shows both codes
    assert any(r[5] and " / " in r[5] for q in d["proofs"] for r in q["st"])
    # model section: matrix rows sum to one, one row per PCCA+ set
    T = np.array(d["model"]["T"])
    assert T.shape == (6, 6) and np.allclose(T.sum(1), 1)
    assert len(d["model"]["sets"]) == 6
    assert abs(sum(s["stationary"] for s in d["model"]["sets"]) - 1) < 1e-6
    assert d["model"]["crosstab"]["areas"]
    # nothing is fetched and no machine paths leak
    assert "http://" not in html and "https://" not in html and "<link" not in html
    assert str(judged_run) not in html and str(tmp_path) not in html
    assert "judge-a" not in html and "judge-b" not in html  # judge ids stay out
    assert d["meta"]["inputs"] == {"msm": "msm", "steps": "judge/sample_steps.parquet",
                                   "labels": "judge/labels.parquet", "pilot_report": None}


def test_patterns_and_related_proofs(judged_run, tmp_path):
    d = page_data(explore.run(judged_run, tmp_path, verbose=False))
    proofs = d["proofs"]
    assert d["patterns"], "the synthetic chain is sticky but does switch"
    for pt in d["patterns"]:
        assert pt["n"] == len(pt["proofs"]) or pt["n"] > len(pt["proofs"])
        assert all(">".join(proofs[i]["ap"]) == pt["key"] for i in pt["proofs"])
    linked = [p for p in proofs if p["ra"]]
    assert linked
    for p in linked:
        for j, n_shared, same in p["ra"]:
            o = proofs[j]
            assert o is not p and n_shared == len(set(p["sa"]) & set(o["sa"])) > 0
            assert same == (o["ap"] == p["ap"])
        # proofs with the same path come before ones that only share a switch
        flags = [same for _, _, same in p["ra"]]
        assert flags == sorted(flags, reverse=True)


def test_pilot_report_numbers_without_paths(tmp_path):
    out = tmp_path / "pilot"
    cfg = {"data": {"source": "synthetic", "synthetic": {"n_proofs": 80, "seed": 4}}, "msm": FAST_MSM,
           "eval": {"gate_nmi": 0.15, "macrostate_sweep": [2, 4]}}
    pilot.run(cfg, out, verbose=False)
    # a synthetic pilot keeps no steps file, so pass the steps it ran on
    from markov_lean.synthetic import generate

    steps, _ = generate(n_proofs=80, seed=4)
    steps.to_parquet(tmp_path / "steps.parquet", index=False)
    d = page_data(explore.run(out, tmp_path / "site", steps=tmp_path / "steps.parquet", verbose=False))
    assert d["meta"]["area_source"] == "hidden true area (synthetic data)"
    assert d["run"]["gate1"]["k"] == 4 and isinstance(d["run"]["gate1"]["passed"], bool)
    assert set(d["run"]["sweep"]) == {"2", "4"}
    assert d["meta"]["inputs"]["steps"] == "steps.parquet"
    assert str(tmp_path) not in json.dumps(d)


def test_model_only_run(judged_run, tmp_path):
    """Before any judging: only msm/ exists, so the page shows macrostates and the model."""
    shutil.copytree(judged_run / "msm", tmp_path / "run" / "msm")
    d = page_data(explore.run(tmp_path / "run", tmp_path / "site", verbose=False))
    assert d["meta"]["area_source"] == "none yet" and not d["meta"]["has_text"]
    assert d["meta"]["n_proofs_shown"] == 60
    assert all(r[3] is None and r[6] is not None for r in d["proofs"][0]["st"])
    assert d["patterns"] == [] and d["model"]["crosstab"] is None
    assert any(p["rm"] for p in d["proofs"])


def test_labels_without_model(judged_run, tmp_path):
    """Judge labels and goal states but no Markov model yet."""
    shutil.copytree(judged_run / "judge", tmp_path / "run" / "judge")
    d = page_data(explore.run(tmp_path / "run", tmp_path / "site", verbose=False))
    assert d["model"] is None and not d["meta"]["has_model"]
    assert all(r[6] is None for p in d["proofs"] for r in p["st"])
    assert d["patterns"]


def test_one_judge_and_partial_labels(judged_run, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(judged_run, run)
    lab = pd.read_parquet(run / "judge" / "labels.parquet")
    lab = lab[(lab["judge"] == "judge-a") & (lab["proof_id"].isin(sorted(lab["proof_id"].unique())[:10]))]
    lab.to_parquet(run / "judge" / "labels.parquet", index=False)
    d = page_data(explore.run(run, tmp_path / "site", verbose=False))
    assert d["meta"]["area_source"] == "one judge"
    labelled = [p for p in d["proofs"] if any(r[3] for r in p["st"])]
    assert len(labelled) == 10
    assert not any(r[5] for p in d["proofs"] for r in p["st"])


def test_size_limits(judged_run, tmp_path):
    big = explore.run(judged_run, tmp_path / "a", verbose=False)
    small = explore.run(judged_run, tmp_path / "b", max_proofs=7, max_state_chars=40, verbose=False)
    d = page_data(small)
    assert d["meta"]["n_proofs_shown"] == 7 and d["meta"]["n_proofs_total"] == 60
    assert all(len(r[2]) <= 42 for p in d["proofs"] for r in p["st"])
    assert d["meta"]["n_truncated_states"] > 0
    assert small.stat().st_size < big.stat().st_size / 3
    # the same seed keeps the same proofs
    again = page_data(explore.run(judged_run, tmp_path / "c", max_proofs=7, max_state_chars=40, verbose=False))
    assert [p["id"] for p in again["proofs"]] == [p["id"] for p in d["proofs"]]


def test_hostile_text_is_escaped(tmp_path):
    """Goal text with HTML or a closing script tag stays data."""
    nasty = '</script><script>alert(1)</script> & <b>x</b>  '
    steps = pd.DataFrame({"proof_id": ["t:a"] * 3, "step": [0, 1, 2], "source": "leandojo", "split": "train",
                          "theorem": ["a"] * 3, "file_path": "Mathlib/X.lean", "tactic": ["simp", nasty, "rfl"],
                          "state_before": [nasty, "⊢ x = x", "⊢ True"], "state_after": "", "true_area": None})
    run = tmp_path / "run"
    (run / "judge").mkdir(parents=True)
    steps.to_parquet(run / "judge" / "sample_steps.parquet", index=False)
    page = explore.run(run, tmp_path / "site", verbose=False)
    html = page.read_text(encoding="utf-8")
    assert html.count("</script>") == 2 and "alert(1)" in html and "<script>alert" not in html
    d = page_data(page)
    assert d["proofs"][0]["st"][0][2] == nasty and d["proofs"][0]["st"][1][1] == nasty


def test_empty_results_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing to show"):
        explore.run(tmp_path, tmp_path / "site", verbose=False)


def test_cli(judged_run, tmp_path, capsys):
    assert cli.main(["explore", "--results", str(judged_run), "--out", str(tmp_path), "--max-proofs", "5",
                     "--title", "Supplement A"]) == 0
    assert "5 of 60 proofs" in capsys.readouterr().out
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "<title>Supplement A</title>" in html


def test_runs_and_switches():
    assert explore.runs(["26", "26", None, "11", "11", "26"]) == ["26", "11", "26"]
    assert explore.switches(["26", "11", "26", "11"]) == ["26>11", "11>26"]
    assert explore.switches([3, 3, None]) == []


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node to run the page's script")
def test_page_script_runs(judged_run, tmp_path):
    """The page's JavaScript loads and every view draws, in Node with a tiny fake DOM."""
    page = explore.run(judged_run, tmp_path, verbose=False)
    r = subprocess.run(["node", str(HERE / "explorer_dom.js"), str(page)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads(r.stdout)
    assert out["errors"] == []
    assert out["items"] == 60 and out["items_after_search"] == 0
    assert out["hash_after_click"].startswith("#view=proofs&p=")
    assert out["model_tables"] == 3 and "PCCA+ sets" in out["model_text"]
    assert out["pattern_links"] > 0 and out["items_after_pattern"] >= 1
    assert out["theme"] == "dark"
