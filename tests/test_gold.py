"""Tests for the gold MSC set (gold.py) and the judge-accuracy report (judge_accuracy.py).

No network: the source files are tiny fixtures written here, and the local judge is the mock
Ollama server from test_ollama_judge.py.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from markov_lean import cli, msc
from markov_lean import gold as G
from markov_lean import judge_accuracy as A

# ---------------------------------------------------------------- rules


@pytest.mark.parametrize("name,status,code", [
    ("mathd_numbertheory_101", "strict", "11"),
    ("numbertheory_2pownm1prime_nprime", "strict", "11"),
    ("mathd_algebra_48", "excluded", ""),
    ("algebra_amgm_sum1toneqn", "excluded", ""),
    ("induction_sum_odd", "excluded", ""),
    ("amc12a_2019_p21", "excluded", ""),
    ("aime_1983_p1", "excluded", ""),
    ("imo_1959_p1", "excluded", ""),
])
def test_minif2f_rule(name, status, code):
    r = G.rule_minif2f(name)
    assert r.status == status and (r.strict or "") == code


def test_putnam_rule_single_multi_and_excluded():
    assert G.rule_putnam(["number_theory"]).strict == "11"
    assert G.rule_putnam(["geometry"]).status == "family"
    assert set(G.rule_putnam(["geometry"]).codes) == {"51", "52"}
    assert G.rule_putnam(["analysis"]).codes == G.ANALYSIS_BLOCK
    # two strict tags that disagree become a family of both codes
    r = G.rule_putnam(["combinatorics", "number_theory"])
    assert r.status == "family" and set(r.codes) == {"05", "11"}
    # any tag with no MSC class excludes the problem
    assert G.rule_putnam(["algebra"]).status == "excluded"
    assert G.rule_putnam(["algebra", "number_theory"]).status == "excluded"
    assert G.rule_putnam([]).status == "excluded"
    assert G.rule_putnam(["topology"]).status == "excluded"


@pytest.mark.parametrize("book,ex,status,code", [
    ("Axler", "exercise_5_4", "strict", "15"),
    ("Ireland-Rosen", "exercise_3_5", "strict", "11"),
    ("Shakarchi", "exercise_1_13a", "strict", "30"),
    ("Munkres", "exercise_13_1", "strict", "54"),
    ("Munkres", "exercise_5_1", "excluded", ""),
    ("Dummit-Foote", "exercise_1_1_2", "strict", "20"),
    ("Dummit-Foote", "exercise_7_1_2", "family", ""),
    ("Dummit-Foote", "exercise_13_1_2", "strict", "12"),
    ("Herstein", "exercise_2_1_18", "strict", "20"),
    ("Herstein", "exercise_5_1_8", "strict", "12"),
    ("Artin", "exercise_2_2_9", "strict", "20"),
    ("Artin", "exercise_10_2_4", "excluded", ""),
    ("Rudin", "exercise_3_3", "strict", "40"),
    ("Rudin", "exercise_4_2a", "family", ""),
    ("Pugh", "exercise_2_12a", "strict", "54"),
    ("Putnam", "exercise_2018_a5", "excluded", ""),
])
def test_proofnet_rule(book, ex, status, code):
    r = G.rule_proofnet(book, ex)
    assert r.status == status and (r.strict or "") == code


def test_every_table_code_is_msc2020():
    rules = [r for _, r in G.MINIF2F_RULES] + list(G.PUTNAM_TAGS.values())
    rules += [r for rows in G.PROOFNET_BOOKS.values() for _, _, r in rows]
    for r in rules:
        assert all(msc.normalize(c) == c for c in r.codes), r.rule
    with pytest.raises(ValueError):
        G.Rule("bad", "36")  # a pre-2000 class


# ---------------------------------------------------------------- building from sources


def _jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def raw(tmp_path):
    r = tmp_path / "raw"
    r.mkdir()
    _jsonl(r / "minif2f.jsonl", [
        {"name": "mathd_numbertheory_1", "split": "valid", "goal": "⊢ 7 ∣ 21", "formal_statement": "x"},
        {"name": "mathd_algebra_2", "split": "test", "goal": "⊢ 1 + 1 = 2"},
        {"name": "amc12a_2020_p3", "split": "test", "goal": "⊢ True"},
    ])
    # exercise_1_1 appears in two books; the informal statement decides which Lean row is whose
    _jsonl(r / "proofnet.jsonl", [
        {"name": "exercise_1_1", "split": "test", "informal_prefix": "/-- Prove that G is abelian.-/",
         "goal": "G : Type\n⊢ abelian"},
        {"name": "exercise_1_1", "split": "test", "informal_prefix": "/-- Show that the series converges.-/",
         "goal": "⊢ Summable f"},
        {"name": "exercise_13_1", "split": "valid", "informal_prefix": "/-- Open sets.-/", "goal": "⊢ IsOpen U"},
    ])
    pn = r / "ProofNet" / "benchmark"
    pn.mkdir(parents=True)
    _jsonl(pn / "test.jsonl", [
        {"id": "Rudin|exercise_1_1", "nl_statement": "Show that the series converges."},
        {"id": "Herstein|exercise_1_1", "nl_statement": "Prove that $G$ is abelian."},
    ])
    _jsonl(pn / "valid.jsonl", [{"id": "Munkres|exercise_13_1", "nl_statement": "Open sets."}])
    pb = r / "PutnamBench"
    (pb / "informal").mkdir(parents=True)
    (pb / "lean4" / "src").mkdir(parents=True)
    (pb / "informal" / "putnam.json").write_text(json.dumps([
        {"problem_name": "putnam_2000_a1", "tags": ["number_theory"]},
        {"problem_name": "putnam_2000_a2", "tags": ["analysis"]},
        {"problem_name": "putnam_2000_a3", "tags": ["algebra"]},
        {"problem_name": "putnam_2000_a4", "tags": ["combinatorics"]},
    ]))
    (pb / "lean4" / "src" / "putnam_2000_a1.lean").write_text(
        "import Mathlib\n\nabbrev putnam_2000_a1_solution : ℕ := 3\n\n/-- doc -/\n"
        "theorem putnam_2000_a1\n  (n : ℕ) :\n  let m := n + 1\n  Odd (2 * m + 1) :=\nsorry\n")
    (pb / "lean4" / "src" / "putnam_2000_a2.lean").write_text(
        "theorem putnam_2000_a2 (f : ℝ → ℝ) : Continuous f := by\n  sorry\n")
    (pb / "lean4" / "src" / "putnam_2000_a3.lean").write_text("theorem putnam_2000_a3 : True := sorry\n")
    # putnam_2000_a4 has no Lean file
    return r


def test_build_from_sources(raw):
    g = G.build(raw)
    assert list(g.columns) == G.GOLD_COLUMNS
    by = g.set_index("id")
    assert by.loc["minif2f:mathd_numbertheory_1", "msc"] == "11"
    assert by.loc["minif2f:mathd_numbertheory_1", "split"] == "val"
    assert by.loc["minif2f:mathd_algebra_2", "status"] == "excluded"
    # the repeated ProofNet name is matched on the informal statement, not on file order
    assert by.loc["proofnet:Rudin|exercise_1_1", "statement"] == "⊢ Summable f"
    assert by.loc["proofnet:Herstein|exercise_1_1", "status"] == "excluded"  # Herstein ch. 1 not listed
    assert by.loc["proofnet:Munkres|exercise_13_1", "msc"] == "54"
    # a `let x := ...` inside the statement does not cut it short
    stmt = by.loc["putnam:putnam_2000_a1", "statement"]
    assert stmt.startswith("theorem putnam_2000_a1") and stmt.endswith("Odd (2 * m + 1)")
    assert by.loc["putnam:putnam_2000_a2", "status"] == "family"
    assert by.loc["putnam:putnam_2000_a3", "status"] == "excluded"
    assert by.loc["putnam:putnam_2000_a4", "reason"] == "no Lean 4 statement in lean4/src"
    assert G.build(raw).equals(g)  # deterministic


def test_cli_build_summary_and_steps(raw, tmp_path, capsys):
    out = tmp_path / "gold" / "g.csv"
    assert cli.main(["gold-set", "build", "--raw", str(raw), "--out", str(out)]) == 0
    assert (tmp_path / "gold" / "g_balance.json").exists()
    capsys.readouterr()
    assert cli.main(["gold-set", "summary", "--gold", str(out), "--json"]) == 0
    bal = json.loads(capsys.readouterr().out)
    assert bal["strict_classes"]["11"]["n"] == 2
    steps_out = tmp_path / "s.parquet"
    assert cli.main(["gold-set", "steps", "--gold", str(out), "--out", str(steps_out), "--strict-only"]) == 0
    s = pd.read_parquet(steps_out)
    assert set(s.proof_id) == set(G.read_gold(out).query("status == 'strict'").id)
    assert (s.step == 0).all() and (s.tactic == "sorry").all()


# ---------------------------------------------------------------- the committed gold set


def test_committed_gold_set_is_consistent():
    g = G.read_gold()
    assert g.id.is_unique
    assert set(g.status) <= {"strict", "family", "excluded"}
    strict, fam, exc = (g[g.status == s] for s in ("strict", "family", "excluded"))
    assert all(msc.normalize(c) == c for c in strict.msc)
    assert (strict.msc == strict.msc_family).all()
    for codes in fam.msc_family:
        assert len(codes.split(";")) >= 2 and all(msc.normalize(c) == c for c in codes.split(";"))
    assert (exc.msc == "").all() and (exc.reason != "").all()
    assert (g[g.status != "excluded"].statement != "").all()
    # every row's rule is the table's answer for its own metadata
    for r in g.itertuples():
        if r.source == "minif2f":
            rule = G.rule_minif2f(r.name)
        elif r.source == "proofnet":
            rule = G.rule_proofnet(*r.name.split("|", 1))
        else:
            rule = G.rule_putnam(r.source_label.removeprefix("tags:").split(",") if r.source_label != "tags:" else [])
        if r.status != "excluded" or rule.status == "excluded":
            assert (rule.strict or "") == r.msc and ";".join(rule.codes) == r.msc_family, r.id
    bal = G.balance(g)
    assert bal["n_problems"] == len(g)
    assert sum(v["n"] for v in bal["strict_classes"].values()) == len(strict)
    assert len(strict) >= 400 and len(bal["strict_classes"]) >= 8


def test_steps_per_class_and_proofs(tmp_path):
    g = G.read_gold()
    s = G.to_steps(g, statuses=("strict",), per_class=5, seed=1)
    counts = g[g.id.isin(s.proof_id)].msc.value_counts()
    assert counts.max() <= 5 and set(counts.index) == set(g[g.status == "strict"].msc)
    assert G.to_steps(g, statuses=("strict",), per_class=5, seed=1).equals(s)
    z = tmp_path / "sol.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("test/mathd_numbertheory_3.lean",
                   "import Mathlib\n/-- doc -/\ntheorem mathd_numbertheory_3 : (∑ x in Finset.range 10, (x + 1) ^ 2) % 10 = 5"
                   " := by\n  decide\n")
    proofs = G.read_minif2f_proofs(z)
    assert proofs == {"mathd_numbertheory_3": "  decide"}
    s2 = G.to_steps(g, statuses=("strict",), proofs=proofs)
    row = s2[s2.proof_id == "minif2f:mathd_numbertheory_3"].iloc[0]
    assert row.tactic == "  decide" and row.state_after == "no goals"
    assert (s2[s2.proof_id != "minif2f:mathd_numbertheory_3"].tactic == "sorry").all()


# ---------------------------------------------------------------- scoring


def _gold():
    rows = []
    for i in range(30):
        rows.append(dict(id=f"t:a{i}", source="t", status="strict", msc="11", msc_family="11", rule="r11"))
    for i in range(10):
        rows.append(dict(id=f"t:b{i}", source="t", status="strict", msc="20", msc_family="20", rule="r20"))
    for i in range(10):
        rows.append(dict(id=f"u:c{i}", source="u", status="family", msc="", msc_family="26;40", rule="ra"))
    rows.append(dict(id="t:x", source="t", status="excluded", msc="", msc_family="", rule="rx"))
    g = pd.DataFrame(rows)
    for c in G.GOLD_COLUMNS:
        if c not in g:
            g[c] = ""
    return g[G.GOLD_COLUMNS]


def _labels(gold, judge, fn, step=-1):
    rows = [(r.id, step, judge, fn(r), 0.9) for r in gold.itertuples()]
    return pd.DataFrame(rows, columns=["proof_id", "step", "judge", "msc", "confidence"])


def test_perfect_and_majority_judges():
    g = _gold()
    truth = lambda r: r.msc or (r.msc_family.split(";")[-1] or "11")
    lab = pd.concat([_labels(g, "perfect", truth), _labels(g, "majority", lambda r: "11")])
    rep = A.report(lab, g, n_boot=200)
    p, m = rep["judges"]["perfect"], rep["judges"]["majority"]
    assert p["strict"]["accuracy"] == 1.0 and p["strict"]["macro_f1"] == 1.0 and p["family"]["accuracy"] == 1.0
    assert p["coverage"] == 1.0 and p["n_gold"] == 50  # the excluded row is never scored
    assert m["strict"]["accuracy"] == pytest.approx(0.75)
    assert m["strict"]["accuracy"] == pytest.approx(m["strict"]["majority_baseline_accuracy"])
    assert m["strict"]["balanced_accuracy"] == pytest.approx(0.5)
    # F1(11) = 2*0.75*1/(1.75); F1(20) = 0
    assert m["strict"]["macro_f1"] == pytest.approx((2 * 0.75 / 1.75) / 2)
    assert m["family"]["family_only_accuracy"] == 0.0
    cm = m["strict"]["confusion"]
    assert cm["rows"] == ["11", "20"] and cm["cols"] == ["11", "20", "other"]
    assert cm["counts"] == [[30, 0, 0], [10, 0, 0]]
    lo, hi = m["strict"]["accuracy_ci"]
    assert lo <= 0.75 <= hi


def test_other_codes_missing_labels_and_step_fallback():
    g = _gold()
    lab = _labels(g, "j", lambda r: "26" if r.id.startswith("t:b") else (r.msc or "40"))
    lab = lab[~lab.proof_id.isin(["t:a0", "t:a1"])]                       # two problems not labelled
    lab.loc[lab.proof_id == "t:a2", "msc"] = "not a code"                  # invalid answer
    lab.loc[lab.proof_id == "t:a3", "step"] = 0                            # only a step label
    rep = A.report(lab, g, n_boot=50)["judges"]["j"]
    assert rep["n_labelled"] == 48 and rep["coverage"] == pytest.approx(48 / 50)
    assert rep["n_from_step_label"] == 1 and rep["n_invalid_code"] == 1
    s = rep["strict"]
    assert s["n"] == 38
    assert s["accuracy"] == pytest.approx(27 / 38)
    assert s["other_codes"] == {"26": 10, "invalid": 1}
    assert s["confusion"]["counts"][1] == [0, 0, 10]
    assert rep["family"]["accuracy"] == pytest.approx((27 + 10) / 48)


def test_labels_for_unknown_ids_are_an_error():
    g = _gold()
    lab = pd.DataFrame([("leandojo:x", -1, "j", "11", 0.5)], columns=["proof_id", "step", "judge", "msc", "confidence"])
    with pytest.raises(ValueError, match="no judge labels match"):
        A.report(lab, g)


def test_cli_writes_report_and_figure(tmp_path, capsys):
    g = _gold()
    gp = tmp_path / "gold.csv"
    g.to_csv(gp, index=False)
    rng = np.random.default_rng(0)
    noisy = lambda r: (r.msc or "26") if rng.random() < 0.8 else "05"
    lp = tmp_path / "labels.parquet"
    pd.concat([_labels(g, "ollama/qwen2.5:7b-instruct", noisy), _labels(g, "claude-haiku", lambda r: "11")]).to_parquet(lp)
    out = tmp_path / "acc"
    assert cli.main(["judge-accuracy", "--gold", str(gp), "--labels", str(lp), "--out", str(out),
                     "--n-boot", "100"]) == 0
    assert "| ollama/qwen2.5:7b-instruct |" in capsys.readouterr().out
    rep = json.loads((out / "judge_accuracy.json").read_text())
    assert set(rep["judges"]) == {"ollama/qwen2.5:7b-instruct", "claude-haiku"}
    assert (out / "confusion_ollama_qwen2.5_7b-instruct.csv").exists()
    pytest.importorskip("seaborn")
    assert (out / "figures" / "judge_accuracy.pdf").stat().st_size > 1000


def test_local_judge_on_gold_steps_end_to_end(tmp_path):
    """gold-set steps -> local-judge label (mock Ollama) -> judge-accuracy."""
    from test_ollama_judge import MockOllama

    steps_path = tmp_path / "gold_steps.parquet"
    assert cli.main(["gold-set", "steps", "--out", str(steps_path), "--strict-only", "--per-class", "2"]) == 0
    n = len(pd.read_parquet(steps_path))
    with MockOllama() as m:  # answers 11 for everything
        assert cli.main(["local-judge", "label", "--host", m.url, "--steps", str(steps_path), "--all",
                         "--stratify-by", "none", "--out", str(tmp_path / "local")]) == 0
    labels = tmp_path / "local" / "labels.parquet"
    assert cli.main(["judge-accuracy", "--labels", str(labels), "--out", str(tmp_path / "acc"),
                     "--n-boot", "50", "--no-figures"]) == 0
    rep = json.loads((tmp_path / "acc" / "judge_accuracy.json").read_text())["judges"]["ollama/qwen2.5:7b-instruct"]
    assert rep["strict"]["n"] == n and rep["n_from_step_label"] == 0
    assert rep["strict"]["per_class"]["11"]["recall"] == 1.0
