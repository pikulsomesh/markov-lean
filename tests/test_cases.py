"""Tests for the case-study picker (markov_lean.cases) on a small hand-made judged sample."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from markov_lean import cases as C
from markov_lean import cli

A, B = "claude-haiku-5-5", "claude-sonnet-5-5"

# proof id -> (file, [(tactic, judge A area, judge B area, confidence)])
PROOFS = {
    # one clean switch 26 -> 11 at step 1
    "leandojo:Real.foo": ("Mathlib/Analysis/Foo.lean", [
        ("intro x", "26", "26", 0.9), ("rw [Real.sqrt_eq]", "26", "26", 0.9),
        ("norm_cast", "11", "11", 0.9), ("omega", "11", "11", 0.9), ("simp", "11", "11", 0.9)]),
    # two switches 13 -> 05 -> 13, all agreed
    "leandojo:Ideal.bar": ("Mathlib/RingTheory/Bar.lean", [
        ("intro I", "13", "13", 0.8), ("rw [Finset.card_eq]", "13", "13", 0.8),
        ("simp", "05", "05", 0.8), ("apply Ideal.span_le", "05", "05", 0.8), ("exact h", "13", "13", 0.8)]),
    # many switches but the judges disagree on half the steps
    "leandojo:Nat.flip": ("Mathlib/NumberTheory/Flip.lean", [
        ("a", "26", "26", 0.9), ("b", "11", "05", 0.9), ("c", "26", "26", 0.9),
        ("d", "11", "05", 0.9), ("e", "26", "26", 0.9), ("f", "11", "11", 0.9)]),
    # switch, but the switching step is low-confidence
    "leandojo:Set.low": ("Mathlib/Topology/Low.lean", [
        ("intro", "54", "54", 0.9), ("simp", "54", "54", 0.5), ("exact h", "26", "26", 0.9),
        ("rfl", "26", "26", 0.9)]),
    # same file and same 26 -> 11 move as Real.foo, ranked just below it
    "leandojo:Real.baz": ("Mathlib/Analysis/Foo.lean", [
        ("intro y", "26", "26", 0.85), ("rw [h]", "26", "26", 0.85), ("ring", "11", "11", 0.85),
        ("omega", "11", "11", 0.85), ("simp", "11", "11", 0.85)]),
    # stays in one area
    "leandojo:Finset.stay": ("Mathlib/Combinatorics/Stay.lean", [
        ("simp", "05", "05", 0.9), ("ring", "05", "05", 0.9), ("omega", "05", "05", 0.9),
        ("rfl", "05", "05", 0.9)]),
    # too long
    "leandojo:Long.proof": ("Mathlib/Analysis/Long.lean",
                            [(f"t{i}", "26" if i < 10 else "11", "26" if i < 10 else "11", 0.9) for i in range(20)]),
    # a switch, but one tactic is too long to read
    "leandojo:Wide.proof": ("Mathlib/Analysis/Wide.lean", [
        ("x" * 80, "26", "26", 0.9), ("simp", "26", "26", 0.9), ("ring", "60", "60", 0.9),
        ("rfl", "60", "60", 0.9)]),
}


def fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    steps, labels = [], []
    for pid, (fp, rows) in PROOFS.items():
        for i, (tac, a, b, conf) in enumerate(rows):
            steps.append({"proof_id": pid, "step": i, "source": "leandojo", "split": "train",
                          "theorem": pid.split(":", 1)[1], "file_path": fp, "tactic": tac,
                          "state_before": "⊢ x ≤ y", "state_after": "no goals", "true_area": None})
            labels += [{"proof_id": pid, "step": i, "judge": A, "msc": a, "confidence": conf},
                       {"proof_id": pid, "step": i, "judge": B, "msc": b, "confidence": conf}]
        labels.append({"proof_id": pid, "step": -1, "judge": A, "msc": rows[0][1], "confidence": 0.9})
    return pd.DataFrame(steps), pd.DataFrame(labels)


def test_step_areas_two_judges():
    _, labels = fixture()
    ar = C.step_areas(labels, 0.7).set_index(["proof_id", "step"])
    assert ar.loc[("leandojo:Nat.flip", 1), "area"] == C.UNCERTAIN
    assert not ar.loc[("leandojo:Nat.flip", 1), "confident"]
    assert ar.loc[("leandojo:Set.low", 1), "area"] == "54"
    assert not ar.loc[("leandojo:Set.low", 1), "confident"]  # agreed but under the threshold
    assert ar.loc[("leandojo:Real.foo", 2), "confident"]
    assert (ar.reset_index()["step"] >= 0).all()  # whole-proof labels are left out


def test_switches_and_paths():
    steps, labels = fixture()
    cs = C.case_steps(steps, C.step_areas(labels, 0.7))
    proofs = C.proof_table(cs, C.CaseConfig()).set_index("proof_id")
    foo = proofs.loc["leandojo:Real.foo"]
    assert foo["n_switches"] == 1 and foo["switch_steps"] == [1]
    assert foo["switch_tactics"] == ["rw [Real.sqrt_eq]"] and foo["switch_pairs"] == [("26", "11")]
    assert foo["area_path"] == ["26", "11"]
    bar = proofs.loc["leandojo:Ideal.bar"]
    assert bar["switch_steps"] == [1, 3] and bar["area_path"] == ["13", "05", "13"]
    assert proofs.loc["leandojo:Set.low", "n_switches"] == 0  # low-confidence step breaks the pair
    assert proofs.loc["leandojo:Finset.stay", "excluded_because"] == "no confident switch"
    assert "judges disagree" in proofs.loc["leandojo:Nat.flip", "excluded_because"]
    assert "length" in proofs.loc["leandojo:Long.proof", "excluded_because"]
    assert "long tactic" in proofs.loc["leandojo:Wide.proof", "excluded_because"]


def test_pick_ranks_and_keeps_variety():
    steps, labels = fixture()
    picked = C.select(steps, labels, C.CaseConfig(n_cases=2)).picked
    # Ideal.bar has two switches, so it ranks first; Real.baz and Real.foo tie and break on the id
    assert list(picked["proof_id"]) == ["leandojo:Ideal.bar", "leandojo:Real.baz"]
    three = C.select(steps, labels, C.CaseConfig(n_cases=3)).picked
    # only three eligible proofs, so the same-file one fills the last slot
    assert set(three["proof_id"]) == {"leandojo:Ideal.bar", "leandojo:Real.foo", "leandojo:Real.baz"}
    assert three["proof_id"].iloc[0] == "leandojo:Ideal.bar"


def test_diversity_skips_repeat_file_until_needed():
    steps, labels = fixture()
    extra_steps = steps[steps.proof_id == "leandojo:Ideal.bar"].assign(
        proof_id="leandojo:Other.qux", theorem="Other.qux", file_path="Mathlib/Algebra/Qux.lean")
    extra_steps.loc[:, "tactic"] = ["a", "b", "c", "d", "e"]
    extra_labels = labels[labels.proof_id == "leandojo:Ideal.bar"].assign(proof_id="leandojo:Other.qux")
    extra_labels.loc[(extra_labels.step >= 0) & (extra_labels.msc == "05"), "msc"] = "60"
    s = pd.concat([steps, extra_steps], ignore_index=True)
    lab = pd.concat([labels, extra_labels], ignore_index=True)
    picked = list(C.select(s, lab, C.CaseConfig(n_cases=3)).picked["proof_id"])
    assert len(picked) == 3
    assert not {"leandojo:Real.foo", "leandojo:Real.baz"} <= set(picked)  # one per file while others remain
    plain = list(C.select(s, lab, C.CaseConfig(n_cases=3, diverse=False)).picked["proof_id"])
    assert len(plain) == 3


def test_ranking_order():
    steps, labels = fixture()
    proofs = C.select(steps, labels, C.CaseConfig()).proofs
    elig = proofs[proofs["eligible"]]
    keys = list(zip(-elig["n_switches"], -elig["agreement"], elig["n_steps"]))
    assert keys == sorted(keys)
    assert proofs["eligible"].iloc[: len(elig)].all()


def test_single_judge_labels():
    steps, labels = fixture()
    one = labels[labels.judge == A]
    cases = C.select(steps, one, C.CaseConfig(min_agreement=0.0))
    assert "leandojo:Nat.flip" in set(cases.proofs.loc[cases.proofs["eligible"], "proof_id"])


def test_unknown_judge_is_an_error():
    _, labels = fixture()
    with pytest.raises(ValueError):
        C.step_areas(labels, 0.7, judges=[A, "nobody"])


def test_tex_escape_handles_lean_text():
    s = C.tex_escape("obtain ⟨x, hx⟩ := h_1 # 50% & {ok} ∀ ε > 0, ℝ ⁻¹ 漢")
    for raw in ("⟨", "∀", "ℝ", "⁻¹", "漢", "#", "%", "&"):
        assert raw not in s.replace(r"\#", "").replace(r"\%", "").replace(r"\&", "")
    assert r"h\_1" in s and r"\{ok\}" in s and r"$\langle$" in s and r"$^{-1}$" in s
    assert C.tex_escape("a.b", breaks=".") == r"a.\allowbreak{}b"


def test_latex_table_lists_picked_proofs():
    steps, labels = fixture()
    cases = C.select(steps, labels, C.CaseConfig(n_cases=2))
    tex = C.latex_table(cases.picked)
    assert tex.count(r"\\") == 1 + len(cases.picked)  # header plus one row per proof
    assert r"\toprule" in tex and r"\bottomrule" in tex
    assert r"13\,$\to$\,05\,$\to$\,13" in tex
    assert r"\texttt{rw [Finset.\allowbreak{}card\_\allowbreak{}eq]} (1)" in tex
    empty = C.latex_table(cases.picked.iloc[0:0])
    assert "no proof met" in empty


def test_short_name_and_tactic_head():
    assert C.short_name("Mathlib.Foo.Bar.baz") == "Bar.baz"
    assert len(C.short_name("A." + "x" * 60)) == 28
    assert C.tactic_head("rw [foo]") == "rw"
    assert C.tactic_head("exact_mod_cast hk") == "exact_mod…"


def test_synthetic_inputs_give_cases():
    steps, labels = C.synthetic_case_inputs(n_proofs=60, seed=3)
    assert set(labels["judge"]) == {"judge-a", "judge-b"}
    cases = C.select(steps, labels)
    assert 1 <= len(cases.picked) <= cases.config.n_cases
    assert (cases.picked["n_switches"] >= 1).all()


def write_run(tmp_path):
    steps, labels = fixture()
    judge = tmp_path / "run" / "judge"
    judge.mkdir(parents=True)
    steps.to_parquet(judge / "sample_steps.parquet", index=False)
    labels.to_parquet(judge / "labels.parquet", index=False)
    return tmp_path / "run"


def test_load_run_and_missing_labels(tmp_path):
    run = write_run(tmp_path)
    s, lab = C.load_run(run)
    assert len(s) and len(lab)
    (run / "judge" / "labels.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="judge labels"):
        C.load_run(run)


def test_config_rejects_unknown_keys():
    assert C.CaseConfig.from_dict({"n_cases": 3}).n_cases == 3
    with pytest.raises(ValueError):
        C.CaseConfig.from_dict({"n_case": 3})


# ------------------------------------------------ figure and CLI (need matplotlib)
def test_case_strips_figure():
    pytest.importorskip("seaborn")
    steps, labels = fixture()
    cases = C.select(steps, labels, C.CaseConfig(n_cases=3))
    fig = C.case_strips(cases.steps, max_steps=14)
    assert fig.n_cases == 3
    texts = {t.get_text() for t in fig.axes[0].texts}
    assert "Ideal.bar" in texts and "rw" in texts
    bold = [t.get_text() for t in fig.axes[0].texts if t.get_fontweight() == "bold"]
    assert sorted(bold) == sorted(["rw", "apply", "rw", "rw"])  # switch tactics of the three picks


def test_cli_writes_figure_table_and_summary(tmp_path, capsys):
    pytest.importorskip("seaborn")
    run = write_run(tmp_path)
    out = tmp_path / "figs"
    assert cli.main(["cases", "--run", str(run), "--out", str(out), "--n", "2"]) == 0
    assert (out / "case_studies.pdf").stat().st_size > 1000
    assert r"Ideal.\allowbreak{}bar" in (out / "case_studies_table.tex").read_text()
    summary = json.loads((out / "case_studies.json").read_text())
    assert summary["n_eligible"] == 3 and len(summary["picked"]) == 2
    assert "picked 2" in capsys.readouterr().out


def test_cli_without_input_fails(capsys):
    assert C.main([]) == 2


def test_cli_nothing_eligible_returns_1(tmp_path):
    run = write_run(tmp_path)
    assert C.main(["--run", str(run), "--out", str(tmp_path / "o"), "--min-agreement", "1.1"]) == 1
    assert "no proof met" in (tmp_path / "o" / "case_studies_table.tex").read_text()
