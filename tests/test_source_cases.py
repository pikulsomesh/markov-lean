from pathlib import Path

import pandas as pd

from markov_lean import case_study, source_cases as sc
from markov_lean.premise_areas import AreaMap

FIX = Path(__file__).parent / "fixtures" / "pfr"

SRC = '''import Mathlib.Data.Real.Basic

namespace Foo

/-- doc -/
theorem bar (a b : ℕ) (h : a = b) : b = a := by
  rw [h]
  -- a comment
  · simp only [Nat.add_comm,
      Nat.mul_comm]
    exact Nat.le_refl _

lemma baz : True := by trivial

theorem term : True := trivial

theorem withHave (p : Prop) (hp : p) : p := by
  have hq : p := by
    exact hp
  induction 3 with
  | zero => simp
  | succ n ih =>
    exact hq
  sorry

end Foo

theorem _root_.Top.qux : True := by
  trivial
'''


def test_scan_file_names_steps_and_sorry(tmp_path):
    p = tmp_path / "Proj" / "A.lean"
    p.parent.mkdir()
    p.write_text(SRC)
    f = sc.scan_file(p, "Proj/A.lean")
    assert f.decls == ["Foo.bar", "Foo.baz", "Foo.term", "Foo.withHave", "Top.qux"]
    proofs = {n: (s, bad) for n, s, bad in f.proofs}
    assert "Foo.term" not in proofs  # term-mode proof
    assert proofs["Foo.bar"][0] == ["rw [h]", "simp only [Nat.add_comm, Nat.mul_comm]", "exact Nat.le_refl _"]
    assert proofs["Foo.baz"][0] == ["trivial"]
    steps, bad = proofs["Foo.withHave"]
    assert bad and steps == ["have hq : p := by", "exact hp", "induction 3 with", "simp", "exact hq", "sorry"]
    st = sc.steps_table([f], "proj")
    assert set(st["proof_id"]) == {"proj:Foo.bar", "proj:Foo.baz", "proj:Top.qux"}  # sorry proof dropped


def test_case_map_mirror_default_and_own():
    cm = sc.CaseMap("Carleson", {"Classical": "42"}, default="28", mirrors=["ToMathlib"])
    assert cm.area("Carleson/Classical/X.lean") == "42"
    assert cm.area("Carleson/Psi.lean") == "28"
    assert cm.area("Carleson/ToMathlib/X.lean") is None
    assert cm.mirrored("Carleson/ToMathlib/MeasureTheory/X.lean") == "Mathlib/MeasureTheory/X.lean"
    base = AreaMap.load()
    m = sc.MirrorAreaMap(base, cm)
    assert m.area("Carleson/ToMathlib/MeasureTheory/X.lean") == base.area("Mathlib/MeasureTheory/X.lean")
    assert m.area("Carleson/Psi.lean") is None
    assert sc.OwnAreaMap(base, cm).area("Carleson/Psi.lean") == "28"


def test_case_study_convert_on_pfr_fixture(tmp_path):
    cfg = Path(__file__).parents[1] / "configs" / "cases" / "pfr.yaml"
    s = case_study.convert(cfg, FIX, tmp_path)
    assert s["case"] == "pfr" and s["proofs"] == 2
    lab = pd.read_parquet(tmp_path / "free_labels.parquet")
    assert set(lab.columns) == {"proof_id", "step", "file_area", "premise_area", "premise_own_area"}
    assert lab["premise_own_area"].notna().sum() >= lab["premise_area"].notna().sum()


def test_configs_load_and_files(tmp_path):
    for p in (Path(__file__).parents[1] / "configs" / "cases").glob("*.yaml"):
        cfg = sc.load_case(p)
        assert {"name", "repo", "rev", "root", "include"} <= set(cfg)
    (tmp_path / "PFR" / "Tactic").mkdir(parents=True)
    (tmp_path / "PFR" / "A.lean").write_text("")
    (tmp_path / "PFR" / "Tactic" / "T.lean").write_text("")
    cfg = sc.load_case(Path(__file__).parents[1] / "configs" / "cases" / "pfr.yaml")
    assert case_study.case_files(cfg, tmp_path) == ["PFR/A.lean"]
