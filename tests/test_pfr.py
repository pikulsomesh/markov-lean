import json
from pathlib import Path

import pandas as pd

from markov_lean import pfr
from markov_lean.data import STEP_COLUMNS
from markov_lean.premise_areas import AreaMap

FIX = Path(__file__).parent / "fixtures" / "pfr"


def test_build_tables_drops_focusing_dots_and_keeps_order():
    steps, prem = pfr.build_tables(pfr.read_raw(FIX))
    # 9 extracted steps, 2 of them focusing dots
    assert len(steps) == 7
    assert not steps["tactic"].str.startswith("·").any()
    assert set(steps["proof_id"]) == {"pfr:Foo.bar", "pfr:baz"}
    bar = steps[steps["proof_id"] == "pfr:Foo.bar"]
    assert list(bar["step"]) == list(range(len(bar)))
    assert bar["tactic"].iloc[0] == "constructor"
    assert set(steps["source"]) == {"pfr"} and list(steps.columns) == STEP_COLUMNS
    assert prem.loc[0, "premise_paths"] == ["PFR/ForMathlib/Entropy/Basic.lean"]


def test_module_path():
    assert pfr.module_path("PFR.Kullback") == "PFR/Kullback.lean"
    assert pfr.module_path("Mathlib.Probability.Notation") == ".lake/packages/mathlib/Mathlib/Probability/Notation.lean"


def test_pfr_map_prefix_and_mathlib_mirror():
    m = pfr.PFRMap.load()
    assert m.area("PFR/ForMathlib/Entropy/Basic.lean") == "94"
    assert m.area("PFR/ForMathlib/Entropy/RuzsaSetDist.lean") == "11"
    assert m.area("PFR/HundredPercent.lean") == "11"
    assert m.area("PFR/Tactic/RPowSimp.lean") is None
    assert m.area("PFR/Mathlib/Probability/IdentDistrib.lean") is None  # handled by the Mathlib table
    assert AreaMap.load().area("PFR/Mathlib/Probability/IdentDistrib.lean") == "60"


def test_free_labels_and_switches():
    steps, prem = pfr.build_tables(pfr.read_raw(FIX))
    pmap = pfr.PFRMap({"Demo": "11", "ForMathlib/Entropy": "94"}, {"pfr": "11"})
    amap = AreaMap({"Combinatorics": "05"}, frozenset())
    lab = pfr.free_labels(steps, prem, amap, pmap, {"baz": "pfr"})
    assert set(lab["file_area"].dropna()) == {"11"}
    assert lab.set_index("proof_id")["chapter_area"].loc["pfr:baz"] == "11"
    first = lab[(lab.proof_id == "pfr:Foo.bar") & (lab.step == 0)].iloc[0]
    assert pd.isna(first["premise_area"]) and first["premise_pfr_area"] == "94"
    s = pfr.switch_stats(lab, "premise_pfr_area")
    assert s["proofs"] == 1 and s["labelled_steps"] == 2 and s["switches_per_move"] == 1.0
    assert s["share_off_home_area"] == 1.0


def test_blueprint_chapters(tmp_path):
    ch = tmp_path / "blueprint" / "src" / "chapter"
    ch.mkdir(parents=True)
    (ch / "entropy.tex").write_text(r"\lean{ProbabilityTheory.entropy, ProbabilityTheory.entropy_le}")
    assert pfr.blueprint_chapters(tmp_path) == {"ProbabilityTheory.entropy": "entropy",
                                                "ProbabilityTheory.entropy_le": "entropy"}
    assert pfr.blueprint_chapters(None) == {}


def test_convert_writes_outputs(tmp_path):
    s = pfr.convert(FIX, tmp_path)
    assert s["proofs"] == 2 and s["steps"] == 7 and s["modules"] == 1
    assert json.loads((tmp_path / "summary.json").read_text())["steps"] == 7
    st = pd.read_parquet(tmp_path / "pfr_steps.parquet")
    assert len(st) == 7
    assert pfr.main(["convert", "--raw", str(FIX), "--out", str(tmp_path / "b")]) == 0
