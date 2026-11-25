import json

import numpy as np
import pandas as pd
import pytest

from markov_lean import cli
from markov_lean import premise_areas as pa
from markov_lean.msc import normalize

ALG_BIG = "Mathlib/Algebra/BigOperators/Basic.lean"
NAT = "Mathlib/Data/Nat/Basic.lean"
TOP = "Mathlib/Topology/Basic.lean"
SET = "Mathlib/Data/Set/Basic.lean"


def amap():
    return pa.AreaMap({"Algebra": "08", "Algebra/BigOperators": "05", "Data": None, "Data/Nat": "11",
                       "Data/Set": "03", "Topology": "54"}, frozenset({"Data/Set"}))


def test_mathlib_relpath():
    assert pa.mathlib_relpath("Mathlib/Algebra/Group/Defs.lean") == "Algebra/Group/Defs"
    assert pa.mathlib_relpath(".lake/packages/mathlib/Mathlib/Topology/Basic.lean") == "Topology/Basic"
    assert pa.mathlib_relpath(".lake/packages/lean4/src/lean/Init/Core.lean") is None
    assert pa.mathlib_relpath(".lake/packages/batteries/Batteries/Data/List/Basic.lean") is None
    assert pa.mathlib_relpath(None) is None and pa.mathlib_relpath("") is None


def test_area_map_longest_prefix_glue_and_null():
    m = amap()
    assert m.area(ALG_BIG) == "05"
    assert m.area("Mathlib/Algebra/Group/Defs.lean") == "08"
    assert m.area("Mathlib/Data/Option/Basic.lean") is None          # null folder
    assert m.area(SET) is None and m.area(SET, glue=True) == "03"    # glue
    assert m.area(TOP, level="folder") == "Topology"
    assert m.area("Mathlib/AlgebraicGeometry/X.lean") is None         # prefix is per folder, not per string
    assert m.area("Init/Core.lean") is None


def test_default_map_loads_with_valid_codes():
    m = pa.AreaMap.load()
    assert all(v is None or normalize(v) == v for v in m.areas.values())
    for top in ("Algebra", "Analysis", "Topology", "MeasureTheory", "NumberTheory", "CategoryTheory",
                "Combinatorics", "RingTheory", "LinearAlgebra", "Order", "SetTheory", "Probability"):
        assert m.areas.get(top) is not None, top
    assert m.area("Mathlib/NumberTheory/Padics/PadicVal.lean") == "11"
    assert m.area("Mathlib/Analysis/Complex/Basic.lean") == "30"
    assert m.area("Mathlib/Logic/Basic.lean") is None and m.area("Mathlib/Logic/Basic.lean", glue=True) == "03"
    assert m.area("Mathlib/Algebra/Order/Ring/Lemmas.lean") is None
    assert m.area("Mathlib/Algebra/Order/Ring/Lemmas.lean", glue=True) == "06"


def test_load_rejects_bad_codes(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("areas:\n  Topology: '99'\n")
    with pytest.raises(ValueError):
        pa.AreaMap.load(p)


def resolver():
    index = {"Finset.sum_comm": ALG_BIG, "Finset.sum_const": ALG_BIG, "foo_bar": TOP, "Set.mem_union": SET}
    index.update({f"Nat.lemma{i}": NAT for i in range(6)})
    return pa.Resolver(index, amap(), min_prefix=5, majority=0.6, max_suffix=10)


def test_resolver_tiers():
    r = resolver()
    assert r.areas("sum_comm", "Finset.my_thm")[(False, "msc")] == ("05", "namespace_open")
    assert r.areas("foo_bar", "Finset.my_thm")[(False, "msc")] == ("54", "exact")
    assert r.areas("Nat.unknown_lemma", "x")[(False, "msc")] == ("11", "namespace_majority")
    assert r.areas("sum_const", "Real.thm")[(False, "msc")] == ("05", "suffix")
    assert r.areas("no_such_lemma", "x")[(False, "msc")] == (None, None)
    # glue: ignored in the main labels, kept in the glue variant
    assert r.areas("Set.mem_union", "x")[(False, "msc")] == (None, "exact")
    assert r.areas("Set.mem_union", "x")[(True, "msc")] == ("03", "exact")
    # too few names under a prefix: not placed
    assert r.areas("Finset.other", "x")[(False, "msc")] == (None, None)


def test_build_index_drops_names_in_two_files():
    t = pd.DataFrame({"theorem": ["a", "a", "b"], "file_path": [TOP, NAT, TOP]})
    assert pa.build_index([t]) == {"b": TOP}


def test_ordered_cited_and_majority():
    assert pa.ordered_cited("rw [foo_bar, Nat.lemma1, foo_bar]") == ["foo_bar", "Nat.lemma1"]
    assert pa.ordered_cited("simp") == []
    assert pa.majority(["54", "11", "11", "54"]) == "54"   # tie: first cited
    assert pa.majority(["11", "54", "54"]) == "54"
    assert pa.majority([]) is None


def steps_table():
    rows = [
        ("p1", 0, "rw [Finset.sum_comm]"), ("p1", 1, "simp"), ("p1", 2, "exact foo_bar"),
        ("p1", 3, "rw [Nat.lemma1, Nat.lemma2, foo_bar]"),
        ("p2", 0, "ring"), ("p2", 1, "exact foo_bar"),
    ]
    return pd.DataFrame({"proof_id": [r[0] for r in rows], "step": [r[1] for r in rows],
                         "tactic": [r[2] for r in rows], "state_before": ["x : ℕ\n⊢ x = x"] * len(rows),
                         "theorem": ["T.p1"] * 4 + ["T.p2"] * 2, "file_path": [TOP] * 6, "split": "train"})


def test_label_steps_and_carry():
    lab = pa.label_steps(steps_table(), resolver())
    p1 = lab[lab["proof_id"] == "p1"]
    assert list(p1["premise"]) == ["05", "", "54", "11"]
    assert list(p1["premise_carry"]) == ["05", "05", "54", "11"]
    assert list(lab[lab["proof_id"] == "p2"]["premise_carry"]) == ["", "54"]   # leading step stays empty
    assert p1["n_cited"].tolist() == [1, 0, 1, 3]
    assert p1["disagree"].iloc[3] == pytest.approx(1 / 3)
    assert p1["premise_folder"].tolist() == ["Algebra", "", "Topology", "Data"]
    assert pa.noise_rate(lab, "premise") == pytest.approx(1 / 3)


def test_read_annotated_and_label(tmp_path):
    raw = [
        {"full_name": "T.p1", "file_path": "Mathlib/Topology/Basic.lean", "traced_tactics": [
            {"tactic": "rw [x]", "state_before": "", "state_after": "",
             "annotated_tactic": ["rw [<a>x</a>]", [{"full_name": "Nat.x", "def_path": NAT}]]},
            {"tactic": "simp [y]", "state_before": "", "state_after": "",
             "annotated_tactic": ["simp [<a>y</a>]", [{"full_name": "y", "def_path": ".lake/packages/mathlib/" + TOP},
                                                    {"full_name": "core", "def_path": "Init/Core.lean"}]]}]},
        {"full_name": "T.p1", "file_path": "Mathlib/Topology/Other.lean", "traced_tactics": [
            {"tactic": "simp", "state_before": "", "state_after": "", "annotated_tactic": ["simp", []]}]},
    ]
    (tmp_path / "train.json").write_text(json.dumps(raw))
    ann = pa.read_annotated(tmp_path, "train")
    assert ann["proof_id"].tolist() == ["leandojo:T.p1", "leandojo:T.p1", "leandojo:T.p1#2"]
    steps = pd.DataFrame({"proof_id": ["leandojo:T.p1"] * 2, "step": [0, 1], "tactic": ["rw [x]", "simp [y]"],
                          "state_before": ["", ""], "theorem": ["T.p1"] * 2, "file_path": [TOP] * 2})
    lab = pa.label_steps(steps, pa.Resolver({}, amap()), ann)
    assert lab["premise"].tolist() == ["11", "54"]
    assert lab["tiers"].tolist() == ["annotated", "annotated annotated"]


def test_judge_agreement(tmp_path):
    lab = pa.label_steps(steps_table(), resolver())
    labels = pd.DataFrame({"proof_id": ["p1", "p1", "p1", "p2"], "step": [0, 2, 3, 1], "judge": "consensus",
                           "msc": ["05", "54", "54", "54"]})
    labels.to_parquet(tmp_path / "labels.parquet")
    a = pa.judge_agreement(lab, tmp_path / "labels.parquet")
    s = a["sources"]["premise"]
    assert a["available"] and s["n_steps"] == 4 and s["accuracy"] == pytest.approx(0.75)
    assert pa.judge_agreement(lab, None)["available"] is False
    assert pa.judge_agreement(lab, tmp_path / "missing.parquet")["available"] is False


def synthetic_steps(n_proofs=60, length=6, seed=0):
    rng = np.random.default_rng(seed)
    lemmas = {"Topology": [f"Topo.l{i}" for i in range(6)], "NumberTheory": [f"Num.l{i}" for i in range(6)]}
    rows = []
    for p in range(n_proofs):
        area = "Topology" if p % 2 else "NumberTheory"
        for s in range(length):
            if rng.random() < 0.2:
                area = "NumberTheory" if area == "Topology" else "Topology"
            other = "NumberTheory" if area == "Topology" else "Topology"
            u = rng.random()
            tac = ("simp" if u < 0.3 else f"rw [{rng.choice(lemmas[area])}]" if u < 0.7 else
                   f"rw [{rng.choice(lemmas[area])}, {rng.choice(lemmas[area if rng.random() < 0.8 else other])}]")
            rows.append({"proof_id": f"p{p}", "step": s, "tactic": tac, "state_before": "", "state_after": "",
                         "theorem": f"Thm.p{p}", "file_path": f"Mathlib/{['Topology', 'NumberTheory'][p % 2]}/A.lean",
                         "split": "train", "source": "leandojo", "true_area": None})
    idx = [{"proof_id": f"i{k}", "step": 0, "tactic": "rfl", "state_before": "", "state_after": "",
            "theorem": name, "file_path": f"Mathlib/{fold}/B.lean", "split": "train", "source": "leandojo",
            "true_area": None} for k, (fold, names) in enumerate(lemmas.items()) for name in names]
    return pd.DataFrame(rows), pd.DataFrame(idx)


def test_run_end_to_end(tmp_path):
    steps, idx = synthetic_steps()
    steps.to_parquet(tmp_path / "steps.parquet")
    idx.to_parquet(tmp_path / "idx.parquet")
    res = pa.run([tmp_path / "steps.parquet"], tmp_path / "out", index_paths=[tmp_path / "idx.parquet"],
                 n_perm=20, n_boot=50, figures=False, verbose=False)
    out = tmp_path / "out"
    for f in ("labels.parquet", "coverage.json", "coverage.csv", "agreement.json",
              "switches/train/switches.json", "switches/train/REPORT.md", "switches/train/table_switches.tex"):
        assert (out / f).exists(), f
    sweep = pd.read_csv(out / "switches/train/noise_sweep.csv")
    assert set(sweep["source"]) == set(pa.SOURCES)
    s = sweep[sweep["source"] == "premise"]
    assert s["null_off_mode"].is_monotonic_increasing or s["null_off_mode"].diff().min() > -0.02
    rep = json.loads((out / "switches/train/switches.json").read_text())
    be = rep["noise_break_even"]["premise"]["off_mode"]
    assert be is None or 0 <= be <= 0.4
    h = res["switches"]["train"]["premise"]
    assert 0 < h["has_switch"] <= 1 and "switch_rate_noise" in h
    cov = pd.read_csv(out / "coverage.csv")
    row = cov[(cov["split"] == "all") & (cov["source"] == "premise")].iloc[0]
    assert 0.5 < row["share_labelled"] < 0.9
    assert set(pd.read_parquet(out / "labels.parquet")["premise"]) <= {"", "11", "54"}


def test_run_min_switch_steps_and_figures(tmp_path):
    pytest.importorskip("seaborn")
    steps, idx = synthetic_steps(n_proofs=20)
    short = steps[steps["proof_id"] == "p0"].head(2)
    steps = pd.concat([steps[steps["proof_id"] != "p0"], short])
    steps.to_parquet(tmp_path / "steps.parquet")
    idx.to_parquet(tmp_path / "idx.parquet")
    res = pa.run([tmp_path / "steps.parquet"], tmp_path / "out", index_paths=[tmp_path / "idx.parquet"], n_perm=10,
                 n_boot=20, min_switch_steps=6, verbose=False)
    proofs = pd.read_csv(tmp_path / "out" / "switches" / "train" / "proofs.csv")
    assert "p0" not in set(proofs["proof_id"])
    sw_dir = tmp_path / "out" / "switches" / "train"
    assert (tmp_path / "out" / "coverage.pdf").exists() and (sw_dir / "switches.pdf").exists()
    assert (sw_dir / "by_length.pdf").exists()
    bl = pd.read_csv(sw_dir / "by_length.csv")
    assert {"observed", "shuffle"} <= set(bl["series"]) and set(bl["statistic"]) == {"has_switch", "switch_rate",
                                                                                     "mean_dwell", "off_mode"}
    assert res["switches"]["train"]["premise"]["n_proofs"] <= 19
    inputs = json.loads((sw_dir / "switches.json").read_text())["inputs"]
    assert inputs == {"split": "train", "min_switch_steps": 6}  # what `markov-lean paper` puts in the caption


def test_cli_passthrough(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["premise-labels", "--help"])
    assert e.value.code == 0
    assert "--raw-root" in capsys.readouterr().out


def test_break_even_interpolates():
    sweep = pd.DataFrame({"epsilon": [0.0, 0.1, 0.2], "null_off_mode": [0.0, 0.1, 0.3], "observed_off_mode": 0.2})
    assert pa.break_even(sweep) == pytest.approx(0.15)
    assert pa.break_even(sweep.assign(observed_off_mode=0.5)) is None
    assert pa.break_even(pd.DataFrame()) is None


def test_short_titles_cut_at_words():
    assert pa._short("Measure and integration") == "Measure and integration"
    assert pa._short("Category theory; homological algebra") == "Category theory"
    assert pa._short("Group theory and generalizations") == "Group theory and…"
