"""Tests for the bridge-step analysis. Synthetic tables only, no files outside tmp_path."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from markov_lean import bridges as br
from markov_lean import cli

STATE = "x y : ℕ\nh₀ : 0 < x\nhxy : x < y\n⊢ x * y > 0"


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("text,name", [
    ("simp only [Nat.add_comm] at h ⊢", "simp"),
    ("· nlinarith [sq_nonneg (x - y)]", "nlinarith"),
    ("case inl h => exact Or.inl h", "exact"),
    ("next => linarith", "linarith"),
    ("exact?", "exact?"),
    ("refine' ⟨_, _⟩", "refine'"),
    ("  (rw [foo_bar])", "rw"),
    ("", ""),
])
def test_tactic_name(text, name):
    assert br.tactic_name(text) == name


def test_local_names_reads_hypotheses_not_goal_or_wrapped_lines():
    s = "x y : ℕ\nh₀ : 0 < x\ninst✝ : Fact p\nlong_hyp :\n  a_b_c = d\n⊢ foo_bar x\n\ncase h\nz : ℤ\n⊢ True"
    assert br.local_names(s) == {"x", "y", "h₀", "inst", "long_hyp", "z"}


@pytest.mark.parametrize("tactic,cited", [
    ("simp only [Nat.add_comm, mul_pos] at hxy ⊢", ["Nat.add_comm", "mul_pos"]),
    ("nlinarith [sq_nonneg (x - y), h₀, hxy.le]", ["sq_nonneg"]),
    ("exact mul_pos h₀ (lt_trans h₀ hxy)", ["lt_trans", "mul_pos"]),
    ("have h_key : x ≤ y := le_of_lt hxy", ["le_of_lt"]),
    ("rw [← Finset.sum_comm]", ["Finset.sum_comm"]),
    ("obtain ⟨k_1, hk⟩ := Nat.exists_eq_add_of_lt hxy", ["Nat.exists_eq_add_of_lt"]),
    ("intro z_pos hz_pos", []),
    ("refine ⟨fun a_b => ?_, le_refl _⟩", ["le_refl"]),
    ("apply _root_.le_trans", ["le_trans"]),
    ("norm_num -- uses foo_bar", []),
    ("field_simp", []),
    ("exact this.trans h_main", []),
])
def test_cited_names(tactic, cited):
    assert br.cited_names(tactic, STATE) == cited


def test_hypothesis_named_like_a_lemma_is_local():
    assert br.cited_names("exact add_pos_h", "add_pos_h : 0 < a\n⊢ 0 < a") == []


# ------------------------------------------------------------------ fixtures
def _synthetic(n_proofs=300, seed=0, p_bridge=0.8):
    """Proofs over areas 11 and 26 where `rw [Nat.cast_sum]` drives most 11 -> 26 switches."""
    rng = np.random.default_rng(seed)
    steps, labels, assign = [], [], []
    for p in range(n_proofs):
        pid = f"synthetic:p{p}"
        n = int(rng.integers(4, 12))
        area = "11"
        for t in range(n):
            last = t == n - 1
            move = None if last else rng.random()
            if area == "11" and not last and move < 0.25:
                nxt = "26"
                tac = "rw [Nat.cast_sum]" if rng.random() < p_bridge else "simp"
            elif area == "26" and not last and move < 0.10:
                nxt = "11"
                tac = "norm_cast"
            else:
                nxt = area
                tac = str(rng.choice(["simp", "linarith", "ring_nf [mul_comm]", "omega"]))
            steps.append({"proof_id": pid, "step": t, "source": "synthetic", "split": "train", "theorem": f"p{p}",
                          "file_path": "", "tactic": tac, "state_before": "n : ℕ\n⊢ goal", "state_after": "",
                          "true_area": area})
            conf = float(rng.uniform(0.6, 1))
            for j in ("claude-haiku-5-5", "claude-sonnet-5-5"):
                labels.append({"proof_id": pid, "step": t, "judge": j, "msc": area, "confidence": conf})
            assign.append({"proof_id": pid, "step": t, "microstate": int(area), "macrostate": 0 if area == "11" else 1,
                           "membership": 0.9})
            area = nxt
        for j in ("claude-haiku-5-5", "claude-sonnet-5-5"):
            labels.append({"proof_id": pid, "step": -1, "judge": j, "msc": "11", "confidence": 0.9})
    return pd.DataFrame(steps), pd.DataFrame(labels), pd.DataFrame(assign)


def _pilot_dir(tmp_path, labels=True, msm=True):
    steps, lab, assign = _synthetic()
    (tmp_path / "judge").mkdir(parents=True)
    steps.to_parquet(tmp_path / "judge" / "sample_steps.parquet", index=False)
    if labels:
        lab.to_parquet(tmp_path / "judge" / "labels.parquet", index=False)
    if msm:
        (tmp_path / "msm").mkdir()
        assign.to_parquet(tmp_path / "msm" / "assignments.parquet", index=False)
    return tmp_path


# ------------------------------------------------------------------ moves
def test_moves_credit_the_tactic_at_the_first_step_and_need_adjacent_labels():
    steps = pd.DataFrame({"proof_id": ["a"] * 4, "step": [0, 1, 2, 3],
                          "tactic": ["simp", "rw [foo_bar]", "linarith", "ring"], "state_before": ["⊢ p"] * 4})
    areas = pd.DataFrame({"proof_id": ["a"] * 3, "step": [0, 1, 3], "area": ["11", "26", "26"]})
    mv = br.build_moves(steps, areas, "judge")
    assert len(mv) == 1  # 1 -> 3 is not a move: step 2 has no label
    r = mv.iloc[0]
    assert (r.step, r.from_area, r.to_area, bool(r.switch), r.tactic_name) == (0, "11", "26", True, "simp")


def test_premises_column_overrides_parsing():
    steps = pd.DataFrame({"proof_id": ["a", "a"], "step": [0, 1], "tactic": ["simp", "ring"],
                          "state_before": ["", ""], "premises": [["Foo.bar", "baz_qux"], []]})
    areas = pd.DataFrame({"proof_id": ["a", "a"], "step": [0, 1], "area": ["1", "2"]})
    assert br.build_moves(steps, areas, "macrostate").iloc[0].premises == "Foo.bar baz_qux"


# ------------------------------------------------------------------ lift
def _moves(rows):
    return pd.DataFrame([{"area_source": "judge", "proof_id": p, "step": i, "from_area": a, "to_area": b,
                          "switch": a != b, "tactic_name": t, "premises": pr, "tactic": t}
                         for i, (p, a, b, t, pr) in enumerate(rows)])


def test_lift_matches_hand_computation():
    # 8 moves, 4 switches (all 1->2); `rw` on 3 switches and 1 stay; `simp` on 1 switch and 3 stays.
    rows = [("p0", "1", "2", "rw", "foo_bar"), ("p1", "1", "2", "rw", ""), ("p2", "1", "2", "rw", ""),
            ("p3", "1", "2", "simp", ""), ("p4", "1", "1", "rw", ""), ("p5", "1", "1", "simp", ""),
            ("p6", "1", "1", "simp", ""), ("p7", "1", "1", "simp", "")]
    t = br.lift_table(_moves(rows), n_boot=0, min_count=1, min_pair_switches=1)
    allt = t[(t.scope == "all") & (t.kind == "tactic")].set_index("feature")
    # P(rw | switch) = 3/4, P(rw) = 4/8 -> 1.5 ; simp: (1/4)/(4/8) = 0.5
    assert allt.loc["rw", "lift"] == pytest.approx(1.5)
    assert allt.loc["simp", "lift"] == pytest.approx(0.5)
    assert allt.loc["rw", "switch_rate_given_feature"] == pytest.approx(0.75)
    assert allt.loc["rw", "base_switch_rate"] == pytest.approx(0.5)
    assert allt.loc["rw", "rank"] == 1
    pair = t[(t.scope == "pair") & (t.kind == "tactic")].set_index("feature")
    assert pair.loc["rw", "lift"] == pytest.approx(1.5)  # all moves start in area 1, so the base is the same
    assert t[(t.kind == "premise")].feature.tolist() == ["foo_bar", "foo_bar"]


def test_pair_base_is_moves_out_of_the_from_area():
    # area 1: `rw` is everywhere, so it is no bridge out of 1; overall it looks like one.
    rows = []
    for i in range(10):
        rows += [(f"a{i}", "1", "2", "rw", ""), (f"b{i}", "1", "1", "rw", ""),
                 (f"c{i}", "3", "3", "simp", ""), (f"d{i}", "3", "3", "simp", "")]
    t = br.lift_table(_moves(rows), n_boot=0, min_count=1, min_pair_switches=1)
    overall = t[(t.scope == "all") & (t.feature == "rw")].lift.item()
    pair = t[(t.scope == "pair") & (t.feature == "rw")].lift.item()
    assert overall == pytest.approx(2.0)
    assert pair == pytest.approx(1.0)


def test_planted_bridge_is_found_with_interval_above_one():
    steps, labels, _ = _synthetic()
    areas, used = br.judge_areas(labels)
    assert used.startswith("consensus of")
    mv = br.build_moves(steps, areas, "judge")
    t = br.lift_table(mv, n_boot=300, seed=1)
    tac = t[(t.scope == "all") & (t.kind == "tactic")].set_index("feature")
    assert tac.loc["rw", "rank"] == 1
    assert tac.loc["rw", "lift_lo"] > 1
    assert tac.loc["linarith", "lift_hi"] < 1  # linarith only appears on stays
    assert tac.loc["rw", "q_value"] < 0.05
    prem = t[(t.scope == "all") & (t.kind == "premise")]
    assert prem.iloc[0].feature == "Nat.cast_sum"
    pair = t[(t.scope == "pair") & (t.from_area == "11") & (t.to_area == "26") & (t.kind == "tactic")]
    assert pair.iloc[0].feature == "rw" and pair.iloc[0].lift_lo > 1
    back = t[(t.scope == "pair") & (t.from_area == "26") & (t.to_area == "11") & (t.kind == "tactic")]
    assert back.iloc[0].feature == "norm_cast"


def test_bootstrap_is_seeded():
    steps, labels, _ = _synthetic(n_proofs=80)
    mv = br.build_moves(steps, br.judge_areas(labels)[0], "judge")
    a = br.lift_table(mv, n_boot=50, seed=3)
    b = br.lift_table(mv, n_boot=50, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_no_switches_gives_empty_table():
    rows = [("p0", "1", "1", "simp", "")]
    assert br.lift_table(_moves(rows), n_boot=10).empty


# ------------------------------------------------------------------ labels
def test_judge_areas_prefers_consensus_and_drops_proof_level():
    lab = pd.DataFrame({"proof_id": ["a", "a", "a", "a"], "step": [0, 0, 1, -1],
                        "judge": ["consensus", "claude-haiku-5-5", "consensus", "consensus"],
                        "msc": ["5", "11", "26", "26"], "confidence": [0.9, 0.8, 0.4, 0.9]})
    areas, used = br.judge_areas(lab)
    assert used == "consensus"
    assert areas.to_dict("records") == [{"proof_id": "a", "step": 0, "area": "05"},
                                        {"proof_id": "a", "step": 1, "area": "26"}]
    assert len(br.judge_areas(lab, min_confidence=0.5)[0]) == 1
    assert br.judge_areas(lab, judge="claude-haiku-5-5")[0].area.tolist() == ["11"]


def test_two_judges_use_rows_they_agree_on():
    lab = pd.DataFrame({"proof_id": ["a"] * 4, "step": [0, 0, 1, 1], "judge": ["j1", "j2", "j1", "j2"],
                        "msc": ["11", "11", "26", "05"], "confidence": [0.9] * 4})
    areas, used = br.judge_areas(lab)
    assert used == "consensus of j1 and j2"
    assert areas.step.tolist() == [0]


# ------------------------------------------------------------------ end to end
def test_cli_writes_tables_report_and_summary(tmp_path):
    run = _pilot_dir(tmp_path / "pilot3")
    out = tmp_path / "bridges"
    code = cli.main(["bridges", "--run", str(run), "--out", str(out), "--n-boot", "100", "--no-figure"])
    assert code == 0
    t = pd.read_parquet(out / "bridges.parquet")
    assert set(t.area_source) == {"judge", "macrostate"}
    assert set(t.scope) == {"all", "pair"}
    assert (out / "bridges.csv").exists() and (out / "moves.parquet").exists()
    mv = pd.read_parquet(out / "moves.parquet")
    assert {"proof_id", "step", "from_area", "to_area", "switch", "tactic_name", "premises"} <= set(mv.columns)
    s = json.loads((out / "summary.json").read_text())
    assert s["sources"]["judge"]["n_switches"] > 0 and s["sources"]["macrostate"]["n_switches"] > 0
    rep = (out / "REPORT.md").read_text()
    assert "`rw`" in rep and "`Nat.cast_sum`" in rep and "Number theory" in rep
    top = t[(t.area_source == "macrostate") & (t.scope == "all") & (t.kind == "tactic")].sort_values("rank")
    assert top.iloc[0].feature == "rw"


def test_missing_labels_falls_back_to_macrostates(tmp_path):
    run = _pilot_dir(tmp_path / "pilot3", labels=False)
    s = br.run(run, tmp_path / "out", n_boot=20, figure=False, verbose=False)
    assert s["sources"]["judge"]["note"] == "no judge labels yet"
    assert s["sources"]["macrostate"]["n_moves"] > 0


def test_nothing_to_analyse_returns_nonzero(tmp_path):
    run = _pilot_dir(tmp_path / "pilot3", labels=False, msm=False)
    assert br.main(["--run", str(run), "--out", str(tmp_path / "out"), "--no-figure"]) == 1


def test_missing_steps_is_a_clear_error(tmp_path, capsys):
    assert br.main(["--run", str(tmp_path / "nothing"), "--out", str(tmp_path / "out")]) == 2
    assert "no steps table" in capsys.readouterr().err


def test_figure_is_saved(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib

    matplotlib.use("Agg")
    run = _pilot_dir(tmp_path / "pilot3")
    s = br.run(run, tmp_path / "out", n_boot=50, verbose=False)
    pdf = tmp_path / "out" / "bridges.pdf"
    assert pdf.exists() and pdf.stat().st_size > 1000
    assert s["figure_source"] == "judge"
