import numpy as np
import pandas as pd
import pytest

from markov_lean import evaluate as ev


def test_agreement_perfect_and_relabelled():
    s = ev.agreement([0, 0, 1, 1, 2], ["a", "a", "b", "b", "c"])
    assert s["nmi"] == pytest.approx(1.0) and s["ari"] == pytest.approx(1.0) and s["n"] == 5


def test_agreement_skips_unassigned_and_missing():
    s = ev.agreement([0, -1, 1, 1], ["a", "a", None, "b"])
    assert s["n"] == 2 and s["coverage"] == pytest.approx(0.5)


def test_score_assignments_broadcasts_whole_proof_labels():
    a = pd.DataFrame({"proof_id": ["p", "p", "q"], "step": [0, 1, 0], "macrostate": [0, 0, 1]})
    labels = pd.DataFrame({"proof_id": ["p", "q"], "step": [-1, 0], "msc": ["11", "26"]})
    s = ev.score_assignments(a, labels, "msc")
    assert s["n"] == 3 and s["nmi"] == pytest.approx(1.0)


def test_next_state_topk_and_baselines():
    a = pd.DataFrame({"proof_id": ["p"] * 4, "step": range(4), "macrostate": [0, 1, 0, 1]})
    T = np.array([[0.1, 0.9], [0.8, 0.2]])
    r = ev.next_state_topk(a, T, k=1)
    assert r["topk"] == 1.0 and r["stay"] == 0.0 and r["n"] == 3
    assert ev.next_state_topk(a, T, k=2)["topk"] == 1.0


def test_next_area_topk_maps_macrostates_to_labels():
    a = pd.DataFrame({"proof_id": ["p"] * 4, "step": range(4), "macrostate": [0, 0, 1, 1]})
    lab = pd.DataFrame({"proof_id": ["p"] * 4, "step": range(4), "area": ["11", "11", "26", "26"]})
    T = np.array([[0.6, 0.4], [0.3, 0.7]])
    r = ev.next_area_topk(a, lab, "area", T, k=1)
    assert r["macrostate_to_label"] == {0: "11", 1: "26"}
    assert r["topk"] == pytest.approx(2 / 3) and r["stay"] == pytest.approx(2 / 3)


def test_kappa_and_judge_kappa():
    df = pd.DataFrame({"a": ["11", "26", "11", None], "b": ["11", "26", "26", "11"]})
    k = ev.kappa(df, "a", "b")
    assert k["n"] == 3 and k["raw_agreement"] == pytest.approx(2 / 3)
    labels = pd.DataFrame({
        "proof_id": ["p", "p", "q", "q"], "step": [0, 0, 0, 0],
        "judge": ["haiku", "sonnet", "haiku", "sonnet"], "msc": ["11", "11", "26", "26"],
    })
    assert ev.judge_kappa(labels, "haiku", "sonnet")["kappa"] == pytest.approx(1.0)


@pytest.mark.parametrize("path,expected", [
    ("Mathlib/NumberTheory/Padics/PadicVal.lean", "NumberTheory"),
    (".lake/packages/mathlib/Mathlib/Topology/Basic.lean", "Topology"),
    ("Mathlib/Order.lean", "Order"),
    ("", None),
    (None, None),
    ("MiniF2F/Test.lean", None),
])
def test_folder_label(path, expected):
    assert ev.folder_label(path) == expected


def test_folder_baseline():
    steps = pd.DataFrame({
        "file_path": ["Mathlib/Analysis/A.lean"] * 2 + ["Mathlib/NumberTheory/B.lean"] * 2,
        "true_area": ["26", "26", "11", "11"],
    })
    assert ev.folder_baseline(steps, "true_area")["nmi"] == pytest.approx(1.0)
