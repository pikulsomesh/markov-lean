import json
from pathlib import Path

import pandas as pd
import pytest

from markov_lean import data

FIX = Path(__file__).parent / "fixtures" / "data"


def test_leandojo_steps_and_order():
    df = data.load_leandojo(FIX / "leandojo_benchmark_4", split="train")
    assert list(df.columns) == data.STEP_COLUMNS
    # the theorem with no traced tactics is dropped
    assert set(df["proof_id"]) == {"leandojo:Real.exp_pos'", "leandojo:SimpleGraph.foo"}
    exp = df[df.theorem == "Real.exp_pos'"]
    assert exp["step"].tolist() == [0, 1]
    assert exp["state_after"].iloc[-1] == "no goals"
    assert (df["source"] == "leandojo").all() and (df["split"] == "train").all()


def test_leandojo_min_steps_and_cap():
    df = data.load_leandojo(FIX / "leandojo_benchmark_4", split="train", min_steps=2)
    assert df["proof_id"].nunique() == 1
    df = data.load_leandojo(FIX / "leandojo_benchmark_4", split="train", max_theorems=1)
    assert df["proof_id"].nunique() == 1


def test_leandojo_rejects_bad_split():
    with pytest.raises(ValueError):
        data.load_leandojo(FIX / "leandojo_benchmark_4", split="dev")


def test_workbook_parquet_skips_disproved_and_empty():
    df = data.load_workbook(FIX / "lean_workbook" / "wkbk_1009.parquet")
    assert df["proof_id"].tolist() == ["workbook:lean_workbook_1"]
    assert df["step"].tolist() == [0]
    all_rows = data.load_workbook(FIX / "lean_workbook" / "wkbk_1009.parquet", proved_only=False)
    assert len(all_rows) == 2


def test_workbook_json_as_statements():
    df = data.load_workbook_statements(FIX / "lean_workbook" / "lean_workbook.json")
    assert df["proof_id"].tolist() == ["workbook:lean_workbook_0"]
    assert (df["step"] == -1).all()


def test_jsonl_statements_split_goal_and_duplicate_names():
    df = data.load_jsonl_statements(FIX / "deepseek_benchmarks" / "proofnet.jsonl", "proofnet")
    assert df["proof_id"].tolist() == ["proofnet:exercise_1_1", "proofnet:exercise_1_1#2", "proofnet:exercise_2_3"]
    assert df["proof_id"].is_unique
    assert df["split"].tolist() == ["val", "test", "test"]
    assert df["state_before"].iloc[0] == "x : ℝ\n⊢ x = x"  # goal state preferred
    assert df["state_before"].iloc[2].startswith("theorem exercise_2_3")  # falls back to statement


def test_statements_putnam():
    put = data.load_statements(FIX / "PutnamBench", "putnam", glob="lean4/**/*.lean")
    assert put["theorem"].tolist() == ["putnam_1962_a1"]
    stmt = put["state_before"].iloc[0]
    assert stmt.startswith("theorem putnam_1962_a1") and "convexHull" in stmt and "sorry" not in stmt
    assert (put["step"] == -1).all()


def test_parse_lean_statements_handles_attributes_and_lemmas():
    text = "@[simp] theorem foo (x : ℕ) : x = x := rfl\nlemma bar : True := by trivial\n"
    names = [n for n, _ in data.parse_lean_statements(text)]
    assert names == ["foo", "bar"]


def test_parquet_roundtrip(tmp_path):
    df = data.load_leandojo(FIX / "leandojo_benchmark_4", split="train")
    p = data.write_steps(df, tmp_path, "leandojo", "train")
    back = data.read_steps(p)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), back)


def test_build_all_skips_missing_sources(tmp_path, capsys):
    written = data.build_all(FIX, tmp_path / "out")
    names = sorted(p.name for p in written)
    assert names == [
        "leandojo_train.parquet",
        "leandojo_val.parquet",
        "minif2f_val.parquet",
        "proofnet_test.parquet",
        "proofnet_val.parquet",
        "putnam_test.parquet",
        "workbook_statements_train.parquet",
        "workbook_train.parquet",
    ]


def test_build_all_reports_missing(tmp_path, capsys):
    data.build_all(tmp_path / "empty", tmp_path / "out")
    out = capsys.readouterr().out
    for name in ("leandojo", "workbook steps", "minif2f", "proofnet", "putnam"):
        assert f"skip {name}" in out


def test_leandojo_repeated_full_names_get_unique_ids(tmp_path):
    # the real benchmark has e.g. two theorems named `aux` in different files
    def thm(name, file, n):
        tac = [{"tactic": f"t{i}", "state_before": f"s{i}", "state_after": f"s{i + 1}"} for i in range(n)]
        return {"full_name": name, "file_path": file, "traced_tactics": tac}

    d = tmp_path / "random"
    d.mkdir()
    (d / "train.json").write_text(json.dumps([
        thm("aux", "Mathlib/A.lean", 2), thm("other", "Mathlib/B.lean", 1), thm("aux", "Mathlib/C.lean", 3),
    ]))
    df = data.load_leandojo(tmp_path, split="train")
    assert not df.duplicated(["proof_id", "step"]).any()
    ids = df.groupby("proof_id")["file_path"].nunique()
    assert (ids == 1).all()
    assert set(df["proof_id"]) == {"leandojo:aux", "leandojo:other", "leandojo:aux#2"}
    assert len(df[df["proof_id"] == "leandojo:aux#2"]) == 3
    # ids do not change when min_steps drops a theorem
    df2 = data.load_leandojo(tmp_path, split="train", min_steps=3)
    assert set(df2["proof_id"]) == {"leandojo:aux#2"}
