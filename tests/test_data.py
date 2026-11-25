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


def test_workbook_skips_disproved_and_empty():
    df = data.load_workbook(FIX / "lean_workbook" / "lean_workbook.json")
    assert df["proof_id"].tolist() == ["workbook:lean_workbook_1"]
    assert df["step"].tolist() == [0]
    all_rows = data.load_workbook(FIX / "lean_workbook" / "lean_workbook.json", proved_only=False)
    assert len(all_rows) == 2


def test_statements_putnam_and_minif2f():
    put = data.load_statements(FIX / "PutnamBench", "putnam", glob="lean4/**/*.lean")
    assert put["theorem"].tolist() == ["putnam_1962_a1"]
    stmt = put["state_before"].iloc[0]
    assert stmt.startswith("theorem putnam_1962_a1") and "convexHull" in stmt and "sorry" not in stmt
    assert (put["step"] == -1).all()
    mf = data.load_statements(FIX / "miniF2F-lean4", "minif2f")
    assert mf["split"].tolist() == ["val"]


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
        "putnam_test.parquet",
        "workbook_train.parquet",
    ]
    assert "skip proofnet" in capsys.readouterr().out
