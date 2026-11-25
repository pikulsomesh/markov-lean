"""Theorem statements must load whole: a ``let x := ...`` inside a statement is not its end.

The files in fixtures/data/putnam_let/ are real PutnamBench problems (trishullab/PutnamBench
at a635813, Apache 2.0). The first four were cut short at their first ``:=`` before the
statement parser learned about ``let``/``letI`` bindings; putnam_2022_a4 has a block
comment between its binders.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from markov_lean import data, gold
from markov_lean import search_eval as se

LET_DIR = Path(__file__).parent / "fixtures" / "data" / "putnam_let"

# name -> text the full statement must end with (just before its closing ``:=``)
REAL = {
    "putnam_1985_a1": "= 2 ^ a * 3 ^ b * 5 ^ c * 7 ^ d",
    "putnam_1984_a5": "((a)! * (b)! * (c)! * (d)! : ℝ) / (n)!)",
    "putnam_1965_b4": "∀ x ∈ s, Tendsto (fun n ↦ f n x) atTop (𝓝 (g x))",
    "putnam_2023_a4": "∃ a : Fin 12 → ℤ, ‖∑ i, a i • v i - w‖ < ε",
    "putnam_2022_a4": "∫ ω, S ω ∂(ℙ : Measure Ω) = putnam_2022_a4_solution",
}


def _text(name: str) -> str:
    return (LET_DIR / f"{name}.lean").read_text()


@pytest.mark.parametrize("name", sorted(REAL))
def test_real_putnam_statements_load_whole(name):
    (got_name, stmt), = data.parse_lean_statements(_text(name))
    assert got_name == name
    assert stmt.startswith(f"theorem {name}")
    assert stmt.endswith(REAL[name])
    assert "sorry" not in stmt


def test_let_bindings_are_kept():
    stmt = dict(data.parse_lean_statements(_text("putnam_1985_a1")))["putnam_1985_a1"]
    assert "let (a, b, c, d) := putnam_1985_a1_solution;" in stmt
    stmt = dict(data.parse_lean_statements(_text("putnam_2023_a4")))["putnam_2023_a4"]
    assert stmt.count("letI") == 3 and "(w : EuclideanSpace ℝ (Fin 3))" in stmt


def test_block_comment_between_binders_is_kept():
    stmt = dict(data.parse_lean_statements(_text("putnam_2022_a4")))["putnam_2022_a4"]
    assert "If there is an `l` such that" in stmt and "(hS : ∀ ω" in stmt


@pytest.mark.parametrize("name", sorted(REAL))
def test_search_eval_and_gold_read_the_same_statement(name):
    text = _text(name)
    stmt = dict(data.parse_lean_statements(text))[name]
    assert se.parse_putnam_file(text, name)["statement"] == stmt + " :="
    assert gold.putnam_statement(text, name) == stmt


def test_search_eval_loads_the_let_fixtures_whole():
    thms = se.load_benchmark("putnam", {"path": str(LET_DIR), "format": "putnam", "glob": "*.lean"})
    assert sorted(t.name for t in thms) == sorted(REAL)
    for t in thms:
        assert t.statement.endswith(REAL[t.name] + " :=")


def test_have_defaults_and_comments():
    text = """\
-- theorem commented_out : True := trivial
/-- doc with := inside -/
theorem t1 (n : ℕ := 3) {f : ℕ → ℕ} (h : f = fun x => (let y := x; y + 1)) :
    have k : ℕ := n + 1   -- a comment with := in it
    f k = k + 1 := by
  sorry
lemma t2 : "a := b".length = 6 := rfl
theorem t3 (x : ℕ) :
    x = x -- trailing comment
:= rfl
"""
    got = dict(data.parse_lean_statements(text))
    assert list(got) == ["t1", "t2", "t3"]
    assert got["t1"].endswith("f k = k + 1") and "have k : ℕ := n + 1" in got["t1"]
    assert got["t2"] == 'lemma t2 : "a := b".length = 6'
    assert got["t3"].endswith("x = x -- trailing comment")
    # a statement ending in a line comment gets its ``:=`` on the next line
    p = se.parse_putnam_file("import Mathlib\n\n" + text.split("lemma t2")[1].split("\n", 1)[1], "t3")
    assert p["statement"].endswith("-- trailing comment\n:=")


def test_unclosed_statement_stops_at_next_declaration():
    text = "theorem a : True\ntheorem b : True := trivial\n"
    assert data.parse_lean_statements(text) == [("a", "theorem a : True"), ("b", "theorem b : True")]


def test_mask_comments_keeps_positions():
    text = 'a -- x := y\n/- /- nested -/ := -/ b "s := t" c'
    masked = data.mask_comments(text)
    assert len(masked) == len(text) and masked.count("\n") == 1
    assert ":=" not in masked and masked.split() == ["a", "b", '"', '"', "c"]
