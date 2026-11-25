import Mathlib

/-! Smoke test: if this file builds, Lean and Mathlib are set up correctly. -/

example (a b : ℕ) : a + b = b + a := Nat.add_comm a b
