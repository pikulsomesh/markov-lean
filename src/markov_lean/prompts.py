"""Versions of the MSC judge prompt.

``v1`` is the prompt the API judges use (``judge.SYSTEM_PROMPT``, ``judge.render_user``,
``judge.output_schema``). It is unchanged and stays the default everywhere, so labels made
before this module existed, and every Claude Haiku/Sonnet run, are still comparable.

``v2`` is an opt-in prompt written for the small local judge (Qwen 2.5 7B through Ollama),
which on the gold set over-called 26 Real functions and 03 Logic and missed 40 Sequences,
05 Combinatorics and 20 Group theory with ``v1``. Reading ``v1`` for causes:

* it is written for proof steps first and the theorem second, while the gold set scores
  the theorem's area and each gold problem is a one-step "proof" whose tactic is ``sorry``
  and whose goal repeats the statement;
* it gives the 63 codes as bare titles, with no boundary between neighbouring classes
  (26 against 40 for a real sequence, 05 against 03 for counting sets, 20 against 11 for
  a group of order 351), and lists 03 first;
* the only guidance about 03 is a step-level rule ("bookkeeping goals are not 03"), so
  nothing stops a whole statement full of ``∀``, ``∃``, ``Set`` and ``↔ ..._solution`` from
  being called logic;
* the model answers with the code straight away, with nothing to say first what the
  statement is about.

``v2`` changes the prompt, not the parsing (which had no bug: the codes are constrained to
the 63 MSC values by the schema and nothing falls back to a default class). It asks for the
statement's area first, after a short ``topic`` field that names the objects the statement
is about; it gives definitions with Lean signals and explicit tie-breaks for the classes
common in Lean; it adds twelve short examples written for this prompt (a test checks that
none is close to a gold-set statement); and it shows a ``sorry`` step without repeating the
goal. Its output keeps every field of ``v1`` (``topic`` is extra and ignored downstream),
so ``judge.rows_from_output`` reads both.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from markov_lean import judge as J
from markov_lean import msc

DEFAULT_VERSION = "v1"

V2_EXAMPLES: tuple[tuple[str, str, str], ...] = (
    # (statement, code, topic); written for this prompt, not taken from any benchmark
    ("G : Type, [Group G], (a b : G), (h : a * b * a⁻¹ = b ^ 2), (ha : orderOf a = 3) ⊢ b ^ 7 = 1",
     "20", "elements of a group and their orders"),
    ("(n : ℕ) ⊢ Nat.gcd (n ^ 2 + 1) (n + 1) ∣ 2", "11", "gcd of integers"),
    ("⊢ {n : ℤ | ∃ a b : ℤ, n = a ^ 2 + b ^ 2 ∧ n % 4 = 3} = ∅", "11",
     "integers that are sums of two squares (a set, but about integers)"),
    ("(G : SimpleGraph (Fin 6)) ⊢ ∃ s : Finset (Fin 6), s.card = 3 ∧ (G.IsClique s ∨ Gᶜ.IsClique s)",
     "05", "cliques in a graph on six vertices"),
    ("⊢ {f : Fin 5 → Fin 5 | Function.Bijective f ∧ ∀ i, f i ≠ i}.ncard = 44", "05",
     "count of permutations without fixed points"),
    ("(a : ℕ → ℝ) (h0 : a 0 = 1) (h : ∀ n, a (n + 1) = (a n + 2 / a n) / 2) ⊢ Tendsto a atTop (𝓝 √2)",
     "40", "limit of a recursive real sequence"),
    ("(f : ℝ → ℝ) (hf : Differentiable ℝ f) (h : ∀ x, deriv f x ≤ 0) ⊢ Antitone f", "26",
     "derivative and monotonicity of a real function"),
    ("(f : ℂ → ℂ) (hf : Differentiable ℂ f) (h : ∀ z, Complex.abs (f z) ≤ 1) ⊢ ∀ z w, f z = f w", "30",
     "bounded entire function"),
    ("K : Type, [Field K], [Fintype K] ⊢ ∃ p n : ℕ, p.Prime ∧ Fintype.card K = p ^ n", "12",
     "size of a finite field"),
    ("(A : Matrix (Fin 3) (Fin 3) ℝ) (h : A ^ 2 = 0) ⊢ A.det = 0", "15", "determinant of a matrix"),
    ("X : Type, [TopologicalSpace X], [T2Space X], (K : Set X), (hK : IsCompact K) ⊢ IsClosed K", "54",
     "compact sets in a Hausdorff space"),
    ("Ω : Type, [MeasureSpace Ω], [IsProbabilityMeasure ℙ], (X : Ω → ℝ), (h : ∀ ω, 0 ≤ X ω), (a : ℝ), "
     "(ha : 0 < a) ⊢ (ℙ {ω | a ≤ X ω}).toReal ≤ (∫ ω, X ω) / a", "60", "tail probability of a random variable"),
    ("⊢ ¬ ∃ f : ℕ → Set ℕ, Function.Surjective f", "03", "cardinality of the power set of ℕ"),
)


def _examples() -> str:
    return "\n".join(f"- `{s}` -> {c} (topic: {t})" for s, c, t in V2_EXAMPLES)


def _all_codes() -> str:
    return "\n".join(f"{c} {t}" for c, t in msc.MSC2020_TOP.items())


SYSTEM_PROMPT_V2 = """You classify Lean 4 theorems by area of mathematics, using the two-digit \
top-level codes of the Mathematics Subject Classification 2020 (MSC 2020).

Work in this order.
1. Read the theorem's statement. In `topic`, name in at most 12 words the mathematical objects \
the statement is about, for example "finite group of order 6", "real sequence and its limit", \
"subsets of a finite set and their number". Name objects, not proof methods.
2. Give `proof_msc`, the area of the statement, using the guide below.
3. Label each proof step shown with the area of its goal state, using the same guide. A step can \
sit in a different area from the theorem; label what the step's goal is about. A goal that is only \
bookkeeping (introducing variables, splitting a conjunction) takes the theorem's area. Classify \
the mathematics, not the tactic: `simp`, `ring` or `omega` say nothing about the area.

The kind of object the statement is about decides the area, not the notation used to state it. \
Every Lean statement uses ∀, ∃, ↔, sets, functions and types; that alone never makes it logic (03). \
A placeholder such as `putnam_1999_b2_solution` or `answer` is an unknown value: classify by what \
it is compared with.

Area guide for the codes most common in Lean:
- 11 Number theory: properties of integers and rationals: divisibility, primes, gcd, congruences \
(`[MOD n]`, `ZMod`, `%`), digits, integer solutions of equations, arithmetic functions, irrationality.
- 05 Combinatorics: counting (`Finset.card`, `ncard` or `Nat.choose` used as a count), \
arrangements, permutations or subsets as things to count, partitions of a finite set, graphs \
(`SimpleGraph`), tournaments, colourings, games and strategies, tilings, pigeonhole arguments.
- 20 Group theory: groups and their elements, subgroups, cosets, normal subgroups, quotient groups, \
group homomorphisms (`G →* H`), orders of elements and of groups, Sylow subgroups, solvable or simple \
groups, permutation groups (`Equiv.Perm`), group actions. Still 20 when the question is about the \
order of the group as a number (`card G = p * m`) or about whether the group is commutative.
- 12 Field theory and polynomials: fields, field extensions, algebraic numbers, roots and \
irreducibility of polynomials over a field, characteristic p, finite fields, Galois theory.
- 13 Commutative algebra, 16 Associative rings: rings, ideals and modules over a ring (13 when the \
ring is commutative, 16 when it need not be).
- 15 Linear algebra: vector spaces, linear maps (`→ₗ`), subspaces (`Submodule` over a field), bases, \
dimension, matrices, determinants, traces, eigenvalues, finite-dimensional inner product spaces.
- 40 Sequences, series, summability: a sequence indexed by n : ℕ (`ℕ → ℝ`, `ℕ → ℂ`, `ℕ → X`) and \
its limit (`Tendsto _ atTop`, `CauchySeq`, recurrences, bounded monotone sequences), and infinite \
series (partial sums `∑ i ∈ range n` as n grows, `Summable`, `tsum`), convergence tests.
- 26 Real functions: functions of a real variable (`ℝ → ℝ`): derivatives (`deriv`), \
differentiability, continuity on an interval, monotonicity, convexity, the mean value theorem, \
inequalities between real functions, Riemann integrals. Not for a statement that only mentions real \
numbers, and not for a sequence ℕ → ℝ, which is 40.
- 30 Functions of a complex variable: holomorphic or analytic functions of ℂ \
(`DifferentiableOn ℂ`, `AnalyticAt`), complex power series, contour integrals, zeros and poles.
- 28 Measure and integration: measures, measurable sets and functions, Lebesgue integrals.
- 60 Probability: probability measures (`IsProbabilityMeasure`, `ℙ`), random variables, \
expectation, independence, random experiments, stochastic processes.
- 54 General topology: topological and metric spaces in general (`TopologicalSpace`, \
`MetricSpace`): open, closed, compact and connected sets, Hausdorff and other separation \
properties, continuous maps between spaces, product and quotient topologies.
- 55 Algebraic topology: fundamental groups, covering spaces, homotopy, homology.
- 46 Functional analysis: normed and Banach spaces of infinite dimension, Hilbert spaces, spaces \
of functions.
- 51 Geometry, 52 Convex and discrete geometry: points, lines, triangles, circles and polygons \
(51); convex sets, lattice points, packings and coverings (52).
- 03 Mathematical logic and foundations: only when the statement is about logic or set theory \
itself: countable against uncountable sets, cardinal and ordinal numbers, the axiom of choice, \
formal languages and models, computability.

When two codes seem to fit, the more specific kind of object wins: a group of order 12 is 20, not \
11; a sequence of reals is 40, not 26; a count of subsets or permutations is 05, not 03; a \
polynomial over a field asked about its roots or irreducibility is 12.

Examples, written for illustration (statement -> code):
""" + _examples() + """

For every label give a confidence between 0 and 1 for how sure you are of the area. Use the full \
range: about 0.9 when the area is unmistakable, about 0.5 when two areas are equally plausible.

Any MSC 2020 top-level code may be answered; the full list:
""" + _all_codes()


def output_schema_v2() -> dict:
    """``v1``'s schema with a short ``topic`` first and the statement's code before the steps,
    so the model names the objects, then classifies the statement, then the steps (Ollama's
    structured outputs keep the property order)."""
    code = {"type": "string", "enum": list(msc.CODES)}
    step = J.output_schema()["properties"]["steps"]
    return {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "proof_msc": code,
            "proof_confidence": {"type": "number"},
            "steps": step,
        },
        "required": ["topic", "proof_msc", "proof_confidence", "steps"],
        "additionalProperties": False,
    }


def render_user_v2(proof: J.Proof) -> str:
    """Like ``judge.render_user``, but a step whose goal is the statement is not printed twice,
    and a proof with no real steps (the gold set's single ``sorry``) says so plainly."""
    parts = [f"Theorem: {proof.theorem}", "", "Statement:", proof.statement, ""]
    if not proof.steps:
        parts.append("No proof steps are given. Return an empty `steps` list and label the statement.")
        return "\n".join(parts)
    if all(s.tactic.strip() == "sorry" for s in proof.steps):
        parts.append("No proof is given. Label the statement, and give each step below the statement's area.")
        for s in proof.steps:
            parts.append(f"[step {s.step}] tactic: sorry")
        return "\n".join(parts)
    if proof.n_steps_total > len(proof.steps):
        parts.append(f"The proof has {proof.n_steps_total} steps; {len(proof.steps)} of them are shown.")
    parts.append("Proof steps. Label each one, using the step numbers given:")
    for s in proof.steps:
        goal = "(the statement above)" if s.state_before.strip() == proof.statement.strip() else s.state_before
        parts += [f"[step {s.step}] tactic: {s.tactic}", "goal before:", goal, ""]
    parts += ["State after the last shown step:", proof.steps[-1].state_after]
    return "\n".join(parts)


@dataclass(frozen=True)
class PromptVersion:
    name: str
    system: str
    render_user: Callable[[J.Proof], str]
    schema: Callable[[], dict]
    extra_output_tokens: int = 0  # room for fields v1 does not have


PROMPTS: dict[str, PromptVersion] = {
    "v1": PromptVersion("v1", J.SYSTEM_PROMPT, J.render_user, J.output_schema),
    "v2": PromptVersion("v2", SYSTEM_PROMPT_V2, render_user_v2, output_schema_v2, extra_output_tokens=48),
}


def get(version: str) -> PromptVersion:
    try:
        return PROMPTS[version]
    except KeyError:
        raise ValueError(f"unknown prompt version {version!r}; known: {', '.join(PROMPTS)}") from None
