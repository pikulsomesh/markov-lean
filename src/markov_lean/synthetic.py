"""Synthetic proof trajectories with a known hidden area per step.

Each proof is a run of a Markov chain over mathematical areas (MSC-like
two-digit codes).  The chain is "sticky": a proof tends to stay in one area
for a while and then jump.  At every step we write a fake Lean goal state whose
identifiers come mostly from the current area's vocabulary, mixed with shared
noise (logic and arithmetic notation every area uses), some leakage from other
areas, and some carry-over from the previous goal state.

The output is a ``steps`` table in the shape docs/schema.md defines, with
``source == "synthetic"`` and ``true_area`` filled in, so the Markov model
pipeline can be tested against a known answer before any real data exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A small set of MSC 2020 top-level codes and, for each, Lean-style namespaces
# and constants that a goal state in that area tends to mention.
AREA_VOCAB: dict[str, dict[str, list[str]]] = {
    "05": {  # combinatorics
        "folder": ["Combinatorics"],
        "names": [
            "Finset.card", "Finset.sum", "Finset.range", "Finset.powerset", "Finset.filter",
            "Nat.choose", "Nat.factorial", "SimpleGraph.Adj", "SimpleGraph.degree",
            "Finset.image", "Fintype.card", "Equiv.Perm", "Nat.descFactorial", "Finset.biUnion",
        ],
    },
    "11": {  # number theory
        "folder": ["NumberTheory"],
        "names": [
            "Nat.Prime", "Nat.gcd", "Nat.Coprime", "ZMod.val", "Int.emod", "Nat.ModEq",
            "padicValNat", "Nat.factorization", "Nat.totient", "legendreSym", "Int.gcd",
            "Nat.divisors", "Nat.primeFactors", "ZMod.pow_card",
        ],
    },
    "13": {  # commutative algebra
        "folder": ["RingTheory"],
        "names": [
            "Ideal.span", "Ideal.IsPrime", "Ideal.IsMaximal", "Polynomial.eval", "Polynomial.degree",
            "Polynomial.roots", "Localization.mk", "Submodule.span", "RingHom.ker", "IsNoetherianRing",
            "Ideal.Quotient.mk", "Polynomial.C", "Polynomial.X", "Algebra.adjoin",
        ],
    },
    "26": {  # real functions
        "folder": ["Analysis"],
        "names": [
            "Real.exp", "Real.log", "Real.sqrt", "deriv", "HasDerivAt", "intervalIntegral",
            "MonotoneOn", "ConvexOn", "Real.rpow", "Real.sin", "Real.cos", "DifferentiableAt",
            "StrictMonoOn", "Real.pi",
        ],
    },
    "54": {  # general topology
        "folder": ["Topology"],
        "names": [
            "IsOpen", "IsClosed", "IsCompact", "Continuous", "closure", "interior",
            "Filter.Tendsto", "nhds", "ConnectedSpace", "T2Space", "IsConnected",
            "Metric.ball", "TopologicalSpace.IsTopologicalBasis", "DenseRange",
        ],
    },
    "60": {  # probability
        "folder": ["Probability"],
        "names": [
            "MeasureTheory.Measure", "ProbabilityTheory.IndepFun", "MeasureTheory.integral",
            "ProbabilityTheory.variance", "ProbabilityTheory.condexp", "MeasureTheory.ae",
            "MeasurableSet", "ProbabilityTheory.Kernel", "MeasureTheory.IsProbabilityMeasure",
            "ENNReal.ofReal", "MeasureTheory.volume", "ProbabilityTheory.iIndepFun",
            "MeasureTheory.Integrable", "Measurable",
        ],
    },
}

# Lemma-name suffixes, as in `Nat.Prime.two_le`.  Appended to area names so the
# vocabulary is larger than the hand-written list.
SUFFIXES = [
    "mul", "add", "pos", "le", "lt", "comm", "eq", "zero", "one", "succ", "ne", "sub",
    "pow", "mono", "of_le", "mk", "iff", "symm", "trans", "nonneg", "injective",
]

# Tokens every area uses: logic, arithmetic notation, variable and hypothesis names.
SHARED_NOISE = [
    "Eq", "Ne", "And", "Or", "Iff", "Not", "Exists", "True", "False", "HAdd.hAdd", "HMul.hMul",
    "HSub.hSub", "HDiv.hDiv", "HPow.hPow", "LE.le", "LT.lt", "OfNat.ofNat", "Nat", "Int", "Real",
    "Set.mem", "Function.comp", "id", "x", "y", "n", "m", "k", "h", "hx", "hy", "hn", "f", "g", "s", "t",
]


@dataclass
class SyntheticConfig:
    """Knobs for the generator.  Defaults give a moderately hard problem."""

    n_proofs: int = 400
    min_steps: int = 6
    max_steps: int = 30
    areas: list[str] = field(default_factory=lambda: list(AREA_VOCAB))
    p_stay: float = 0.85          # probability the next step stays in the same area
    noise: float = 0.45           # fraction of tokens drawn from SHARED_NOISE
    leak: float = 0.10            # fraction drawn from a random other area's vocabulary
    carryover: float = 0.30       # fraction of the previous goal's tokens kept
    tokens_per_state: tuple[int, int] = (8, 20)
    seed: int = 0


def area_transition_matrix(n_areas: int, p_stay: float, rng: np.random.Generator) -> np.ndarray:
    """Row-stochastic matrix with ``p_stay`` on the diagonal and random off-diagonal jumps."""
    off = rng.dirichlet(np.ones(n_areas - 1), size=n_areas)
    T = np.zeros((n_areas, n_areas))
    for i in range(n_areas):
        T[i, np.arange(n_areas) != i] = (1.0 - p_stay) * off[i]
        T[i, i] = p_stay
    return T


def _area_tokens(area: str) -> list[str]:
    names = AREA_VOCAB[area]["names"]
    return names + [f"{n}.{s}" for n in names for s in SUFFIXES]


def _format_goal(tokens: list[str], rng: np.random.Generator) -> str:
    """Lay tokens out as hypotheses plus a target, roughly like Lean's pretty printer."""
    n_hyp = int(rng.integers(0, 3))
    cut = np.sort(rng.choice(np.arange(1, len(tokens)), size=min(n_hyp, len(tokens) - 1), replace=False))
    chunks = np.split(np.array(tokens, dtype=object), cut)
    lines = [f"h{i} : " + " ".join(c) for i, c in enumerate(chunks[:-1])]
    lines.append("⊢ " + " ".join(chunks[-1]))
    return "\n".join(lines)


def generate(config: SyntheticConfig | None = None, **overrides) -> tuple[pd.DataFrame, np.ndarray]:
    """Make a synthetic ``steps`` table.

    Returns ``(steps, T)`` where ``T`` is the true area transition matrix, with
    rows and columns in the order of ``config.areas``.
    """
    cfg = config or SyntheticConfig()
    if overrides:
        cfg = SyntheticConfig(**{**cfg.__dict__, **overrides})
    rng = np.random.default_rng(cfg.seed)
    areas = list(cfg.areas)
    A = len(areas)
    if A < 2:
        raise ValueError("need at least two areas")
    T = area_transition_matrix(A, cfg.p_stay, rng)
    vocab = {a: _area_tokens(a) for a in areas}
    lo, hi = cfg.tokens_per_state

    rows = []
    for p in range(cfg.n_proofs):
        n_steps = int(rng.integers(cfg.min_steps, cfg.max_steps + 1))
        a = int(rng.integers(A))
        home = areas[a]
        folder = AREA_VOCAB[home]["folder"][0] if home in AREA_VOCAB else "Misc"
        theorem = f"Synthetic.{folder}.thm_{p}"
        proof_id = f"synthetic:{theorem}"
        file_path = f"Mathlib/{folder}/Synthetic{p % 7}.lean"
        split = "train" if p % 10 < 8 else ("val" if p % 10 == 8 else "test")
        prev: list[str] = []
        states = []
        labels = []
        for _ in range(n_steps + 1):  # one extra goal so the last step has a state_after
            n_tok = int(rng.integers(lo, hi + 1))
            n_keep = min(len(prev), int(round(cfg.carryover * n_tok)))
            toks = list(rng.choice(prev, size=n_keep, replace=False)) if n_keep else []
            u = rng.random(n_tok - n_keep)
            for r in u:
                if r < cfg.noise:
                    toks.append(SHARED_NOISE[rng.integers(len(SHARED_NOISE))])
                elif r < cfg.noise + cfg.leak:
                    other = areas[int(rng.integers(A))]
                    toks.append(vocab[other][rng.integers(len(vocab[other]))])
                else:
                    v = vocab[areas[a]]
                    toks.append(v[rng.integers(len(v))])
            rng.shuffle(toks)
            states.append(_format_goal(toks, rng))
            labels.append(areas[a])
            prev = toks
            a = int(rng.choice(A, p=T[a]))
        for s in range(n_steps):
            rows.append({
                "proof_id": proof_id,
                "step": s,
                "source": "synthetic",
                "split": split,
                "theorem": theorem,
                "file_path": file_path,
                "tactic": "simp" if rng.random() < 0.5 else "rw [h]",
                "state_before": states[s],
                "state_after": "no goals" if s == n_steps - 1 else states[s + 1],
                "true_area": labels[s],
            })
    steps = pd.DataFrame(rows)
    steps["step"] = steps["step"].astype("int64")
    return steps, T
