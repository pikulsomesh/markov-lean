"""MSC 2020 top-level classes (the two-digit level).

Source: Mathematics Subject Classification 2020, published jointly by
Mathematical Reviews and zbMATH Open, https://msc2020.org/ (MSC_2020.pdf,
CC BY-NC-SA). MSC2020 made no changes at the two-digit level and has 63
two-digit classes (msc2020.org front page).

How this list was checked (2026-11-25):
  * Codes and titles 00 to 20 were compared line by line with the official
    MSC_2020.pdf from msc2020.org.
  * The set of 63 codes was compared with the MathSciNet MSC search page
    (https://mathscinet.ams.org/mathscinet/searchMSC.html), keeping only the
    codes it does not mark "not currently valid" and dropping 36 and 79,
    which are pre-2000 classes that MSC2020 does not contain.
  * Titles 22 to 97 follow MSC2020 wording (e.g. 39 "Difference and
    functional equations", 42 "Harmonic analysis on Euclidean spaces",
    91 "... finance, and other social and behavioral sciences", 94
    "Information and communication theory, circuits"). The official PDF text
    could not be read past 20-XX from the cloud environment, so re-check
    these titles against MSC_2020.pdf once on the Mac Mini
    (``tests/test_judge.py::test_msc_titles_against_official_csv`` does it
    automatically when ``MSC2020_CSV`` points at the official CSV).
Only the codes matter for the ``labels`` table; titles are for prompts.
"""
from __future__ import annotations

import re

MSC2020_TOP: dict[str, str] = {
    "00": "General and overarching topics; collections",
    "01": "History and biography",
    "03": "Mathematical logic and foundations",
    "05": "Combinatorics",
    "06": "Order, lattices, ordered algebraic structures",
    "08": "General algebraic systems",
    "11": "Number theory",
    "12": "Field theory and polynomials",
    "13": "Commutative algebra",
    "14": "Algebraic geometry",
    "15": "Linear and multilinear algebra; matrix theory",
    "16": "Associative rings and algebras",
    "17": "Nonassociative rings and algebras",
    "18": "Category theory; homological algebra",
    "19": "K-theory",
    "20": "Group theory and generalizations",
    "22": "Topological groups, Lie groups",
    "26": "Real functions",
    "28": "Measure and integration",
    "30": "Functions of a complex variable",
    "31": "Potential theory",
    "32": "Several complex variables and analytic spaces",
    "33": "Special functions",
    "34": "Ordinary differential equations",
    "35": "Partial differential equations",
    "37": "Dynamical systems and ergodic theory",
    "39": "Difference and functional equations",
    "40": "Sequences, series, summability",
    "41": "Approximations and expansions",
    "42": "Harmonic analysis on Euclidean spaces",
    "43": "Abstract harmonic analysis",
    "44": "Integral transforms, operational calculus",
    "45": "Integral equations",
    "46": "Functional analysis",
    "47": "Operator theory",
    "49": "Calculus of variations and optimal control; optimization",
    "51": "Geometry",
    "52": "Convex and discrete geometry",
    "53": "Differential geometry",
    "54": "General topology",
    "55": "Algebraic topology",
    "57": "Manifolds and cell complexes",
    "58": "Global analysis, analysis on manifolds",
    "60": "Probability theory and stochastic processes",
    "62": "Statistics",
    "65": "Numerical analysis",
    "68": "Computer science",
    "70": "Mechanics of particles and systems",
    "74": "Mechanics of deformable solids",
    "76": "Fluid mechanics",
    "78": "Optics, electromagnetic theory",
    "80": "Classical thermodynamics, heat transfer",
    "81": "Quantum theory",
    "82": "Statistical mechanics, structure of matter",
    "83": "Relativity and gravitational theory",
    "85": "Astronomy and astrophysics",
    "86": "Geophysics",
    "90": "Operations research, mathematical programming",
    "91": "Game theory, economics, finance, and other social and behavioral sciences",
    "92": "Biology and other natural sciences",
    "93": "Systems theory; control",
    "94": "Information and communication theory, circuits",
    "97": "Mathematics education",
}

CODES: tuple[str, ...] = tuple(MSC2020_TOP)

# Areas that carry most of Mathlib and the competition benchmarks
# (miniF2F, ProofNet, Lean Workbook, PutnamBench). Shown first in prompts as a
# hint; the judge may still answer any of the 63 codes.
LEAN_COMMON: tuple[str, ...] = (
    "03", "05", "06", "11", "12", "13", "14", "15", "16", "18", "20", "22",
    "26", "28", "30", "40", "46", "47", "51", "52", "54", "55", "57", "60", "68",
)

_CODE_RE = re.compile(r"^\s*(\d{1,2})(?:[-A-Za-z0-9]*)\s*$")


def normalize(code: object) -> str | None:
    """Return the two-digit top-level code for ``code`` or None if invalid.

    Accepts "26", "26-XX", "26A42", 26 and "5" (-> "05").
    """
    if code is None:
        return None
    if isinstance(code, float):  # pandas may read "20" back as 20.0
        if code != code or not code.is_integer():
            return None
        code = int(code)
    if isinstance(code, int):
        s = f"{code:02d}"
    else:
        m = _CODE_RE.match(str(code))
        if not m:
            return None
        s = m.group(1).zfill(2)
    return s if s in MSC2020_TOP else None


def title(code: str) -> str:
    c = normalize(code)
    if c is None:
        raise KeyError(f"not an MSC 2020 top-level code: {code!r}")
    return MSC2020_TOP[c]


def prompt_listing() -> str:
    """The code list as it appears in judge prompts: common Lean areas first."""
    common = [f"{c} {MSC2020_TOP[c]}" for c in LEAN_COMMON]
    rest = [f"{c} {t}" for c, t in MSC2020_TOP.items() if c not in LEAN_COMMON]
    return (
        "Most common in Lean/Mathlib:\n" + "\n".join(common)
        + "\n\nAll other MSC 2020 top-level classes:\n" + "\n".join(rest)
    )
