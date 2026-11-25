"""A gold set of benchmark problems with known MSC 2020 top-level codes.

The judges label proof steps with MSC codes, and their agreement says nothing
about whether those labels are right. This module builds a set of Lean
problems whose area is known from the benchmark's own metadata, so a judge can
be scored against it (`markov-lean judge-accuracy`, in ``judge_accuracy.py``).

No label here comes from a language model or from reading a statement. Every
label is a lookup on metadata the benchmark authors wrote: a miniF2F problem's
name prefix (taken from the MATH dataset subject or the competition), a
PutnamBench problem's topic tags, or the textbook and chapter a ProofNet
exercise comes from. The lookup tables are below, as data, and
``docs/judge_accuracy.md`` gives the reasoning for each row.

Each problem gets one of three statuses:

``strict``
    The metadata names one MSC top-level class. Scored by exact match, and
    used for accuracy, macro-F1 and the confusion matrix.
``family``
    The metadata names a group of neighbouring classes but not one class
    (Putnam "analysis" could be 26, 30 or 40). Scored only by whether the
    judge's code is in the group ("family accuracy"); strict rows count in
    family accuracy too, with a group of one.
``excluded``
    The metadata gives no area (an AMC problem), or an area with no MSC
    counterpart (school "algebra"). Never scored; the reason is kept.

Sources, pinned to the commits checked on 2026-11-25 (``scripts/fetch_gold_sources.sh``
downloads exactly these):

* miniF2F, Lean 4 copy with goal states: DeepSeek-Prover-V1.5
  ``datasets/minif2f.jsonl`` (MIT).
* ProofNet, Lean 4 copy with goal states: DeepSeek-Prover-V1.5
  ``datasets/proofnet.jsonl`` (MIT); book and chapter ids from the original
  ProofNet ``benchmark/{valid,test}.jsonl`` (MIT), matched by split, exercise
  name and informal statement.
* PutnamBench: ``informal/putnam.json`` tags and ``lean4/src/*.lean``
  statements (Apache 2.0).
* Proofs: DeepSeek-Prover-V2 ``minif2f-solutions.zip``, whole proofs for 438 of
  the 488 miniF2F problems. They are read at ``steps`` time and never stored
  in the gold table, since that archive states no licence for its proofs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from markov_lean import msc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLD = ROOT / "data" / "gold" / "msc_gold.csv"
DEFAULT_RAW = ROOT / "data" / "raw" / "gold"

PINS = {
    "deepseek_prover_v1_5": "2c4ba9119eef74d0d611f494261b2c5bae98c69a",
    "proofnet": "509ad79710ed4f46ff5c282ed5640c1aa9ac3f30",
    "putnambench": "a635813f5edf297d928f9bd5ef8324a26bdde09c",
    "deepseek_prover_v2": "e598a57ea3284997d4a2a168a069fdd5064afbc8",
}

GOLD_COLUMNS = [
    "id", "source", "split", "name", "source_label", "status", "msc", "msc_family", "rule", "reason",
    "statement",
]

# The MSC analysis block, 26 to 49, as the codes MSC 2020 actually has in that range.
ANALYSIS_BLOCK: tuple[str, ...] = tuple(c for c in msc.CODES if 26 <= int(c) <= 49)


@dataclass(frozen=True)
class Rule:
    """One row of a lookup table. ``strict`` is a single code, ``family`` a group."""

    rule: str
    strict: str | None = None
    family: tuple[str, ...] | None = None
    reason: str = ""

    def __post_init__(self):
        for c in ([self.strict] if self.strict else []) + list(self.family or ()):
            if msc.normalize(c) != c:
                raise ValueError(f"{self.rule}: {c!r} is not a two-digit MSC 2020 code")

    @property
    def status(self) -> str:
        if self.strict:
            return "strict"
        return "family" if self.family else "excluded"

    @property
    def codes(self) -> tuple[str, ...]:
        if self.strict:
            return (self.strict,)
        return tuple(self.family or ())


def _x(rule: str, reason: str) -> Rule:
    return Rule(rule, reason=reason)


# ---------------------------------------------------------------- miniF2F, by name prefix
# miniF2F names come from the MATH dataset subject (mathd_algebra, mathd_numbertheory),
# from the competition (amc12a, aime, imo) or from the authors' own exercises
# (algebra_*, numbertheory_*, induction_*). Only number theory names one MSC class.
MINIF2F_RULES: tuple[tuple[str, Rule], ...] = (
    (r"^mathd_numbertheory_", Rule("minif2f:mathd_numbertheory", "11")),
    (r"^numbertheory_", Rule("minif2f:numbertheory", "11")),
    (r"^mathd_algebra_", _x("minif2f:mathd_algebra", "school algebra has no single MSC class")),
    (r"^algebra_", _x("minif2f:algebra", "school algebra has no single MSC class")),
    (r"^induction_", _x("minif2f:induction", "the name gives a proof method, not an area")),
    (r"^(amc|aime|imo)", _x("minif2f:competition", "the competition name gives no area")),
)

# ---------------------------------------------------------------- PutnamBench, by topic tag
PUTNAM_TAGS: dict[str, Rule] = {
    "number_theory": Rule("putnam:number_theory", "11"),
    "combinatorics": Rule("putnam:combinatorics", "05"),
    "linear_algebra": Rule("putnam:linear_algebra", "15"),
    "probability": Rule("putnam:probability", "60"),
    "set_theory": Rule("putnam:set_theory", "03"),
    # Putnam geometry is Euclidean (51) or convex and discrete (52), often within one year
    "geometry": Rule("putnam:geometry", family=("51", "52")),
    # real functions, series, complex variables, integrals, ODEs, functional equations ...
    "analysis": Rule("putnam:analysis", family=ANALYSIS_BLOCK),
    # groups, rings, fields
    "abstract_algebra": Rule("putnam:abstract_algebra", family=("08", "12", "13", "16", "17", "20")),
    # polynomials, inequalities, sequences, functional equations: no MSC group covers it
    "algebra": _x("putnam:algebra", "Putnam 'algebra' spans polynomials, inequalities and sequences"),
}

# ---------------------------------------------------------------- ProofNet, by book and chapter
# Keys: book -> list of (first chapter, last chapter, rule). The chapter is the first number
# in the exercise id (for Munkres it is the section). Ranges not listed are excluded.
_ALL = (0, 10_000)
PROOFNET_BOOKS: dict[str, tuple[tuple[int, int, Rule], ...]] = {
    "Axler": ((*_ALL, Rule("proofnet:axler", "15")),),                        # Linear Algebra Done Right
    "Ireland-Rosen": ((*_ALL, Rule("proofnet:ireland_rosen", "11")),),        # Classical Intro. to Modern NT
    "Shakarchi": ((*_ALL, Rule("proofnet:shakarchi", "30")),),                # Complex Analysis
    "Munkres": (                                                              # Topology, 2nd ed.
        (12, 50, Rule("proofnet:munkres_part1", "54")),                       # Part I, general topology
        (51, 10_000, Rule("proofnet:munkres_part2", "55")),                   # Part II, algebraic topology
    ),
    "Dummit-Foote": (                                                         # Abstract Algebra, 3rd ed.
        (1, 6, Rule("proofnet:dummit_foote_groups", "20")),                   # Part I, group theory
        (7, 9, Rule("proofnet:dummit_foote_rings", family=("13", "16"))),     # Part II, ring theory
        (10, 10, Rule("proofnet:dummit_foote_modules", family=("13", "16"))),
        (11, 11, Rule("proofnet:dummit_foote_vector_spaces", "15")),
        (12, 12, Rule("proofnet:dummit_foote_modules_pid", family=("13", "15", "16"))),
        (13, 14, Rule("proofnet:dummit_foote_fields", "12")),                 # Part IV, fields and Galois
    ),
    "Herstein": (                                                             # Topics in Algebra, 2nd ed.
        (2, 2, Rule("proofnet:herstein_groups", "20")),
        (3, 3, Rule("proofnet:herstein_rings", family=("13", "16"))),
        (4, 4, Rule("proofnet:herstein_vector_spaces", "15")),
        (5, 5, Rule("proofnet:herstein_fields", "12")),
        (6, 6, Rule("proofnet:herstein_linear_transformations", "15")),
    ),
    "Artin": (                                                                # Algebra; only chapters whose
        (2, 2, Rule("proofnet:artin_groups", "20")),                          # topic is the same in the 1st
        (3, 3, Rule("proofnet:artin_vector_spaces", "15")),                   # and 2nd editions
    ),
    "Rudin": (                                                                # Principles of Math. Analysis
        (1, 1, Rule("proofnet:rudin_numbers", family=("11", "12", "26"))),    # real and complex number systems
        (2, 2, Rule("proofnet:rudin_topology", "54")),                        # basic topology (metric spaces)
        (3, 3, Rule("proofnet:rudin_series", "40")),                          # numerical sequences and series
        (4, 4, Rule("proofnet:rudin_continuity", family=("26", "54"))),       # continuity on metric spaces
        (5, 5, Rule("proofnet:rudin_differentiation", "26")),                 # differentiation
        (6, 6, Rule("proofnet:rudin_integration", family=("26", "28"))),      # Riemann-Stieltjes integral
        (7, 7, Rule("proofnet:rudin_function_sequences", family=("26", "40", "46"))),
    ),
    "Pugh": (                                                                 # Real Mathematical Analysis
        (1, 1, Rule("proofnet:pugh_numbers", family=("26", "54"))),
        (2, 2, Rule("proofnet:pugh_topology", "54")),                         # a taste of topology
        (3, 3, Rule("proofnet:pugh_real_functions", "26")),                   # functions of a real variable
        (4, 4, Rule("proofnet:pugh_function_spaces", family=("26", "40", "46"))),
    ),
}
PROOFNET_UNLISTED = {
    "Putnam": "ProofNet's Putnam problems carry no topic",
    "Artin": "Artin's chapter numbering differs between editions past chapter 3",
}


def rule_minif2f(name: str) -> Rule:
    for pattern, rule in MINIF2F_RULES:
        if re.match(pattern, name):
            return rule
    return _x("minif2f:unknown", "name prefix not in the table")


def rule_putnam(tags: Iterable[str]) -> Rule:
    """All tags must have a rule. Strict only when every tag is strict with one code;
    family when every tag has codes (the union); otherwise excluded."""
    tags = sorted(set(tags))
    if not tags:
        return _x("putnam:untagged", "no topic tag")
    rules = []
    for t in tags:
        r = PUTNAM_TAGS.get(t)
        if r is None:
            return _x("putnam:unknown_tag", f"tag {t!r} not in the table")
        rules.append(r)
    name = "putnam:" + "+".join(tags)
    if len(rules) == 1:
        return rules[0]
    if any(r.status == "excluded" for r in rules):
        return _x(name, "one of the tags has no MSC class: " + "; ".join(r.reason for r in rules if r.reason))
    stricts = {r.strict for r in rules}
    if all(r.status == "strict" for r in rules) and len(stricts) == 1:
        return Rule(name, rules[0].strict)
    return Rule(name, family=tuple(sorted({c for r in rules for c in r.codes})))


_EX = re.compile(r"^exercise_(\d+)")


def rule_proofnet(book: str, exercise: str) -> Rule:
    m = _EX.match(exercise)
    if book not in PROOFNET_BOOKS:
        return _x(f"proofnet:{book.lower()}", PROOFNET_UNLISTED.get(book, "book not in the table"))
    if not m:
        return _x(f"proofnet:{book.lower()}", "no chapter number in the exercise id")
    ch = int(m.group(1))
    for lo, hi, rule in PROOFNET_BOOKS[book]:
        if lo <= ch <= hi:
            return rule
    return _x(f"proofnet:{book.lower()}_ch{ch}",
              PROOFNET_UNLISTED.get(book, f"chapter {ch} is not in the table"))


# ---------------------------------------------------------------- reading the sources

def _row(source: str, name: str, split: str, label: str, rule: Rule, statement: str) -> dict:
    return {
        "id": f"{source}:{name}",
        "source": source,
        "split": split,
        "name": name,
        "source_label": label,
        "status": rule.status,
        "msc": rule.strict or "",
        "msc_family": ";".join(rule.codes),
        "rule": rule.rule,
        "reason": rule.reason,
        "statement": statement,
    }


def _jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _split(s: str) -> str:
    return {"valid": "val"}.get(s, s)


def load_minif2f(path: Path) -> pd.DataFrame:
    rows = []
    for r in _jsonl(path):
        name = r["name"]
        prefix = re.split(r"_(?=\d|p\d)", name)[0]
        rows.append(_row("minif2f", name, _split(r.get("split", "test")), f"name:{prefix}", rule_minif2f(name),
                         r.get("goal") or r.get("formal_statement") or ""))
    return pd.DataFrame(rows)


def _norm(text: str) -> str:
    text = text.replace("/--", "").replace("-/", "")
    return re.sub(r"\W+", "", text).lower()


def load_proofnet(deepseek_path: Path, proofnet_dir: Path) -> pd.DataFrame:
    """ProofNet's Lean 4 statements, with the book from the original ids.

    DeepSeek's copy drops the book from the name, and some names repeat across books
    (exercise_1_1 is in several), so a repeated name is matched on the informal statement.
    A ProofNet id that cannot be matched to exactly one Lean 4 row is excluded.
    """
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in _jsonl(deepseek_path):
        by_key.setdefault((_split(r["split"]), r["name"]), []).append(r)
    rows = []
    for fname in ("valid", "test"):
        for r in _jsonl(Path(proofnet_dir) / "benchmark" / f"{fname}.jsonl"):
            book, exercise = r["id"].split("|", 1)
            split = _split(fname)
            cands = by_key.get((split, exercise), [])
            if len(cands) > 1:
                key = _norm(r.get("nl_statement", ""))[:40]
                cands = [c for c in cands if key and key in _norm(c.get("informal_prefix", ""))]
            rule = rule_proofnet(book, exercise)
            stmt = ""
            if len(cands) == 1:
                stmt = cands[0].get("goal") or cands[0].get("formal_statement") or ""
            elif rule.status != "excluded":
                rule = _x(rule.rule, "no unique Lean 4 statement for this id")
            rows.append(_row("proofnet", f"{book}|{exercise}", split, f"book:{book} ch{_chapter(exercise)}",
                             rule, stmt))
    return pd.DataFrame(rows)


def _chapter(exercise: str) -> str:
    m = _EX.match(exercise)
    return m.group(1) if m else "?"


def putnam_statement(text: str, name: str) -> str:
    """The ``theorem <name> ...`` statement of a PutnamBench file, up to the ``:=`` that
    closes it. Statements may hold ``let x := ...``; ``data.statement_end`` skips those, so
    this uses the same parser as the data and search-eval loaders."""
    from markov_lean.data import mask_comments, statement_end

    masked = mask_comments(text)
    m = re.search(r"^theorem\s+" + re.escape(name) + r"\b", masked, re.M)
    if not m:
        return ""
    return text[m.start(): statement_end(masked, m.end())].strip()


def load_putnam(putnam_dir: Path) -> pd.DataFrame:
    putnam_dir = Path(putnam_dir)
    meta = json.loads((putnam_dir / "informal" / "putnam.json").read_text())
    stmts: dict[str, str] = {}
    for p in sorted((putnam_dir / "lean4" / "src").glob("*.lean")):
        stmt = putnam_statement(p.read_text(errors="replace"), p.stem)
        if stmt:
            stmts[p.stem] = stmt
    rows = []
    for r in meta:
        name, tags = r["problem_name"], list(r.get("tags") or [])
        rule = rule_putnam(tags)
        stmt = stmts.get(name, "")
        if not stmt and rule.status != "excluded":
            rule = _x(rule.rule, "no Lean 4 statement in lean4/src")
        rows.append(_row("putnam", name, "test", "tags:" + ",".join(sorted(tags)), rule, stmt))
    return pd.DataFrame(rows)


def build(raw: str | Path = DEFAULT_RAW) -> pd.DataFrame:
    """Build the gold table from the sources under ``raw`` (see ``scripts/fetch_gold_sources.sh``)."""
    raw = Path(raw)
    parts = []
    if (raw / "minif2f.jsonl").exists():
        parts.append(load_minif2f(raw / "minif2f.jsonl"))
    if (raw / "proofnet.jsonl").exists() and (raw / "ProofNet").exists():
        parts.append(load_proofnet(raw / "proofnet.jsonl", raw / "ProofNet"))
    if (raw / "PutnamBench").exists():
        parts.append(load_putnam(raw / "PutnamBench"))
    if not parts:
        raise FileNotFoundError(f"no gold sources under {raw}; run scripts/fetch_gold_sources.sh")
    gold = pd.concat(parts, ignore_index=True)[GOLD_COLUMNS]
    dup = gold.id.duplicated()
    if dup.any():
        raise ValueError(f"repeated gold ids: {gold.id[dup].tolist()[:5]}")
    return gold


def read_gold(path: str | Path = DEFAULT_GOLD) -> pd.DataFrame:
    gold = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(GOLD_COLUMNS) - set(gold.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return gold


def family_codes(gold: pd.DataFrame) -> pd.Series:
    """The set of accepted codes per row (a group of one for strict rows)."""
    return gold.msc_family.map(lambda s: frozenset(c for c in str(s).split(";") if c))


# ---------------------------------------------------------------- summary

def balance(gold: pd.DataFrame) -> dict:
    """Counts by status, strict class, family rule and source, and excluded reasons."""
    scored = gold[gold.status != "excluded"]
    strict = gold[gold.status == "strict"]
    fam = gold[gold.status == "family"]
    out = {
        "n_problems": int(len(gold)),
        "status": {k: int(v) for k, v in gold.status.value_counts().items()},
        "by_source": {s: {k: int(v) for k, v in g.status.value_counts().items()}
                      for s, g in gold.groupby("source")},
        "strict_classes": {c: {"n": int(n), "title": msc.MSC2020_TOP[c], "share": round(n / len(strict), 4)}
                           for c, n in strict.msc.value_counts().items()},
        "strict_classes_by_source": {c: {s: int(n) for s, n in g.source.value_counts().items()}
                                     for c, g in strict.groupby("msc")},
        "family_rules": {r: {"n": int(len(g)), "codes": g.msc_family.iloc[0]} for r, g in fam.groupby("rule")},
        "excluded_reasons": {k: int(v) for k, v in gold[gold.status == "excluded"].reason.value_counts().items()},
        "n_scored": int(len(scored)),
    }
    if len(strict):
        out["majority_class"] = strict.msc.value_counts().index[0]
        out["majority_share"] = round(float(strict.msc.value_counts().iloc[0] / len(strict)), 4)
    return out


def balance_markdown(b: dict) -> str:
    lines = [f"Gold set: {b['n_problems']} problems; {b['status'].get('strict', 0)} strict, "
             f"{b['status'].get('family', 0)} family, {b['status'].get('excluded', 0)} excluded.", "",
             "| class | title | n | share | by source |", "|---|---|---|---|---|"]
    for c, v in sorted(b["strict_classes"].items(), key=lambda kv: -kv[1]["n"]):
        src = ", ".join(f"{s} {n}" for s, n in b["strict_classes_by_source"][c].items())
        lines.append(f"| {c} | {v['title']} | {v['n']} | {v['share']:.1%} | {src} |")
    lines += ["", "| family rule | codes | n |", "|---|---|---|"]
    for r, v in sorted(b["family_rules"].items()):
        lines.append(f"| {r} | {v['codes'].replace(';', ' ')} | {v['n']} |")
    lines += ["", "| excluded because | n |", "|---|---|"]
    for r, n in b["excluded_reasons"].items():
        lines.append(f"| {r} | {n} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- dev/test split

DEFAULT_EVAL_SPLIT = ROOT / "data" / "gold" / "msc_gold_eval_split.csv"
EVAL_SPLIT_SEED = 20261125
EVAL_SPLITS = ("dev", "test")


def _split_key(seed: int, pid: str) -> str:
    return hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()


def eval_split(gold: pd.DataFrame, seed: int = EVAL_SPLIT_SEED) -> pd.Series:
    """A fixed dev/test split of the gold set, for designing judge prompts without
    scoring on the problems they were tuned on.

    Problems are grouped by stratum (strict rows by their class, family rows by their
    rule, excluded rows together). Within a stratum they are ordered by
    ``sha256(f"{seed}:{id}")``; the first ``floor(n / 2)`` go to ``dev`` and the rest to
    ``test``, so test gets the extra problem of an odd stratum and every class with two
    or more problems is in both halves. The split depends only on the ids, the statuses
    and the seed, not on row order. Returns a Series of "dev"/"test" indexed like ``gold``.
    """
    stratum = np.where(gold.status == "strict", "strict:" + gold.msc,
                       np.where(gold.status == "family", "family:" + gold.rule, "excluded"))
    out = pd.Series("test", index=gold.index, dtype=object)
    for _, idx in pd.Series(gold.index, index=stratum).groupby(level=0):
        ids = gold.loc[idx.to_numpy(), "id"].astype(str)
        order = sorted(ids.index, key=lambda i: _split_key(seed, ids[i]))
        out.loc[order[: len(order) // 2]] = "dev"
    return out


def read_eval_split(path: str | Path = DEFAULT_EVAL_SPLIT) -> dict[str, str]:
    """The committed split (``data/gold/msc_gold_eval_split.csv``): gold id -> "dev" or "test"."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return dict(zip(df.id, df.eval_split))


def select_split(gold: pd.DataFrame, which: str, split: dict[str, str] | None = None) -> pd.DataFrame:
    """The gold rows in ``which`` ("dev", "test" or "all")."""
    if which == "all":
        return gold
    if which not in EVAL_SPLITS:
        raise ValueError(f"unknown split {which!r}; use dev, test or all")
    split = split if split is not None else read_eval_split()
    missing = set(gold.id) - set(split)
    if missing:
        raise ValueError(f"{len(missing)} gold ids have no dev/test assignment (e.g. {sorted(missing)[:3]}); "
                         "run `markov-lean gold-set eval-split` after rebuilding the gold table")
    return gold[gold.id.map(split) == which]


# ---------------------------------------------------------------- judge input

def read_minif2f_proofs(zip_path: str | Path) -> dict[str, str]:
    """Whole proofs from DeepSeek-Prover-V2's ``minif2f-solutions.zip``: name -> Lean text
    after the theorem's ``:= by`` (imports, options and the docstring dropped)."""
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if not n.endswith(".lean"):
                continue
            name = Path(n).stem
            text = z.read(n).decode("utf-8", errors="replace")
            m = re.search(r"^theorem\s+" + re.escape(name) + r"\b", text, re.M)
            if not m:
                continue
            body = text[m.end():]
            k = body.find(":= by")
            if k >= 0:
                out[name] = body[k + len(":= by"):].strip("\n")
    return out


def to_steps(gold: pd.DataFrame, *, statuses: Iterable[str] = ("strict", "family"), per_class: int | None = None,
             seed: int = 0, proofs: dict[str, str] | None = None) -> pd.DataFrame:
    """A `steps` table (docs/schema.md) for the judges: one step per problem.

    Step 0 shows the problem's initial goal. Its tactic is ``sorry`` unless ``proofs``
    has a whole proof for the problem (miniF2F only), which is then shown as the tactic.
    The benchmarks give no traced goal states, so the judges' proof-level label
    (``step -1``) is what `judge-accuracy` scores.

    ``per_class`` keeps at most that many problems per strict class and per family rule,
    chosen with a fixed seed, for a shorter local run.
    """
    g = gold[gold.status.isin(list(statuses)) & (gold.statement != "")].copy()
    if per_class:
        key = np.where(g.status == "strict", "strict:" + g.msc, g.rule)
        rng = np.random.default_rng(seed)
        keep = []
        for _, idx in pd.Series(g.index, index=key).groupby(level=0):
            idx = idx.to_numpy()
            keep.extend(sorted(rng.choice(idx, size=min(per_class, len(idx)), replace=False)))
        g = g.loc[sorted(keep)]
    proofs = proofs or {}
    tactic = [proofs.get(n, "sorry") if s == "minif2f" else "sorry" for n, s in zip(g.name, g.source)]
    return pd.DataFrame({
        "proof_id": g.id.to_numpy(),
        "step": 0,
        "source": g.source.to_numpy(),
        "split": g.split.to_numpy(),
        "theorem": g.name.to_numpy(),
        "file_path": "",
        "tactic": tactic,
        "state_before": g.statement.to_numpy(),
        "state_after": ["no goals" if t != "sorry" else "" for t in tactic],
        "true_area": None,
    })


# ---------------------------------------------------------------- command line

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="markov-lean gold-set",
                                 description="Benchmark problems with MSC codes known from their metadata.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the gold table from the downloaded sources")
    b.add_argument("--raw", default=str(DEFAULT_RAW), help="folder from scripts/fetch_gold_sources.sh")
    b.add_argument("--out", default=str(DEFAULT_GOLD))
    s = sub.add_parser("summary", help="print the class balance")
    s.add_argument("--gold", default=str(DEFAULT_GOLD))
    s.add_argument("--json", action="store_true")
    st = sub.add_parser("steps", help="write a steps table of the gold problems for a judge")
    st.add_argument("--gold", default=str(DEFAULT_GOLD))
    st.add_argument("--out", required=True, help="steps parquet to write")
    st.add_argument("--per-class", type=int, default=None,
                    help="at most this many problems per strict class and per family rule")
    st.add_argument("--strict-only", action="store_true", help="leave out the family rows")
    st.add_argument("--proofs", default=None,
                    help="DeepSeek-Prover-V2 minif2f-solutions.zip: show miniF2F proofs as the tactic")
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--eval-split", choices=["all", *EVAL_SPLITS], default="all",
                    help="only the dev or test half of the fixed dev/test split (default all)")
    es = sub.add_parser("eval-split", help="write the fixed dev/test split of the gold set")
    es.add_argument("--gold", default=str(DEFAULT_GOLD))
    es.add_argument("--out", default=str(DEFAULT_EVAL_SPLIT))
    es.add_argument("--seed", type=int, default=EVAL_SPLIT_SEED)
    a = ap.parse_args(argv)

    if a.cmd == "build":
        gold = build(a.raw)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        gold.to_csv(out, index=False)
        bal = balance(gold)
        out.with_name(out.stem + "_balance.json").write_text(json.dumps(bal, indent=1) + "\n")
        print(balance_markdown(bal))
        print(f"wrote {out}")
    elif a.cmd == "summary":
        bal = balance(read_gold(a.gold))
        print(json.dumps(bal, indent=1) if a.json else balance_markdown(bal))
    elif a.cmd == "steps":
        gold = select_split(read_gold(a.gold), a.eval_split)
        proofs = read_minif2f_proofs(a.proofs) if a.proofs else None
        statuses = ("strict",) if a.strict_only else ("strict", "family")
        steps = to_steps(gold, statuses=statuses, per_class=a.per_class, seed=a.seed, proofs=proofs)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        steps.to_parquet(out, index=False)
        n_proof = int((steps.tactic != "sorry").sum())
        print(f"wrote {out}: {len(steps)} problems ({n_proof} with a proof shown)")
    elif a.cmd == "eval-split":
        gold = read_gold(a.gold)
        sp = eval_split(gold, a.seed)
        out = Path(a.out)
        pd.DataFrame({"id": gold.id, "eval_split": sp}).to_csv(out, index=False)
        counts = pd.crosstab(gold.status, sp)
        print(counts.to_string())
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
