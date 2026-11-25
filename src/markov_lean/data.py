"""Loaders that turn public Lean datasets into the `steps` table (docs/schema.md).

Sources and their raw layouts (as downloaded by scripts/fetch_data.sh into data/raw/):

- LeanDojo Benchmark 4: ``leandojo_benchmark_4/<strategy>/<split>.json`` where strategy is
  ``random`` or ``novel_premises``. Each file is a JSON list of theorems with keys
  ``url, commit, file_path, full_name, start, end, traced_tactics``; each traced tactic has
  ``tactic, annotated_tactic, state_before, state_after``. Field names confirmed from
  LeanDojo's export code (scripts/generate-benchmark-lean4.ipynb in lean-dojo/LeanDojo).
  The export drops tactics whose state_before is "no goals" and focusing dots, so step
  numbers count the kept tactics only.
- Lean Workbook: ``lean_workbook/lean_workbook.json`` with columns ``id, status, tactic,
  state_before, state_after, natural_language_statement, answer, formal_statement``.
  Each row is one problem with at most one recorded tactic, so it yields at most one step.
- miniF2F, ProofNet, PutnamBench: git checkouts of Lean source files. They hold theorem
  statements without proofs, so they yield one row per theorem with ``step = -1``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

STEP_COLUMNS = [
    "proof_id",
    "step",
    "source",
    "split",
    "theorem",
    "file_path",
    "tactic",
    "state_before",
    "state_after",
    "true_area",
]

LEANDOJO_STRATEGIES = ("random", "novel_premises")
SPLITS = ("train", "val", "test")


def _frame(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows), columns=STEP_COLUMNS)
    df["step"] = df["step"].astype("int64")
    df["true_area"] = df["true_area"].astype("object")
    return df


# ---------------------------------------------------------------------------
# LeanDojo Benchmark 4
# ---------------------------------------------------------------------------

def iter_leandojo_theorems(path: Path) -> Iterator[dict]:
    """Yield theorem dicts from one LeanDojo split file.

    The files are single JSON arrays (the largest, random/train.json, is a few hundred MB
    unpacked), so this reads the whole file; use ``max_theorems`` in the caller to cap
    the rows kept in memory.
    """
    with open(path) as f:
        data = json.load(f)
    yield from data


def load_leandojo(
    root: Path,
    split: str = "train",
    strategy: str = "random",
    max_theorems: int | None = None,
    min_steps: int = 1,
) -> pd.DataFrame:
    """Load one split of LeanDojo Benchmark 4 as a steps table.

    ``min_steps`` drops proofs with fewer kept tactics (single-step proofs carry no
    transition information for the Markov model).
    """
    if strategy not in LEANDOJO_STRATEGIES:
        raise ValueError(f"strategy must be one of {LEANDOJO_STRATEGIES}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    path = Path(root) / strategy / f"{split}.json"
    rows: list[dict] = []
    kept = 0
    for thm in iter_leandojo_theorems(path):
        tactics = thm.get("traced_tactics") or []
        if len(tactics) < min_steps:
            continue
        name = thm["full_name"]
        for i, t in enumerate(tactics):
            rows.append(
                {
                    "proof_id": f"leandojo:{name}",
                    "step": i,
                    "source": "leandojo",
                    "split": split,
                    "theorem": name,
                    "file_path": thm.get("file_path", ""),
                    "tactic": t["tactic"],
                    "state_before": t["state_before"],
                    "state_after": t["state_after"],
                    "true_area": None,
                }
            )
        kept += 1
        if max_theorems is not None and kept >= max_theorems:
            break
    return _frame(rows)


# ---------------------------------------------------------------------------
# Lean Workbook
# ---------------------------------------------------------------------------

def load_workbook(path: Path, proved_only: bool = True) -> pd.DataFrame:
    """Load Lean Workbook as a steps table (one step per problem with a recorded tactic).

    With ``proved_only`` rows whose status says the statement was disproved are dropped,
    since their tactic proves the negation.

    Accepts the JSON file (a list of records, or JSON lines) or a Parquet file.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        records = pd.read_parquet(path).to_dict("records")
    else:
        text = path.read_text()
        stripped = text.lstrip()
        if stripped.startswith("["):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows = []
    for r in records:
        tactic = r.get("tactic")
        if not tactic:
            continue
        if proved_only and "disprov" in str(r.get("status", "")).lower():
            continue
        pid = str(r.get("id"))
        rows.append(
            {
                "proof_id": f"workbook:{pid}",
                "step": 0,
                "source": "workbook",
                "split": "train",
                "theorem": pid,
                "file_path": "",
                "tactic": tactic,
                "state_before": r.get("state_before") or "",
                "state_after": r.get("state_after") or "",
                "true_area": None,
            }
        )
    return _frame(rows)


# ---------------------------------------------------------------------------
# Statement-only benchmarks (miniF2F, ProofNet, PutnamBench)
# ---------------------------------------------------------------------------

_DECL = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+)?(theorem|lemma)\s+([^\s(:{\[]+)", re.M)
_END = re.compile(r":=\s*(?:by\b|sorry\b)?", re.M)


def _split_from_path(path: Path) -> str:
    low = str(path).lower()
    if "valid" in low or "/val" in low:
        return "val"
    if "test" in low:
        return "test"
    return "test"


def parse_lean_statements(text: str) -> list[tuple[str, str]]:
    """Return (name, statement) for each theorem/lemma in a Lean source file.

    The statement runs from the declaration keyword to the first ``:=`` after it, which
    is how these benchmarks write their (unproved) theorems. Doc comments are dropped.
    """
    out = []
    for m in _DECL.finditer(text):
        start = m.start()
        end_m = _END.search(text, m.end())
        stop = end_m.start() if end_m else len(text)
        stmt = text[start:stop].strip()
        out.append((m.group(2), stmt))
    return out


def load_statements(root: Path, source: str, glob: str = "**/*.lean") -> pd.DataFrame:
    """Scan a benchmark checkout for theorem statements; one row per theorem, step -1."""
    root = Path(root)
    rows = []
    for path in sorted(root.glob(glob)):
        if ".lake" in path.parts:
            continue
        rel = path.relative_to(root)
        for name, stmt in parse_lean_statements(path.read_text(errors="replace")):
            rows.append(
                {
                    "proof_id": f"{source}:{name}",
                    "step": -1,
                    "source": source,
                    "split": _split_from_path(rel),
                    "theorem": name,
                    "file_path": str(rel),
                    "tactic": "",
                    "state_before": stmt,
                    "state_after": "",
                    "true_area": None,
                }
            )
    df = _frame(rows)
    return df.drop_duplicates(subset=["proof_id"], keep="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parquet I/O and convenience
# ---------------------------------------------------------------------------

def processed_path(out_dir: Path, source: str, split: str) -> Path:
    return Path(out_dir) / f"{source}_{split}.parquet"


def write_steps(df: pd.DataFrame, out_dir: Path, source: str, split: str) -> Path:
    missing = set(STEP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"steps table is missing columns: {sorted(missing)}")
    out = processed_path(out_dir, source, split)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[STEP_COLUMNS].to_parquet(out, index=False)
    return out


def read_steps(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(STEP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def build_all(raw_dir: Path, out_dir: Path, max_theorems: int | None = None) -> list[Path]:
    """Convert every downloaded source under ``raw_dir`` into Parquet files in ``out_dir``.

    Sources that are not present are skipped with a message, so this can run after a
    partial download.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    written: list[Path] = []

    ld = raw_dir / "leandojo_benchmark_4"
    if ld.exists():
        for split in SPLITS:
            if (ld / "random" / f"{split}.json").exists():
                df = load_leandojo(ld, split=split, max_theorems=max_theorems)
                written.append(write_steps(df, out_dir, "leandojo", split))
    else:
        print(f"skip leandojo: {ld} not found")

    wb = raw_dir / "lean_workbook" / "lean_workbook.json"
    if wb.exists():
        written.append(write_steps(load_workbook(wb), out_dir, "workbook", "train"))
    else:
        print(f"skip workbook: {wb} not found")

    for source, sub in (("minif2f", "miniF2F-lean4"), ("proofnet", "ProofNet-lean4"), ("putnam", "PutnamBench")):
        d = raw_dir / sub
        if not d.exists():
            print(f"skip {source}: {d} not found")
            continue
        glob = "lean4/**/*.lean" if source == "putnam" else "**/*.lean"
        df = load_statements(d, source, glob=glob)
        for split, part in df.groupby("split"):
            written.append(write_steps(part, out_dir, source, split))

    for p in written:
        print(f"wrote {p} ({len(pd.read_parquet(p, columns=['proof_id']))} rows)")
    return written
