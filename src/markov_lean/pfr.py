"""PFR case study: turn extracted PFR proof steps into a `steps` table and free area labels.

The polynomial Freiman-Ruzsa conjecture was proved with entropy methods (Gowers, Green,
Manners and Tao, 2023) and formalized in Lean 4 in teorth/pfr. It is a cross-area proof:
additive-combinatorics statements proved with information theory. This module compares
it with the LeanDojo (Mathlib) proofs the paper studies.

Input is the JSON-lines output of ``scripts/pfr/ExtractSteps.lean`` (one file per PFR
module, written by ``scripts/pfr_extract.sh``). Each line is one tactic step with
``theorem, module, file_path, line, tactic, state_before, state_after`` and ``premises``
(constants the step cites, each with the module that defines it). The extractor follows
LeanDojo's ExtractData rules; like LeanDojo's benchmark export, focusing dots are dropped
here, so step numbers count the kept tactics only.

``python -m markov_lean.pfr convert --raw results/pfr/raw --pfr-root ~/pfr --out results/pfr``
writes

| file | what |
|---|---|
| ``pfr_steps.parquet`` | the `steps` table (docs/schema.md); ``source = split = "pfr"`` |
| ``premises.parquet`` | per step: cited constants and their defining modules |
| ``free_labels.parquet`` | per step: ``file_area`` and ``chapter_area`` (where the proof lives), ``premise_area`` (Mathlib premises only, as `markov-lean premise-labels`) and ``premise_pfr_area`` (Mathlib premises plus PFR's own, by configs/pfr_msc.yaml) |
| ``summary.json`` | counts, coverage, area mix, and switch statistics per label source |

No API call is made. Judge labels come from ``python -m markov_lean.judge`` on
``pfr_steps.parquet``, exactly as for LeanDojo.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .data import STEP_COLUMNS, _frame
from .premise_areas import AreaMap, majority

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PFR_MAP = ROOT / "configs" / "pfr_msc.yaml"
LABEL_SOURCES = ("premise_area", "premise_pfr_area")


# --------------------------------------------------------------------------- reading


def read_raw(raw_dir: str | Path) -> list[dict]:
    """All step records from ``<raw_dir>/*.jsonl``, in file then line order."""
    rows = []
    for p in sorted(Path(raw_dir).glob("*.jsonl")):
        with open(p, encoding="utf8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def is_focusing_dot(tactic: str) -> bool:
    return tactic.lstrip().startswith("·")


def module_path(module: str) -> str:
    """``PFR.ForMathlib.Entropy.Basic`` -> ``PFR/ForMathlib/Entropy/Basic.lean``; Mathlib modules get
    LeanDojo's ``.lake/packages/mathlib/`` prefix so `premise_areas.mathlib_relpath` reads them."""
    rel = module.replace(".", "/") + ".lean"
    return ".lake/packages/mathlib/" + rel if module.split(".")[0] == "Mathlib" else rel


def build_tables(records: Iterable[dict], min_steps: int = 1, source: str = "pfr") -> tuple[pd.DataFrame, pd.DataFrame]:
    """`steps` table and per-step premises. A proof is one declaration in one module.

    ``source`` names the project in proof ids (``pfr:<name>``) and in the source and split columns."""
    by_proof: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        if is_focusing_dot(r["tactic"]):
            continue
        by_proof.setdefault((r["module"], r["theorem"]), []).append(r)
    steps, prem, seen = [], [], Counter()
    for (module, name), rs in by_proof.items():
        rs = sorted(rs, key=lambda r: r.get("line", 0))  # stable: ties keep extraction order
        seen[name] += 1
        if len(rs) < min_steps:
            continue
        pid = f"{source}:{name}" if seen[name] == 1 else f"{source}:{name}#{seen[name]}"
        for i, r in enumerate(rs):
            steps.append({"proof_id": pid, "step": i, "source": source, "split": source, "theorem": name,
                          "file_path": r.get("file_path") or module_path(module), "tactic": r["tactic"],
                          "state_before": r["state_before"], "state_after": r["state_after"],
                          "true_area": None})
            ps = r.get("premises") or []
            prem.append({"proof_id": pid, "step": i, "module": module,
                         "premises": [p["full_name"] for p in ps],
                         "premise_paths": [module_path(p["module"]) for p in ps]})
    return _frame(steps), pd.DataFrame(prem, columns=["proof_id", "step", "module", "premises", "premise_paths"])


# --------------------------------------------------------------------------- free labels


class PFRMap:
    """Longest-prefix map from PFR module paths to MSC codes, plus blueprint chapter codes."""

    def __init__(self, areas: dict[str, str | None], chapters: dict[str, str] | None = None):
        self.areas = areas
        self.chapters = chapters or {}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PFRMap":
        raw = yaml.safe_load(Path(path or DEFAULT_PFR_MAP).read_text()) or {}
        norm = lambda v: None if v is None else str(v).zfill(2)  # noqa: E731
        return cls({str(k).strip("/"): norm(v) for k, v in (raw.get("areas") or {}).items()},
                   {str(k): norm(v) for k, v in (raw.get("chapters") or {}).items()})

    def area(self, path: Any) -> str | None:
        """Area of a PFR-native module path (``PFR/Kullback.lean``); None for PFR/Mathlib/* and others."""
        if not isinstance(path, str):
            return None
        parts = [p for p in path.replace("\\", "/").split("/") if p]
        if not parts or parts[0] != "PFR" or (len(parts) > 1 and parts[1] == "Mathlib"):
            return None
        rest = parts[1:]
        if rest and rest[-1].endswith(".lean"):
            rest[-1] = rest[-1][:-5]
        for k in range(len(rest), 0, -1):
            key = "/".join(rest[:k])
            if key in self.areas:
                return self.areas[key]
        return None


_LEAN_TAG = re.compile(r"\\lean\{([^}]*)\}")


def blueprint_chapters(pfr_root: str | Path | None) -> dict[str, str]:
    """Declaration name -> blueprint chapter, from ``blueprint/src/chapter/*.tex`` \\lean{...} tags."""
    if pfr_root is None:
        return {}
    out = {}
    for tex in sorted((Path(pfr_root) / "blueprint" / "src" / "chapter").glob("*.tex")):
        for m in _LEAN_TAG.finditer(tex.read_text(encoding="utf8")):
            for name in m.group(1).split(","):
                out.setdefault(name.strip(), tex.stem)
    return out


def free_labels(steps: pd.DataFrame, prem: pd.DataFrame, amap: AreaMap, pmap: PFRMap,
                chapters: dict[str, str]) -> pd.DataFrame:
    """One row per step with the four free label sources (``None``: no label)."""
    df = steps[["proof_id", "step", "theorem", "file_path"]].merge(prem, on=["proof_id", "step"], how="left")

    def file_area(p):
        return pmap.area(p) or amap.area(p, glue=True)

    def prem_area(paths, with_pfr):
        areas = []
        for p in paths if isinstance(paths, (list, np.ndarray)) else []:
            a = amap.area(p)
            if a is None and with_pfr:
                a = pmap.area(p)
            if a is not None:
                areas.append(a)
        return majority(areas)

    return pd.DataFrame({
        "proof_id": df["proof_id"], "step": df["step"],
        "file_area": df["file_path"].map(file_area),
        "chapter_area": df["theorem"].map(lambda n: pmap.chapters.get(chapters.get(n, ""))),
        "premise_area": df["premise_paths"].map(lambda ps: prem_area(ps, False)),
        "premise_pfr_area": df["premise_paths"].map(lambda ps: prem_area(ps, True)),
    })


# --------------------------------------------------------------------------- summary


def switch_stats(lab: pd.DataFrame, col: str, ref: str = "file_area") -> dict:
    """Switches between consecutive labelled steps of a proof (unlabelled steps skipped), the
    share of proofs with a switch, mean dwell, and the share of labelled steps off the proof's
    ``ref`` area (where the proof lives). A quick look only; the paper's tests use `switches`."""
    moves = sw = proofs = with_sw = off = labelled = 0
    dwell = []
    for _, g in lab.sort_values(["proof_id", "step"]).groupby("proof_id", sort=False):
        ys = [y for y in g[col] if y is not None and y == y]
        if not ys:
            continue
        home = next((r for r in g[ref] if r is not None and r == r), None)
        labelled += len(ys)
        off += sum(1 for y in ys if home is not None and y != home)
        proofs += 1
        n = sum(1 for a, b in zip(ys, ys[1:]) if a != b)
        moves += len(ys) - 1
        sw += n
        with_sw += n > 0
        run = 1
        for a, b in zip(ys, ys[1:]):
            if a == b:
                run += 1
            else:
                dwell.append(run)
                run = 1
        dwell.append(run)
    return {"proofs": proofs, "labelled_steps": labelled, "moves": moves,
            "switches_per_move": sw / moves if moves else None,
            "share_proofs_with_switch": with_sw / proofs if proofs else None,
            "mean_dwell": float(np.mean(dwell)) if dwell else None,
            "share_off_home_area": off / labelled if labelled else None}


def summarize(steps: pd.DataFrame, lab: pd.DataFrame) -> dict:
    per_proof = steps.groupby("proof_id").size()
    out = {"proofs": int(per_proof.size), "steps": int(len(steps)),
           "steps_per_proof": {"mean": float(per_proof.mean()) if len(per_proof) else None,
                               "median": float(per_proof.median()) if len(per_proof) else None,
                               "max": int(per_proof.max()) if len(per_proof) else None},
           "coverage": {c: float(lab[c].notna().mean()) for c in
                        ("file_area", "chapter_area", "premise_area", "premise_pfr_area")},
           "area_mix": {c: {k: int(v) for k, v in lab[c].value_counts().items()} for c in
                        ("file_area", "chapter_area", "premise_area", "premise_pfr_area")},
           "switches": {c: switch_stats(lab, c) for c in LABEL_SOURCES}}
    return out


# --------------------------------------------------------------------------- CLI


def convert(raw: str | Path, out: str | Path, pfr_root: str | Path | None = None,
            map_path: str | Path | None = None, pfr_map_path: str | Path | None = None) -> dict:
    records = read_raw(raw)
    if not records:
        raise SystemExit(f"no step records in {raw}/*.jsonl; run scripts/pfr_extract.sh first")
    steps, prem = build_tables(records)
    lab = free_labels(steps, prem, AreaMap.load(map_path), PFRMap.load(pfr_map_path), blueprint_chapters(pfr_root))
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    steps[STEP_COLUMNS].to_parquet(o / "pfr_steps.parquet", index=False)
    prem.to_parquet(o / "premises.parquet", index=False)
    lab.to_parquet(o / "free_labels.parquet", index=False)
    summary = summarize(steps, lab)
    summary["modules"] = int(len({r["module"] for r in records}))
    (o / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m markov_lean.pfr")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="extracted PFR steps -> steps table, free labels and summary")
    c.add_argument("--raw", required=True, help="folder of ExtractSteps .jsonl files")
    c.add_argument("--out", required=True)
    c.add_argument("--pfr-root", help="teorth/pfr checkout, for blueprint chapter labels")
    c.add_argument("--map", help="Mathlib folder-to-MSC table (default configs/premise_msc.yaml)")
    c.add_argument("--pfr-map", help="PFR module-to-MSC table (default configs/pfr_msc.yaml)")
    args = ap.parse_args(argv)
    s = convert(args.raw, args.out, args.pfr_root, args.map, args.pfr_map)
    print(f"{s['proofs']} proofs, {s['steps']} steps from {s['modules']} modules -> {args.out}")
    for k, v in s["switches"].items():
        if v["switches_per_move"] is not None:
            print(f"  {k}: {v['switches_per_move']:.3f} switches/move, "
                  f"{v['share_proofs_with_switch']:.1%} of proofs switch, dwell {v['mean_dwell']:.1f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
