"""Cross-area case studies from exactly extracted steps: the PFR route for any Lean 4 project.

``scripts/case_extract.sh configs/cases/<name>.yaml`` builds the project on the Mac and runs
``scripts/pfr/ExtractSteps.lean`` on each file (LeanDojo's rules, premises from elaboration).
This module turns that JSON-lines output into the same tables as ``markov_lean.pfr``:

| file | what |
|---|---|
| ``steps.parquet`` | the `steps` table (docs/schema.md); ``source = split = <name>`` |
| ``premises.parquet`` | per step: cited constants and their defining modules |
| ``free_labels.parquet`` | per step: ``file_area`` (the proof's module, by the case table), ``premise_area`` (Mathlib premises, counting the project's ``ToMathlib``-style mirrors as Mathlib) and ``premise_own_area`` (also the project's own lemmas, by the case table) |
| ``summary.json`` | counts, coverage, area mix and quick switch statistics per label source |

The paper's tests then run with ``markov-lean switches`` on ``free_labels`` or on judge labels
(``python -m markov_lean.judge`` on ``steps.parquet``), and the label-free macrostates with
``markov-lean project``, exactly as for PFR. No API call is made here.

    python -m markov_lean.case_study convert --config configs/cases/pnt.yaml --raw results/cases/pnt/raw \\
        --out results/cases/pnt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import STEP_COLUMNS
from .pfr import build_tables, read_raw, switch_stats
from .premise_areas import AreaMap, majority
from .source_cases import CaseMap, MirrorAreaMap, OwnAreaMap, load_case

LABELS = ("premise_area", "premise_own_area")


def free_labels(steps: pd.DataFrame, prem: pd.DataFrame, amap: AreaMap, cmap: CaseMap) -> pd.DataFrame:
    df = steps[["proof_id", "step", "file_path"]].merge(prem, on=["proof_id", "step"], how="left")
    mirror, own = MirrorAreaMap(amap, cmap), OwnAreaMap(amap, cmap)

    def prem_area(paths, m):
        ps = paths if isinstance(paths, (list, np.ndarray)) else []
        return majority([a for a in (m.area(p) for p in ps) if a is not None])

    return pd.DataFrame({
        "proof_id": df["proof_id"], "step": df["step"],
        "file_area": df["file_path"].map(lambda p: cmap.area(p) or mirror.area(p, glue=True)),
        "premise_area": df["premise_paths"].map(lambda ps: prem_area(ps, mirror)),
        "premise_own_area": df["premise_paths"].map(lambda ps: prem_area(ps, own)),
    })


def case_files(cfg: dict, checkout: str | Path) -> list[str]:
    """Included ``.lean`` files, relative to the checkout, sorted; the same set `source_cases` reads."""
    root = Path(checkout).expanduser()
    out = set()
    for pat in cfg["include"]:
        for p in root.glob(pat):
            r = p.relative_to(root).as_posix()
            if not any(Path(r).match(e) or r.startswith(e.rstrip("*")) for e in cfg["exclude"]):
                out.add(r)
    return sorted(out)


def convert(config: str | Path, raw: str | Path, out: str | Path, map_path: str | Path | None = None) -> dict:
    cfg = load_case(config)
    records = read_raw(raw)
    if not records:
        raise SystemExit(f"no step records in {raw}/*.jsonl; run scripts/case_extract.sh {config} first")
    steps, prem = build_tables(records, source=cfg["name"])
    cmap = CaseMap(cfg["root"], cfg.get("areas") or {}, cfg.get("default"), cfg.get("mirrors") or ())
    lab = free_labels(steps, prem, AreaMap.load(map_path), cmap)
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    steps[STEP_COLUMNS].to_parquet(o / "steps.parquet", index=False)
    prem.to_parquet(o / "premises.parquet", index=False)
    lab.to_parquet(o / "free_labels.parquet", index=False)
    per_proof = steps.groupby("proof_id").size()
    summary = {
        "case": cfg["name"], "repo": cfg.get("repo"), "rev": cfg.get("rev"),
        "proofs": int(per_proof.size), "steps": int(len(steps)),
        "modules": int(len({r["module"] for r in records})),
        "steps_per_proof": {"mean": float(per_proof.mean()), "median": float(per_proof.median()),
                            "max": int(per_proof.max())},
        "coverage": {c: float(lab[c].notna().mean()) for c in ("file_area",) + LABELS},
        "area_mix": {c: {k: int(v) for k, v in lab[c].value_counts().items()} for c in ("file_area",) + LABELS},
        "switches": {c: switch_stats(lab, c) for c in LABELS},
    }
    (o / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m markov_lean.case_study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="extracted steps -> steps table, free labels and summary")
    c.add_argument("--config", required=True, help="configs/cases/<name>.yaml")
    c.add_argument("--raw", required=True, help="folder of ExtractSteps .jsonl files")
    c.add_argument("--out", required=True)
    c.add_argument("--map", help="Mathlib folder-to-MSC table (default configs/premise_msc.yaml)")
    f = sub.add_parser("field", help="print one field of a case config (for scripts/case_extract.sh)")
    f.add_argument("config")
    f.add_argument("key")
    m = sub.add_parser("files", help="the project's .lean files the config includes, one per line")
    m.add_argument("config")
    m.add_argument("checkout")
    args = ap.parse_args(argv)
    if args.cmd == "field":
        v = load_case(args.config).get(args.key)
        print(" ".join(v) if isinstance(v, list) else ("" if v is None else v))
        return 0
    if args.cmd == "files":
        print("\n".join(case_files(load_case(args.config), args.checkout)))
        return 0
    s = convert(args.config, args.raw, args.out, args.map)
    print(f"{s['case']}: {s['proofs']} proofs, {s['steps']} steps from {s['modules']} modules -> {args.out}")
    for k, v in s["switches"].items():
        if v["switches_per_move"] is not None:
            print(f"  {k}: {v['switches_per_move']:.3f} switches/move, "
                  f"{v['share_proofs_with_switch']:.1%} of proofs switch, dwell {v['mean_dwell']:.1f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
