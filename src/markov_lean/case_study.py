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
import re
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


def build_failures(log: str) -> list[str]:
    """Modules a failed ``lake build`` names under "Some required targets logged failures:"."""
    out, inside = [], False
    for line in log.splitlines():
        t = line.strip()
        if t.startswith("Some required targets logged failures"):
            inside = True
            continue
        if inside:
            if t.startswith("- "):
                m = t[2:].strip().lstrip("+").split(":")[0].strip()
                if m and m not in out:
                    out.append(m)
            else:
                inside = False
    return out


_IMPORT = re.compile(r"^\s*(?:public\s+|private\s+)?(?:meta\s+)?import\s+([\w.']+)", re.M)


def without(files: list[str], checkout: str | Path, skip: set[str]) -> list[str]:
    """``files`` less the modules in ``skip`` and every module of the checkout that imports one,
    directly or through other modules of the checkout (they cannot build either)."""
    root = Path(checkout).expanduser()
    cache: dict[str, list[str]] = {}

    def imports(mod: str) -> list[str]:
        if mod not in cache:
            p = root / (mod.replace(".", "/") + ".lean")
            cache[mod] = _IMPORT.findall(p.read_text(errors="replace")) if p.exists() else []
        return cache[mod]

    bad = set(skip)
    memo: dict[str, bool] = {}

    def broken(mod: str, path: tuple = ()) -> bool:
        if mod in bad:
            return True
        if mod in memo:
            return memo[mod]
        if mod in path:
            return False
        memo[mod] = any(broken(m, path + (mod,)) for m in imports(mod)
                        if (root / (m.replace(".", "/") + ".lean")).exists())
        return memo[mod]

    return [f for f in files if not broken(f[:-len(".lean")].replace("/", "."))]


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


# --------------------------------------------------------------------------- judge labels against the control

COMPARE_STATS = ("switch_rate", "excess", "has_switch")


def judge_proofs(case_dir: str | Path, source: str = "judge") -> pd.DataFrame | None:
    """Per-proof rows of ``markov-lean switches`` on one case (``<case>/switches/proofs.csv``), with
    ``excess`` = switch rate minus its within-proof shuffle mean; None when that run is missing."""
    p = Path(case_dir) / "switches" / "proofs.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d["source"] == source] if "source" in d else d
    if d.empty or "shuffle_switch_rate" not in d:
        return None
    return d.assign(excess=d["switch_rate"] - d["shuffle_switch_rate"])


def _boot_diff(a: np.ndarray, b: np.ndarray, rng, n_boot: int) -> list[float]:
    da = rng.integers(0, len(a), (n_boot, len(a)))
    db = rng.integers(0, len(b), (n_boot, len(b)))
    d = a[da].mean(1) - b[db].mean(1)
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def _perm_p(a: np.ndarray, b: np.ndarray, rng, n_perm: int) -> float:
    """Two-sided permutation p of the difference in means, proofs swapped between the two projects."""
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        hits += abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def compare(cases: dict[str, str | Path], control: str, n_boot: int = 2000, n_perm: int = 2000,
            seed: int = 0) -> dict:
    """Judge-label switching of each case against the control project, each proof weighted equally.

    Per case: proofs, mean switch rate, its shuffle mean, their ratio, mean excess over the shuffle
    and the share of proofs with a switch. Per case other than the control: the difference from the
    control in switch rate, in excess over the shuffle (ordering) and in share with a switch, each
    with a 95% bootstrap interval over proofs and a two-sided permutation p. Spends nothing."""
    rng = np.random.default_rng(seed)
    rows = {n: judge_proofs(d) for n, d in cases.items()}
    out: dict = {"control": control, "source": "judge", "n_boot": n_boot, "n_perm": n_perm,
                 "cases": {}, "vs_control": {}, "missing": sorted(n for n, r in rows.items() if r is None)}
    for n, d in rows.items():
        if d is None:
            continue
        rate, shuf = float(d["switch_rate"].mean()), float(d["shuffle_switch_rate"].mean())
        out["cases"][n] = {"proofs": int(len(d)), "switch_rate": rate, "shuffle": shuf,
                           "ratio": rate / shuf if shuf else None, "excess": float(d["excess"].mean()),
                           "has_switch": float(d["has_switch"].mean())}
    ctl = rows.get(control)
    if ctl is None:
        return out
    for n, d in rows.items():
        if d is None or n == control:
            continue
        e = {}
        for s in COMPARE_STATS:
            a, b = d[s].to_numpy(float), ctl[s].to_numpy(float)
            e[s] = {"difference": float(a.mean() - b.mean()), "ci95": _boot_diff(a, b, rng, n_boot),
                    "p_two_sided": _perm_p(a, b, rng, n_perm)}
        out["vs_control"][n] = e
    return out


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
    m.add_argument("--skip", help="file of module names to leave out (modules that failed to build)")
    b = sub.add_parser("build-failures", help="modules a failed lake build log names, one per line")
    b.add_argument("log")
    k = sub.add_parser("compare", help="judge-label switching of each case against the control (no API calls)")
    k.add_argument("--case", action="append", required=True, metavar="NAME=DIR",
                   help="a case and its folder (holding switches/proofs.csv); repeat")
    k.add_argument("--control", required=True, help="the control case's NAME")
    k.add_argument("--out", required=True, help="JSON file to write")
    k.add_argument("--n-boot", type=int, default=2000)
    k.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        cases = dict(c.split("=", 1) for c in args.case)
        r = compare(cases, args.control, args.n_boot, args.n_perm)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(r, indent=2))
        for n, c in r["cases"].items():
            print(f"{n}: {c['proofs']} proofs, {c['switch_rate']:.3f} switches/move vs shuffle {c['shuffle']:.3f} "
                  f"(ratio {c['ratio']:.2f}), {c['has_switch']:.1%} switch")
        for n, e in r["vs_control"].items():
            d = e["switch_rate"]
            print(f"{n} - {args.control}: switch rate {d['difference']:+.3f} [{d['ci95'][0]:+.3f}, "
                  f"{d['ci95'][1]:+.3f}], p {d['p_two_sided']:.3g}")
        if r["missing"]:
            print(f"no judge switches run yet for: {', '.join(r['missing'])}")
        print(f"wrote {args.out}")
        return 0
    if args.cmd == "field":
        v = load_case(args.config).get(args.key)
        print(" ".join(v) if isinstance(v, list) else ("" if v is None else v))
        return 0
    if args.cmd == "files":
        skip = set()
        if args.skip and Path(args.skip).exists():
            skip = {m.strip() for m in Path(args.skip).read_text().splitlines() if m.strip()}
        files = case_files(load_case(args.config), args.checkout)
        if skip:
            files = without(files, args.checkout, skip)
        print("\n".join(files))
        return 0
    if args.cmd == "build-failures":
        print("\n".join(build_failures(Path(args.log).read_text(errors="replace"))))
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
