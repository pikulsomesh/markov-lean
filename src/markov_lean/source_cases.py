"""Case studies from Lean source text alone: a quick, build-free proxy for `markov_lean.pfr`.

The exact route (``scripts/case_extract.sh``) builds a project and extracts its tactic steps
with Lean itself, which needs the project's Mathlib cache and hours on the Mac. This module
reads the ``.lean`` files instead, so it runs anywhere in seconds and can compare many
projects with Mathlib under one labeller:

1. **Declarations.** ``theorem``/``lemma``/``def``/... names are read with their namespaces,
   giving a name -> file index over Mathlib and the project.
2. **Steps.** Each ``theorem``/``lemma`` proved ``:= by`` is split into tactic steps by
   layout: a new line at or left of the current tactic's column starts a step; a deeper line
   continues it, unless the previous step opened a tactic block (ends in ``by``, a
   ``case``/``next``/``conv``/``|`` arm, or a focusing dot). Dots and ``| pat =>`` heads are
   dropped, as LeanDojo drops focusing dots. Term-mode proofs are skipped.
3. **Labels.** Cited names are read and resolved exactly as ``premise_areas`` does for
   LeanDojo's text (``bridges.cited_names``, then the exact / namespace / majority / suffix
   tiers). ``premise`` keeps Mathlib premises only (configs/premise_msc.yaml; a project's
   own ``*/Mathlib/*`` mirror counts as Mathlib); ``premise_own`` also counts the project's
   own lemmas by the module table in its case config.
4. **Switch test.** ``switches.analyse_source`` with the shuffle, Markov (pooled, one chain)
   and noise nulls, the same settings as the PFR judge run.

It is a proxy: no goal states (so local names are filtered by spelling only), no
elaboration (so ``simp`` lemma sets and notation are invisible), and layout heuristics for
step boundaries. Run it on Mathlib with the same code so comparisons stay like for like.

    python -m markov_lean.source_cases run --mathlib ~/mathlib4 --case configs/cases/pfr.yaml=~/pfr \\
        --case configs/cases/sphere_packing.yaml=~/Sphere-Packing-Lean --out results/cases/source
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from . import switches as sw
from .data import STEP_COLUMNS, _frame
from .premise_areas import AreaMap, Resolver, label_steps, mathlib_relpath, noise_rate

# --------------------------------------------------------------------------- reading source

_BLOCK = re.compile(r"/-.*?-/", re.DOTALL)
_LINE = re.compile(r"--[^\n]*")
_STR = re.compile(r'"(?:\\.|[^"\\])*"')
_MODS = r"(?:(?:private|protected|noncomputable|nonrec|unsafe|partial|scoped|local)\s+)*"
_DECL = re.compile(r"^(?:@\[[^\]]*\]\s*)*" + _MODS +
                   r"(theorem|lemma|def|abbrev|instance|structure|class|inductive|opaque|axiom|irreducible_def)"
                   r"\s+([^\s:({\[⦃]+)")
_TOP = re.compile(r"^(?:@\[|/--|namespace\b|end\b|section\b|open\b|variable\b|universe\b|#|set_option\b|"
                  r"attribute\b|example\b|noncomputable\s+section|import\b|mutual\b|include\b|omit\b|"
                  r"local\b|scoped\b|macro\b|syntax\b|elab\b|notation\b|infix|prefix|postfix|"
                  + _MODS + r"(?:theorem|lemma|def|abbrev|instance|structure|class|inductive|opaque|axiom|"
                  r"irreducible_def)\b)")
_BY = re.compile(r":=\s*by\b(.*)$")
_OPENS_ARM = re.compile(r"^(?:case\b|case'|next\b|conv\b|conv_lhs\b|conv_rhs\b|on_goal\b|\|)")


def strip_comments(text: str) -> str:
    """Remove block and line comments, keeping line breaks (so line numbers and columns stay)."""
    text = _BLOCK.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    out = []
    for line in text.split("\n"):
        masked = _STR.sub(lambda m: "x" * len(m.group(0)), line)
        i = masked.find("--")
        out.append(line if i < 0 else line[:i])
    return "\n".join(out)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def split_tactics(lines: list[tuple[int, str]]) -> list[str]:
    """Tactic steps from ``(column, text)`` lines of one ``by`` block (see the module notes)."""
    steps: list[list] = []  # [start column, text, opens block]
    for col, raw in lines:
        text = raw.strip()
        if not text:
            continue
        # a focusing dot or an arm head (| pat =>, case h =>) is dropped; what follows is the tactic
        head_opens = False
        while True:
            m = re.match(r"^(?:[·•]|\.(?=\s))\s*", text)
            if m:
                col += m.end()
                text = text[m.end():]
                head_opens = True
                continue
            m = re.match(r"^(?:\|[^=]*=>|(?:case|case'|next)\b[^=]*=>|on_goal\s+-?\d+\s*=>)\s*", text)
            if m:
                col += m.end()
                text = text[m.end():]
                head_opens = True
                continue
            break
        prev = steps[-1] if steps else None
        if not text:
            if prev is not None:
                prev[2] = True  # bare arm head: the lines below are its tactics
            continue
        cont = (prev is not None and not head_opens and col > prev[0] and not prev[2])
        if cont:
            prev[1] += " " + text
            prev[2] = _opens(prev[1])
        else:
            steps.append([col, text, _opens(text)])
    return [s[1] for s in steps]


def _opens(text: str) -> bool:
    t = text.rstrip()
    if re.search(r"(?:^|[\s(])by$", t):
        return True
    return t.endswith("=>") and bool(_OPENS_ARM.match(t.lstrip()))


@dataclass
class SourceFile:
    rel: str
    decls: list[str] = field(default_factory=list)
    proofs: list[tuple[str, list[str], bool]] = field(default_factory=list)  # name, steps, has sorry


def scan_file(path: Path, rel: str) -> SourceFile:
    """Declarations (full names) and tactic proofs of one ``.lean`` file."""
    lines = strip_comments(path.read_text(encoding="utf8", errors="replace")).split("\n")
    out = SourceFile(rel)
    stack: list[tuple[str, list[str]]] = []  # (kind, namespace parts)
    ns = lambda: [p for _, ps in stack for p in ps]  # noqa: E731
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()
        if _indent(line) == 0 and s:
            m = re.match(r"^namespace\s+(\S+)", s)
            if m:
                stack.append(("ns", m.group(1).split(".")))
            elif re.match(r"^(?:noncomputable\s+)?section\b", s):
                stack.append(("sec", []))
            elif re.match(r"^end\b", s):
                name = s[3:].strip()
                if stack:
                    if name and stack[-1][0] == "ns" and ".".join(stack[-1][1]) != name:
                        # `end A.B` closing two pushes of `namespace A` `namespace B`
                        parts = name.split(".")
                        while stack and parts:
                            k, ps = stack.pop()
                            parts = parts[: len(parts) - len(ps)] if ps else parts
                    else:
                        stack.pop()
        m = _DECL.match(s) if _indent(line) == 0 else None
        if not m:
            i += 1
            continue
        kind, name = m.group(1), m.group(2)
        full = name[len("_root_."):] if name.startswith("_root_.") else ".".join(ns() + [name])
        if kind == "instance" and name.startswith("["):
            i += 1
            continue
        out.decls.append(full)
        j = i + 1
        while j < n and not (lines[j].strip() and _indent(lines[j]) == 0 and _TOP.match(lines[j].strip())):
            j += 1
        if kind in ("theorem", "lemma"):
            body = lines[i:j]
            for k, bl in enumerate(body):
                mb = _BY.search(bl)
                if mb is None:
                    continue
                first = mb.group(1)
                blk = []
                if first.strip():
                    blk.append((len(bl) - len(first) + _indent(first), first))
                blk += [(_indent(x), x) for x in body[k + 1:]]
                text = " ".join(t for _, t in blk)
                out.proofs.append((full, split_tactics(blk), bool(re.search(r"\b(?:sorry|admit)\b", text))))
                break
        i = j
    return out


def scan_tree(root: Path, include: Iterable[str], exclude: Iterable[str] = (), rel_prefix: str = "") -> list[SourceFile]:
    include, exclude = list(include), list(exclude)
    files = []
    for pat in include:
        for p in sorted(root.glob(pat)):
            r = rel_prefix + p.relative_to(root).as_posix()
            if any(Path(r).match(e) or r.startswith(e.rstrip("*")) for e in exclude):
                continue
            files.append(scan_file(p, r))
    return files


# --------------------------------------------------------------------------- areas


class CaseMap:
    """Module table of a case config: longest whole-folder prefix under ``root`` -> MSC code."""

    def __init__(self, root: str, areas: dict[str, str | None], default: str | None = None,
                 mirrors: Iterable[str] = ()):
        self.root = root.strip("/")
        self.areas = {k.strip("/"): (None if v is None else str(v).zfill(2)) for k, v in areas.items()}
        self.default = None if default is None else str(default).zfill(2)
        self.mirrors = tuple(mirrors)

    def mirrored(self, path: Any) -> str | None:
        """``Carleson/ToMathlib/MeasureTheory/X.lean`` -> ``Mathlib/MeasureTheory/X.lean`` for a mirror folder."""
        if not isinstance(path, str) or not self.mirrors:
            return None
        parts = [p for p in path.replace("\\", "/").split("/") if p]
        for i, p in enumerate(parts[:-1]):
            if p in self.mirrors and (i == 0 or parts[i - 1] == self.root):
                return "Mathlib/" + "/".join(parts[i + 1:])
        return None

    def area(self, path: Any) -> str | None:
        if not isinstance(path, str):
            return None
        parts = [p for p in path.replace("\\", "/").split("/") if p]
        if self.root not in parts or "Mathlib" in parts:
            return None
        rest = parts[parts.index(self.root) + 1:]
        if rest and rest[-1].endswith(".lean"):
            rest[-1] = rest[-1][:-5]
        for k in range(len(rest), 0, -1):
            key = "/".join(rest[:k])
            if key in self.areas:
                return self.areas[key]
        if rest and rest[0] in self.mirrors:
            return None
        return self.default


class MirrorAreaMap(AreaMap):
    """Mathlib folders, reading a project's ``ToMathlib``-style mirror folders as Mathlib."""

    def __init__(self, base: AreaMap, case: "CaseMap"):
        super().__init__(base.areas, base.glue)
        self.case = case

    def area(self, path: Any, glue: bool = False, level: str = "msc") -> str | None:
        a = super().area(path, glue, level)
        if a is None:
            m = self.case.mirrored(path)
            if m is not None:
                a = super().area(m, glue, level)
        return a


class OwnAreaMap(MirrorAreaMap):
    """Mathlib folders (and mirrors) first, then the project's own module table (``premise_own``)."""

    def area(self, path: Any, glue: bool = False, level: str = "msc") -> str | None:
        a = super().area(path, glue, level)
        if a is None and mathlib_relpath(path) is None and self.case.mirrored(path) is None and level == "msc":
            a = self.case.area(path)
        return a


# --------------------------------------------------------------------------- tables


def steps_table(files: list[SourceFile], source: str, min_steps: int = 1, drop_sorry: bool = True) -> pd.DataFrame:
    rows, seen = [], defaultdict(int)
    for f in files:
        for name, steps, has_sorry in f.proofs:
            if (drop_sorry and has_sorry) or len(steps) < min_steps:
                continue
            seen[name] += 1
            pid = f"{source}:{name}" if seen[name] == 1 else f"{source}:{name}#{seen[name]}"
            for i, t in enumerate(steps):
                rows.append({"proof_id": pid, "step": i, "source": source, "split": source, "theorem": name,
                             "file_path": f.rel, "tactic": t, "state_before": "", "state_after": "",
                             "true_area": None})
    return _frame(rows)[STEP_COLUMNS]


def name_index(*file_lists: list[SourceFile]) -> dict[str, str]:
    files: dict[str, set[str]] = defaultdict(set)
    for fl in file_lists:
        for f in fl:
            for d in f.decls:
                files[d].add(f.rel)
    return {n: next(iter(fs)) for n, fs in files.items() if len(fs) == 1}


def analyse(lab: pd.DataFrame, col: str, steps: pd.DataFrame, *, n_perm: int, n_boot: int, seed: int) -> dict | None:
    """`switches.analyse_source` on one label column, with one pooled Markov chain (as the PFR judge run)."""
    a = lab[lab[col].astype(str).str.len() > 0]
    seq = sw.Sequences(pd.DataFrame({"proof_id": a["proof_id"].values, "step": a["step"].values,
                                     "area": a[col].values}))
    if seq.P == 0:
        return None
    eps = noise_rate(lab, "premise")
    r = sw.analyse_source(seq, np.full(seq.P, None, dtype=object), ["shuffle", "markov", "noise"], n_perm=n_perm,
                          n_boot=n_boot, seed=seed, noise_rate=eps, markov_group="none", pooled_within="all")
    return {k: v for k, v in r.items() if not k.startswith("_")} | {"_proofs": r["_proofs"]}


def _flat(res: dict, stratum: str = "all") -> dict:
    """Headline numbers of one stratum (``all`` or a length bin) of an `analyse_source` result."""
    ent = next((e for e in res["strata"] if e["stratum"] == stratum), res["strata"][0])
    out = {"stratum": ent["stratum"], "n_proofs": ent["n_proofs"], "n_steps": ent["n_steps"]}
    for stat, e in ent["statistics"].items():
        out[stat] = e["observed"]
        out[f"{stat}_ci95"] = e["ci95"]
        for null, v in e["nulls"].items():
            out[f"{stat}_{null}"] = v["null_mean"]
            out[f"{stat}_{null}_p"] = v.get("p_two_sided")
    sr, sh = out.get("switch_rate"), out.get("switch_rate_shuffle")
    out["switch_ratio_shuffle"] = sr / sh if sr is not None and sh else None
    return out


def run_case(name: str, steps: pd.DataFrame, resolver: Resolver, own: Resolver | None, *, n_perm: int,
             n_boot: int, seed: int, out: Path) -> dict:
    lab = label_steps(steps, resolver)
    if own is not None:
        lab_own = label_steps(steps, own)
        lab["premise_own"] = lab_own["premise"].values
        lab["disagree_own"] = lab_own["disagree"].values
    out.mkdir(parents=True, exist_ok=True)
    steps.to_parquet(out / "steps.parquet", index=False)
    lab.to_parquet(out / "labels.parquet", index=False)
    per_proof = steps.groupby("proof_id").size()
    rep = {"case": name, "proofs": int(per_proof.size), "steps": int(len(steps)),
           "median_steps": float(per_proof.median()) if len(per_proof) else None,
           "share_steps_labelled": float((lab["premise"] != "").mean()) if len(lab) else None,
           "area_mix": {k: int(v) for k, v in lab.loc[lab["premise"] != "", "premise"].value_counts().head(8).items()},
           "sources": {}}
    for i, col in enumerate(["premise"] + (["premise_own"] if own is not None else [])):
        l2 = lab if col == "premise" else lab.assign(premise=lab["premise_own"], disagree=lab["disagree_own"])
        res = analyse(l2, "premise", steps, n_perm=n_perm, n_boot=n_boot, seed=seed + i)
        if res is not None:
            res.pop("_proofs").to_csv(out / f"proofs_{col}.csv", index=False)
            rep["sources"][col] = res
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=sw._jsonable))
    return rep


def load_case(cfg_path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(cfg_path).read_text()) or {}
    cfg.setdefault("include", [f"{cfg['root']}/**/*.lean"])
    cfg.setdefault("exclude", [])
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m markov_lean.source_cases")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="source-level switch test on Mathlib and case-study projects")
    r.add_argument("--mathlib", required=True, help="mathlib4 checkout (root holding Mathlib/)")
    r.add_argument("--mathlib-sample", type=int, default=0, help="keep N random Mathlib proofs (0: all)")
    r.add_argument("--case", action="append", default=[], metavar="CONFIG=CHECKOUT")
    r.add_argument("--out", required=True)
    r.add_argument("--map", help="Mathlib folder-to-MSC table (default configs/premise_msc.yaml)")
    r.add_argument("--n-perm", type=int, default=1000)
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    amap = AreaMap.load(args.map)
    ml = scan_tree(Path(args.mathlib).expanduser(), ["Mathlib/**/*.lean"])
    cases = []
    for spec in args.case:
        cfg_path, checkout = spec.split("=", 1)
        cfg = load_case(cfg_path)
        root = Path(checkout).expanduser()
        cases.append((cfg, scan_tree(root, cfg["include"], cfg["exclude"])))
    summary = {}
    base_index = name_index(ml)
    ml_steps = steps_table(ml, "mathlib")
    if args.mathlib_sample:
        keep = pd.Series(ml_steps["proof_id"].unique()).sample(
            min(args.mathlib_sample, ml_steps["proof_id"].nunique()), random_state=args.seed)
        ml_steps = ml_steps[ml_steps["proof_id"].isin(set(keep))]
    print(f"mathlib: {ml_steps['proof_id'].nunique()} proofs, {len(ml_steps)} steps")
    summary["mathlib"] = run_case("mathlib", ml_steps, Resolver(base_index, amap), None, n_perm=args.n_perm,
                                  n_boot=args.n_boot, seed=args.seed, out=out / "mathlib")
    for cfg, files in cases:
        nm = cfg["name"]
        idx = name_index(ml, files)
        st = steps_table(files, nm)
        cmap = CaseMap(cfg["root"], cfg.get("areas") or {}, cfg.get("default"), cfg.get("mirrors") or ())
        print(f"{nm}: {st['proof_id'].nunique()} proofs, {len(st)} steps from {len(files)} files")
        summary[nm] = run_case(nm, st, Resolver(idx, MirrorAreaMap(amap, cmap)), Resolver(idx, OwnAreaMap(amap, cmap)),
                               n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed, out=out / nm)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=sw._jsonable))
    tab = comparison_table(summary)
    tab.to_csv(out / "comparison.csv", index=False)
    lt = length_table(out, summary)
    lt.to_csv(out / "by_length.csv", index=False)
    try:
        figure(lt, out / "switch_by_length")
    except ImportError as e:  # pragma: no cover
        print(f"no figure ({e})")
    print(tab[["case", "labels", "n_proofs", "switch_rate", "switch_rate_shuffle", "switch_ratio_shuffle"]].to_string(index=False))
    return 0


def length_table(out: Path, names: Iterable[str]) -> pd.DataFrame:
    """Mean switches per move by proof length (labelled steps), observed and under the shuffle, per case."""
    rows = []
    for nm in names:
        p = out / nm / "proofs_premise.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        for lb, g in d.groupby("length_bin"):
            rows.append({"case": nm, "length_bin": lb, "n_proofs": len(g), "switch_rate": g["switch_rate"].mean(),
                         "shuffle": g["shuffle_switch_rate"].mean(), "n_areas": g["n_areas"].mean()})
    return pd.DataFrame(rows)


def figure(tab: pd.DataFrame, path_stem: Path, min_proofs: int = 10) -> list[Path]:
    """Switches per move by proof length for each case, Mathlib in black; shuffle dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid", font="sans-serif",
                  rc={"font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9})
    order = ["2-3", "4-7", "8-15", "16-31", "32+"]
    t = tab[tab["n_proofs"] >= min_proofs].copy()
    t["x"] = t["length_bin"].map({b: i for i, b in enumerate(order)})
    cases = [c for c in t["case"].unique() if c != "mathlib"]
    pal = dict(zip(cases, sns.color_palette("colorblind", len(cases))))
    pal["mathlib"] = (0, 0, 0)
    fig, ax = plt.subplots(figsize=(6.75, 3.0), layout="constrained")
    for c in ["mathlib"] + cases:
        g = t[t["case"] == c].sort_values("x")
        if g.empty:
            continue
        ax.plot(g["x"], g["switch_rate"], marker="o", ms=3.5, lw=1.6 if c == "mathlib" else 1.1, color=pal[c], label=c)
        ax.plot(g["x"], g["shuffle"], ls="--", lw=0.8, color=pal[c], alpha=0.6)
    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel("labelled steps in the proof")
    ax.set_ylabel("premise-area switches per move")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="lower right")
    paths = []
    for ext in ("pdf", "png"):
        p = path_stem.with_suffix("." + ext)
        fig.savefig(p, dpi=200)
        paths.append(p)
    plt.close(fig)
    return paths


def comparison_table(summary: dict) -> pd.DataFrame:
    rows = []
    for nm, rep in summary.items():
        for src, res in rep["sources"].items():
            rows.append({"case": nm, "labels": src, **_flat(res)})
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
