"""Step-level areas from the premises a tactic uses: free, deterministic labels for every proof.

``markov-lean premise-labels`` labels each tactic step by the Mathlib folders of
the lemmas and definitions it cites, maps the folders to MSC 2020 top-level
codes with the table in ``configs/premise_msc.yaml``, and (unless
``--no-switches``) runs the within-proof switch test of ``markov-lean
switches`` on those labels.  It makes no API call.

**Premises.**  Two sources, used in this order:

* ``annotated``: LeanDojo's own premise annotations, read from the raw
  benchmark JSON (``--raw-root data/raw/leandojo_benchmark_4/random``).  Each
  traced tactic has ``annotated_tactic = [text, [{full_name, def_path, ...}]]``
  and ``def_path`` is the file the premise is defined in.  This is exact.
* text: when the raw JSON is not given (the processed ``steps`` tables drop the
  annotations), cited names are read from the tactic text as ``markov-lean
  bridges`` reads them (``bridges.cited_names``) and placed in a file by an
  index of every theorem name in the given steps tables (``theorem`` ->
  ``file_path``).  A name is resolved by the first tier that succeeds:

  1. ``namespace_open``: the name inside the theorem's own namespaces
     (``sum_comm`` in ``Finset.foo`` -> ``Finset.sum_comm``), as Lean resolves it;
  2. ``exact``: the name itself;
  3. ``namespace_majority``: a dotted name whose longest known prefix (at least
     ``min_prefix`` indexed names) has one area for at least ``majority`` of them
     (``Nat.succ_le_iff`` -> the area of most ``Nat.*`` theorems);
  4. ``suffix``: a bare name that is the last component of 1 to ``max_suffix``
     indexed names, all in one area (``sum_comm`` -> ``Finset.sum_comm``).

  Tiers 1 and 2 are the ``strict`` resolver.  Lemmas proved by a term rather
  than tactics (``mul_comm``, ``le_trans``) are not in LeanDojo and stay
  unresolved unless tier 3 or 4 places them.

**Step label.**  A step's area is the majority area of its resolved premises
(ties go to the premise cited first).  Premises in ``null`` folders, outside
Mathlib, or (main labels) in ``glue`` folders count as no premise.  A step
with none gets no label.  The sources compared:

| source | rule |
|---|---|
| ``premise`` | all tiers, glue ignored, unlabelled steps dropped (the main labels) |
| ``premise_carry`` | as ``premise``, but an unlabelled step carries the previous label |
| ``premise_strict`` | tiers 1 and 2 only |
| ``premise_glue`` | glue folders keep their code |
| ``premise_folder`` | top-level Mathlib folder of the premises instead of the MSC table |

**Noise rate.**  For the switch test's noise-floor null, epsilon is the mean
share of a step's resolved premises outside the step's majority area, over
steps with two or more resolved premises: how often one premise read alone
would disagree with its own step.  That overstates the error of a majority label, so
``noise_sweep.csv`` repeats the noise null over a grid of error rates and ``switches.json``
gives the break-even rate (``noise_break_even``): the error rate at which a proof that never
moves would look as mixed as the observed proofs.

**Outputs** in ``--out``:

| file | what it holds |
|---|---|
| ``labels.parquet`` / ``.csv.gz`` | one row per step: split, proof, cited names, resolved premises, areas per source |
| ``coverage.json`` / ``coverage.csv`` | share of steps cited, resolved and labelled per split and source; tier counts |
| ``agreement.json`` | agreement with judge labels on shared steps (``--judge-labels``), or why none |
| ``switches/<split>/`` | ``markov-lean switches`` outputs for each split in ``--switch-splits``, plus ``noise_sweep.csv`` and ``by_length.csv`` / ``by_length.pdf`` (switch rate and share off the main area by proof length) |
| ``coverage.pdf`` / ``.png`` | labelled share and area mix per source (``figures`` extra) |

Caveat: premise folders measure what a step *uses*, not what it is *about*.
A topology proof that rewrites with a ring lemma switches to algebra here.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from . import switches as sw
from .bridges import _clean, _IDENT, cited_names, read_table

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = ROOT / "configs" / "premise_msc.yaml"
SOURCES = ("premise", "premise_carry", "premise_strict", "premise_glue", "premise_folder")
TIERS = ("annotated", "namespace_open", "exact", "namespace_majority", "suffix")
STRICT_TIERS = frozenset({"annotated", "namespace_open", "exact"})
SOURCE_NOTES = {
    "premise": "majority MSC area of the step's premises (all resolver tiers, glue folders ignored); "
               "steps without one dropped",
    "premise_carry": "as premise, with unlabelled steps carrying the previous step's area",
    "premise_strict": "as premise, premises resolved by exact or own-namespace name only",
    "premise_glue": "as premise, with glue folders (logic, sets, basic order and intervals, Fin, Fintype, List, "
                    "ordered-ring inequalities, basic group and ring identities) kept",
    "premise_folder": "majority top-level Mathlib folder of the premises, no MSC table",
}


# --------------------------------------------------------------------------- folder table


def mathlib_relpath(path: Any) -> str | None:
    """Path inside Mathlib without ``.lean``: ``.lake/packages/mathlib/Mathlib/Algebra/Group/Defs.lean``
    -> ``Algebra/Group/Defs``.  ``None`` outside Mathlib (Lean core, Batteries, Aesop)."""
    if not isinstance(path, str) or not path.strip():
        return None
    parts = [p for p in path.replace("\\", "/").split("/") if p and p != "."]
    if "Mathlib" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("Mathlib")
    rest = parts[i + 1:]
    if not rest:
        return None
    if rest[-1].endswith(".lean"):
        rest[-1] = rest[-1][:-5]
    return "/".join(rest)


@dataclass
class AreaMap:
    """Longest-prefix map from Mathlib folders to MSC codes (``None``: no area), with glue folders."""

    areas: dict[str, str | None]
    glue: frozenset[str] = frozenset()
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AreaMap":
        raw = yaml.safe_load(Path(path or DEFAULT_MAP).read_text()) or {}
        areas = {str(k).strip("/"): (None if v is None else str(v).zfill(2)) for k, v in (raw.get("areas") or {}).items()}
        from .msc import normalize

        bad = [k for k, v in areas.items() if v is not None and normalize(v) is None]
        if bad:
            raise ValueError(f"not MSC 2020 top-level codes in {path}: {bad}")
        return cls(areas, frozenset(str(g).strip("/") for g in raw.get("glue") or ()))

    def _match(self, rel: str, keys: Iterable[str]) -> str | None:
        parts = rel.split("/")
        keys = set(keys)
        for k in range(len(parts), 0, -1):
            pre = "/".join(parts[:k])
            if pre in keys:
                return pre
        return None

    def area(self, path: Any, glue: bool = False, level: str = "msc") -> str | None:
        """Area of a premise defined in ``path``; ``level='folder'`` gives the top-level folder instead."""
        key = (path, glue, level)
        if key in self._cache:
            return self._cache[key]
        rel = mathlib_relpath(path)
        out = None
        if rel is not None:
            hit = self._match(rel, self.areas)
            code = self.areas.get(hit) if hit else None
            is_glue = self._match(rel, self.glue) is not None
            if code is not None and (glue or not is_glue):
                out = code if level == "msc" else rel.split("/")[0]
        self._cache[key] = out
        return out


# --------------------------------------------------------------------------- name index


def build_index(tables: Iterable[pd.DataFrame]) -> dict[str, str]:
    """Theorem name -> defining file from steps tables; names defined in two files are left out."""
    files: dict[str, set[str]] = defaultdict(set)
    for t in tables:
        sub = t[["theorem", "file_path"]].drop_duplicates()
        for n, f in zip(sub["theorem"].astype(str), sub["file_path"].astype(str)):
            files[n].add(f)
    return {n: next(iter(fs)) for n, fs in files.items() if len(fs) == 1}


class Resolver:
    """Place cited names in files (and areas) with the tiers in the module notes."""

    def __init__(self, index: dict[str, str], amap: AreaMap, *, min_prefix: int = 5, majority: float = 0.6,
                 max_suffix: int = 10):
        self.index, self.amap = index, amap
        self.min_prefix, self.majority, self.max_suffix = min_prefix, majority, max_suffix
        self.prefix_files: dict[str, list[str]] = defaultdict(list)
        self.suffix_files: dict[str, list[str]] = defaultdict(list)
        for name, f in index.items():
            parts = name.split(".")
            for k in range(1, len(parts)):
                self.prefix_files[".".join(parts[:k])].append(f)
            self.suffix_files[parts[-1]].append(f)
        self._cache: dict = {}

    def _common_area(self, files: list[str], glue: bool, level: str, share: float) -> str | None:
        c = Counter(self.amap.area(f, glue, level) for f in files)
        (top, n), = c.most_common(1)
        return top if n / len(files) >= share else None

    def resolve(self, name: str, theorem: str = "") -> tuple[str | None, str | None]:
        """``(file, tier)`` when tier 1 or 2 places the name in a file, else ``(None, None)``."""
        ns = theorem.split(".")[:-1]
        for k in range(len(ns), 0, -1):
            cand = ".".join(ns[:k]) + "." + name
            if cand in self.index:
                return self.index[cand], "namespace_open"
        if name in self.index:
            return self.index[name], "exact"
        return None, None

    def areas(self, name: str, theorem: str = "") -> dict[tuple[bool, str], tuple[str | None, str | None]]:
        """For each (glue, level): (area, tier) of a cited name; tier None when unresolved."""
        key = (name, ".".join(theorem.split(".")[:-1]))
        if key in self._cache:
            return self._cache[key]
        f, tier = self.resolve(name, theorem)
        out = {}
        for glue, level in ((False, "msc"), (True, "msc"), (False, "folder")):
            if f is not None:
                out[(glue, level)] = (self.amap.area(f, glue, level), tier)
                continue
            a, t = None, None
            parts = name.split(".")
            if len(parts) > 1:
                for k in range(len(parts) - 1, 0, -1):
                    fs = self.prefix_files.get(".".join(parts[:k]))
                    if fs and len(fs) >= self.min_prefix:
                        a = self._common_area(fs, glue, level, self.majority)
                        t = "namespace_majority" if a is not None else None
                        break
            else:
                fs = self.suffix_files.get(name)
                if fs and len(fs) <= self.max_suffix:
                    a = self._common_area(fs, glue, level, 1.0)
                    t = "suffix" if a is not None else None
            out[(glue, level)] = (a, t)
        self._cache[key] = out
        return out


def ordered_cited(tactic: Any, state_before: Any = "") -> list[str]:
    """``bridges.cited_names`` in order of first appearance in the tactic text."""
    names = set(cited_names(tactic, state_before))
    if not names:
        return []
    seen, out = set(), []
    for m in _IDENT.finditer(_clean(tactic)):
        n = m.group(0)
        n = n[len("_root_."):] if n.startswith("_root_.") else n
        if n in names and n not in seen:
            seen.add(n)
            out.append(n)
    out += sorted(names - seen)
    return out


def majority(areas: list[str]) -> str | None:
    """Most common area; ties go to the one that appears first."""
    if not areas:
        return None
    c = Counter(areas)
    best = max(c.values())
    return next(a for a in areas if c[a] == best)


# --------------------------------------------------------------------------- raw LeanDojo premises


def read_annotated(root: str | Path, split: str) -> pd.DataFrame:
    """Premises per step from LeanDojo's raw ``<root>/<split>.json`` (``annotated_tactic``).

    Proof ids follow ``data.load_leandojo`` (``leandojo:<full_name>``, ``#2`` for repeats) and
    ``step`` is the index in ``traced_tactics``.  Columns: ``proof_id, step, premises, premise_paths``.
    """
    from .data import iter_leandojo_theorems

    rows, seen = [], {}
    for thm in iter_leandojo_theorems(Path(root) / f"{split}.json"):
        name = thm["full_name"]
        seen[name] = seen.get(name, 0) + 1
        pid = f"leandojo:{name}" if seen[name] == 1 else f"leandojo:{name}#{seen[name]}"
        for i, t in enumerate(thm.get("traced_tactics") or []):
            ann = t.get("annotated_tactic")
            provs = ann[1] if isinstance(ann, (list, tuple)) and len(ann) > 1 and isinstance(ann[1], list) else []
            rows.append({"proof_id": pid, "step": i,
                         "premises": [str(p.get("full_name", "")) for p in provs],
                         "premise_paths": [str(p.get("def_path", "")) for p in provs]})
    return pd.DataFrame(rows, columns=["proof_id", "step", "premises", "premise_paths"])


# --------------------------------------------------------------------------- labelling


def label_steps(steps: pd.DataFrame, resolver: Resolver, annotated: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per step with cited names, resolved premises and the area of each source (``""``: none)."""
    st = steps.assign(proof_id=steps["proof_id"].astype(str), step=pd.to_numeric(steps["step"]).astype("int64"))
    st = st.sort_values(["proof_id", "step"], kind="stable").reset_index(drop=True)
    ann = {}
    if annotated is not None and len(annotated):
        a = annotated.assign(proof_id=annotated["proof_id"].astype(str))
        ann = {(p, int(s)): (list(n), list(f)) for p, s, n, f in
               zip(a["proof_id"], a["step"], a["premises"], a["premise_paths"])}
    amap = resolver.amap
    out = defaultdict(list)
    for pid, stp, tac, sb, thm in zip(st["proof_id"], st["step"], st["tactic"], st["state_before"],
                                      st["theorem"].astype(str)):
        per: dict[str, list] = {"msc": [], "strict": [], "glue": [], "folder": []}
        tiers: list[str] = []
        if (pid, stp) in ann:
            names, paths = ann[(pid, stp)]
            for f in paths:
                tiers.append("annotated")
                per["msc"].append(amap.area(f))
                per["strict"].append(amap.area(f))
                per["glue"].append(amap.area(f, glue=True))
                per["folder"].append(amap.area(f, level="folder"))
        else:
            names = ordered_cited(tac, sb)
            for n in names:
                r = resolver.areas(n, thm)
                a, t = r[(False, "msc")]
                tiers.append(t or "")
                per["msc"].append(a)
                per["strict"].append(a if t in STRICT_TIERS else None)
                per["glue"].append(r[(True, "msc")][0])
                per["folder"].append(r[(False, "folder")][0])
        clean = {k: [x for x in v if x] for k, v in per.items()}
        out["proof_id"].append(pid)
        out["step"].append(stp)
        out["n_cited"].append(len(names))
        out["cited"].append(" ".join(names))
        out["tiers"].append(" ".join(t or "-" for t in tiers))
        out["n_resolved"].append(len(clean["msc"]))
        out["premise_areas"].append(" ".join(clean["msc"]))
        maj = majority(clean["msc"])
        out["premise"].append(maj or "")
        out["premise_strict"].append(majority(clean["strict"]) or "")
        out["premise_glue"].append(majority(clean["glue"]) or "")
        out["premise_folder"].append(majority(clean["folder"]) or "")
        out["disagree"].append(float(np.mean([x != maj for x in clean["msc"]])) if len(clean["msc"]) >= 2 else np.nan)
        out["glue_disagree"].append(_disagree(clean["glue"]))
        out["strict_disagree"].append(_disagree(clean["strict"]))
        out["folder_disagree"].append(_disagree(clean["folder"]))
    lab = pd.DataFrame(out)
    keep = [c for c in ("split", "theorem", "file_path") if c in st.columns]
    lab = pd.concat([st[keep].reset_index(drop=True), lab], axis=1)
    lab["premise_carry"] = carry_forward(lab, "premise")
    return lab


def _disagree(xs: list[str]) -> float:
    if len(xs) < 2:
        return np.nan
    m = majority(xs)
    return float(np.mean([x != m for x in xs]))


def carry_forward(lab: pd.DataFrame, col: str) -> pd.Series:
    """``col`` with empty steps filled by the previous labelled step of the same proof (leading ones stay empty)."""
    s = lab[col].replace("", np.nan)
    return s.groupby(lab["proof_id"], sort=False).ffill().fillna("").astype(str)


NOISE_COL = {"premise": "disagree", "premise_carry": "disagree", "premise_strict": "strict_disagree",
             "premise_glue": "glue_disagree", "premise_folder": "folder_disagree"}


def noise_rate(lab: pd.DataFrame, source: str) -> float | None:
    d = lab[NOISE_COL[source]].dropna()
    return float(d.mean()) if len(d) else None


def area_table(lab: pd.DataFrame, source: str) -> pd.DataFrame:
    a = lab[lab[source].astype(str).str.len() > 0]
    return pd.DataFrame({"proof_id": a["proof_id"].values, "step": a["step"].values, "area": a[source].values})


NOISE_GRID = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)


def noise_sweep(lab: pd.DataFrame, source: str, grid: Iterable[float] = NOISE_GRID, R: int = 50,
                seed: int = 0) -> pd.DataFrame:
    """The noise-floor null at each label error rate in ``grid``: mean statistics over proofs, against observed.

    The error rate at which the null's share off the main area reaches the observed share is the largest
    labeller error under which the observed movement could still be a proof that never moves.
    """
    seq = sw.Sequences(area_table(lab, source))
    if seq.P == 0:
        return pd.DataFrame()
    obs = {k: float(v[0].mean()) for k, v in sw.per_proof_stats(seq, seq.y[None]).items() if k in sw.STATS}
    rows = []
    for j, eps in enumerate(grid):
        rng = np.random.default_rng([seed, j])
        pp = sw.per_proof_stats(seq, sw.simulate_noise(seq, float(eps), R, rng))
        rows.append({"source": source, "epsilon": float(eps), "n_proofs": seq.P,
                     **{f"null_{s}": float(pp[s].mean()) for s in sw.STATS},
                     **{f"observed_{s}": obs[s] for s in sw.STATS}})
    return pd.DataFrame(rows)


def break_even(sweep: pd.DataFrame, stat: str = "off_mode") -> float | None:
    """Smallest error rate at which the noise null's ``stat`` reaches the observed value (linear interpolation)."""
    if sweep.empty:
        return None
    x, y, o = sweep["epsilon"].to_numpy(), sweep[f"null_{stat}"].to_numpy(), float(sweep[f"observed_{stat}"].iat[0])
    above = np.flatnonzero(y >= o)
    if not len(above):
        return None
    i = above[0]
    if i == 0:
        return float(x[0])
    return float(x[i - 1] + (o - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))


# --------------------------------------------------------------------------- coverage and agreement


def coverage(lab: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    groups = [("all", lab)] + ([(s, g) for s, g in lab.groupby("split", sort=False)] if "split" in lab else [])
    for split, g in groups:
        n = len(g)
        base = {"split": split, "steps": n, "proofs": int(g["proof_id"].nunique()),
                "share_citing": float((g["n_cited"] > 0).mean()) if n else np.nan}
        for src in SOURCES:
            labelled = g[src].astype(str).str.len() > 0
            per_proof = labelled.groupby(g["proof_id"]).sum()
            rows.append({**base, "source": src, "share_labelled": float(labelled.mean()) if n else np.nan,
                         "proofs_2plus_labelled": int((per_proof >= 2).sum()),
                         "n_areas": int(g.loc[labelled, src].nunique()),
                         "noise_rate": noise_rate(g, src)})
    tab = pd.DataFrame(rows)
    tiers = Counter(t for s in lab["tiers"] for t in s.split() if s)
    mix = lab.loc[lab["premise"] != "", "premise"].value_counts(normalize=True)
    return {"by_split": tab.to_dict("records"), "tiers": dict(tiers),
            "premise_area_mix": {k: float(v) for k, v in mix.items()}}, tab


def judge_agreement(lab: pd.DataFrame, labels_path: str | Path | None) -> dict:
    """Step-level agreement of each source with judge MSC labels on shared steps."""
    if labels_path is None:
        return {"available": False, "reason": "no --judge-labels given"}
    p = Path(labels_path)
    if not p.exists():
        return {"available": False, "reason": f"{p} not found"}
    from sklearn.metrics import cohen_kappa_score, normalized_mutual_info_score

    from .bridges import judge_areas

    j, used = judge_areas(read_table(p))
    out: dict[str, Any] = {"available": True, "labels": str(p), "judge": used, "sources": {}}
    for src in ("premise", "premise_carry", "premise_strict", "premise_glue"):
        m = area_table(lab, src).merge(j.rename(columns={"area": "judge"}), on=["proof_id", "step"])
        if m.empty:
            out["sources"][src] = {"n_steps": 0}
            continue
        a, b = m["area"].astype(str), m["judge"].astype(str)
        pm = m.groupby("proof_id").agg(pa=("area", lambda s: s.mode().iat[0]), pj=("judge", lambda s: s.mode().iat[0]))
        out["sources"][src] = {"n_steps": int(len(m)), "n_proofs": int(m["proof_id"].nunique()),
                               "accuracy": float((a == b).mean()), "kappa": float(cohen_kappa_score(a, b)),
                               "nmi": float(normalized_mutual_info_score(b, a)),
                               "proof_mode_accuracy": float((pm["pa"].astype(str) == pm["pj"].astype(str)).mean())}
    return out


# --------------------------------------------------------------------------- switch test


def run_switches(lab: pd.DataFrame, steps: pd.DataFrame, out: Path, *, sources: Iterable[str] = SOURCES,
                 nulls: list[str] | None = None, n_perm: int = 1000, n_boot: int = 2000, seed: int = 0,
                 min_proofs: int = 30, figures: bool = True, verbose: bool = True, title: str = "",
                 inputs: dict | None = None) -> dict:
    """``markov-lean switches`` on the premise sources of ``lab``; writes the same files as that command.

    ``inputs`` (the split and the tactic-step cut, say) is kept in ``switches.json`` for ``markov-lean paper``.
    """
    say = print if verbose else (lambda *a, **k: None)
    nulls = list(nulls or sw.NULLS)
    stp = steps.assign(proof_id=steps["proof_id"].astype(str))
    results, eps_note = {}, {}
    for i, src in enumerate(sources):
        seq = sw.Sequences(area_table(lab, src))
        if seq.P == 0:
            say(f"premise-switches: {src}: no proof with two labelled steps; skipped")
            continue
        eps = noise_rate(lab, src)
        eps_note[src] = ("mean share of a step's premises outside its majority area (steps with 2+ premises)"
                         if eps is not None else "none")
        say(f"premise-switches: {src}: {seq.P} proofs, {seq.N} steps, {seq.K} areas")
        results[src] = sw.analyse_source(seq, sw._folders(seq, stp), nulls, n_perm=n_perm, n_boot=n_boot,
                                         seed=seed + 1000 * i, noise_rate=eps, min_proofs=min_proofs)
    out.mkdir(parents=True, exist_ok=True)
    sweep = pd.concat([noise_sweep(lab, s, seed=seed) for s in results] or [pd.DataFrame()], ignore_index=True)
    sweep.to_csv(out / "noise_sweep.csv", index=False)
    even = {s: {st: break_even(sweep[sweep["source"] == s], st) for st in ("off_mode", "has_switch")}
            for s in results} if len(sweep) else {}
    report = {
        "run": title or str(out), "inputs": dict(inputs or {}), "notes": {s: SOURCE_NOTES[s] for s in results},
        "noise_break_even": even,
        "noise_rate_source": eps_note,
        "settings": {"n_perm": n_perm, "n_boot": n_boot, "seed": seed, "nulls": nulls, "markov_group": "folder",
                     "pooled_within": "folder", "min_proofs": min_proofs, "same_proofs": False,
                     "length_edges": list(sw.LENGTH_EDGES), "resampling_unit": "proof"},
        "headline": sw.headline(results),
        "sources": {k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for k, r in results.items()},
    }
    (out / "switches.json").write_text(json.dumps(report, indent=2, default=sw._jsonable))
    sw.long_table(results).to_csv(out / "switches.csv", index=False)
    pd.concat([r["_proofs"].assign(source=k) for k, r in results.items()]).to_csv(out / "proofs.csv", index=False)
    dw = sw.dwell_table(results)
    dw.to_csv(out / "dwell.csv", index=False)
    (out / "table_switches.tex").write_text(sw.table_tex(results))
    (out / "REPORT.md").write_text(sw.markdown(report, results))
    lt = length_table(results)
    lt.to_csv(out / "by_length.csv", index=False)
    if figures:
        try:
            sw.figure(results, dw, out / "switches")
            if "premise" in results:
                _save(draw_by_length(lt), out / "by_length")
        except ImportError as e:  # pragma: no cover
            say(f"premise-switches: no figure ({e})")
    return report


# --------------------------------------------------------------------------- figure


def _short(title: str, width: int = 30) -> str:
    """An MSC title cut at a word boundary: ``Measure and integration`` stays, long ones end in an ellipsis."""
    t = title.split(";")[0].split(",")[0]
    if len(t) <= width:
        return t
    return t[:width].rsplit(" ", 1)[0].rstrip(" ,") + "…"


def draw_coverage(cov: pd.DataFrame, mix: dict[str, float]):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .figures import paper_rc
    from .msc import MSC2020_TOP

    with matplotlib.rc_context(paper_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), layout="constrained")
        c = cov[cov["split"] != "all"].copy()
        c["source"] = c["source"].map(lambda s: sw.SOURCE_LABELS.get(s, s))
        sns.barplot(data=c, y="source", x="share_labelled", hue="split", ax=axes[0])
        axes[0].set_xlabel("Share of steps labelled")
        axes[0].set_ylabel("")
        axes[0].legend(title="", frameon=False, fontsize="x-small")
        m = pd.Series(mix).sort_values(ascending=False).head(15)
        sns.barplot(x=m.values, y=[f"{k} {_short(MSC2020_TOP.get(k, ''))}" for k in m.index], ax=axes[1],
                    color=sns.color_palette()[0])
        axes[1].set_xlabel("Share of labelled steps (premise)")
        for ax in axes:
            sns.despine(ax=ax)
    return fig


def length_table(results: dict[str, dict]) -> pd.DataFrame:
    """Observed and null means of each statistic by proof length (labelled steps), per source."""
    rows = []
    for src, r in results.items():
        for e in r["strata"]:
            if e["kind"] != "length":
                continue
            for st, v in e["statistics"].items():
                rows.append({"source": src, "length": e["stratum"], "n_proofs": e["n_proofs"], "statistic": st,
                             "series": "observed", "value": v["observed"], "lo": v["ci95"][0], "hi": v["ci95"][1]})
                for n, d in v["nulls"].items():
                    rows.append({"source": src, "length": e["stratum"], "n_proofs": e["n_proofs"], "statistic": st,
                                 "series": n, "value": d["null_mean"], "lo": np.nan, "hi": np.nan})
    return pd.DataFrame(rows)


def draw_by_length(tab: pd.DataFrame, source: str = "premise"):
    """Switch rate and share off the main area by proof length: observed (with 95% CI) against each null."""
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .figures import paper_rc

    t = tab[tab["source"] == source]
    order = list(dict.fromkeys(t["length"]))
    with matplotlib.rc_context(paper_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), layout="constrained")
        series = [*[n for n in sw.NULLS if n in set(t["series"])], "observed"]   # observed drawn on top
        pal = dict(zip(["observed", *sw.NULLS], sns.color_palette(n_colors=1 + len(sw.NULLS))))
        for ax, st in zip(axes, ("switch_rate", "off_mode")):
            d = t[t["statistic"] == st]
            for s in series:
                ds = d[d["series"] == s].set_index("length").reindex(order)
                x = np.arange(len(order))
                ax.plot(x, ds["value"], marker="o", ms=3, color=pal[s], ls="-" if s == "observed" else "--",
                        lw=1.6 if s == "observed" else 1.0, zorder=3 if s == "observed" else 2,
                        label="Observed" if s == "observed" else sw.NULL_LABELS[s])
                if s == "observed":
                    ax.fill_between(x, ds["lo"], ds["hi"], color=pal[s], alpha=0.2, lw=0)
            ax.set_xticks(range(len(order)), order)
            ax.set_xlabel("Proof length (labelled steps)")
            ax.set_ylabel(sw.STAT_LABELS[st])
            sns.despine(ax=ax)
        axes[1].legend(frameon=False, fontsize="x-small")
    return fig


def _save(fig, stem: Path) -> list[Path]:
    import matplotlib
    import matplotlib.pyplot as plt

    from .figures import paper_rc

    paths = []
    with matplotlib.rc_context(paper_rc()):
        for suf in (".pdf", ".png"):
            fig.savefig(stem.with_suffix(suf))
            paths.append(stem.with_suffix(suf))
    plt.close(fig)
    return paths


def write_coverage_figure(cov: pd.DataFrame, mix: dict[str, float], stem: Path) -> list[Path]:
    return _save(draw_coverage(cov, mix), stem)


# --------------------------------------------------------------------------- run


def run(steps_paths: list[str | Path], out: str | Path, *, index_paths: list[str | Path] | None = None,
        raw_root: str | Path | None = None, map_path: str | Path | None = None, judge_labels=None,
        switch_splits: list[str] | None = None, min_switch_steps: int = 1, nulls: list[str] | None = None,
        n_perm: int = 1000, n_boot: int = 2000, seed: int = 0, min_proofs: int = 30, figures: bool = True,
        switches: bool = True, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = [read_table(Path(p)) for p in steps_paths]
    steps = pd.concat(tables, ignore_index=True)
    if "split" not in steps.columns:
        steps["split"] = "all"
    idx_tables = tables + [read_table(Path(p)) for p in (index_paths or [])]
    index = build_index(idx_tables)
    amap = AreaMap.load(map_path)
    resolver = Resolver(index, amap)
    annotated = None
    if raw_root is not None:
        annotated = pd.concat([read_annotated(raw_root, s) for s in steps["split"].astype(str).unique()],
                              ignore_index=True)
        say(f"premise-labels: {len(annotated)} annotated steps from {raw_root}")
    say(f"premise-labels: {len(steps)} steps, {steps['proof_id'].nunique()} proofs, index of {len(index)} names")
    lab = label_steps(steps, resolver, annotated)
    lab.to_parquet(out / "labels.parquet", index=False)
    cov, cov_tab = coverage(lab)
    cov.update({"map": str(map_path or DEFAULT_MAP), "index_names": len(index),
                "premise_source": "annotated" if annotated is not None else "tactic text",
                "steps_tables": [str(p) for p in steps_paths]})
    (out / "coverage.json").write_text(json.dumps(cov, indent=2, default=sw._jsonable))
    cov_tab.to_csv(out / "coverage.csv", index=False)
    agree = judge_agreement(lab, judge_labels)
    (out / "agreement.json").write_text(json.dumps(agree, indent=2))
    if figures:
        try:
            write_coverage_figure(cov_tab, cov["premise_area_mix"], out / "coverage")
        except ImportError as e:  # pragma: no cover
            say(f"premise-labels: no figure ({e})")
    for r in cov_tab[cov_tab["split"] == "all"].itertuples():
        say(f"premise-labels: {r.source}: {r.share_labelled:.1%} of steps labelled")
    reports = {}
    if switches:
        for split in switch_splits or sorted(steps["split"].astype(str).unique()):
            g = lab[lab["split"].astype(str) == split] if split != "all" else lab
            if min_switch_steps > 1:
                n = steps[steps["split"].astype(str) == split].groupby("proof_id")["step"].size() if split != "all" \
                    else steps.groupby("proof_id")["step"].size()
                g = g[g["proof_id"].map(n).fillna(0) >= min_switch_steps]
            stp = steps[steps["split"].astype(str) == split] if split != "all" else steps
            reports[split] = run_switches(g, stp, out / "switches" / split, nulls=nulls, n_perm=n_perm,
                                          n_boot=n_boot, seed=seed, min_proofs=min_proofs, figures=figures,
                                          verbose=verbose, title=f"premise labels, {split}",
                                          inputs={"split": split, "min_switch_steps": min_switch_steps})
    say(f"wrote {out}")
    return {"coverage": cov, "agreement": agree, "switches": {k: v["headline"] for k, v in reports.items()}}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="markov-lean premise-labels",
                                description="Label tactic steps by the Mathlib folders of the premises they use, "
                                            "map folders to MSC areas, and run the within-proof switch test.")
    p.add_argument("--steps", nargs="+", required=True, help="steps tables to label (e.g. leandojo_train.parquet)")
    p.add_argument("--index-steps", nargs="*", default=None,
                   help="more steps tables whose theorem names join the name index (not labelled)")
    p.add_argument("--raw-root", default=None,
                   help="raw LeanDojo split folder (holds <split>.json) for exact premise annotations")
    p.add_argument("--map", default=None, help=f"folder-to-MSC table (default {DEFAULT_MAP.relative_to(ROOT)})")
    p.add_argument("--out", required=True)
    p.add_argument("--judge-labels", default=None, help="judge labels.parquet, for step-level agreement")
    p.add_argument("--switch-splits", nargs="*", default=None, help="splits to test (default every split given)")
    p.add_argument("--min-switch-steps", type=int, default=1,
                   help="only test proofs with at least this many tactic steps (labelled or not)")
    p.add_argument("--nulls", nargs="+", choices=list(sw.NULLS), default=None)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-proofs", type=int, default=30)
    p.add_argument("--no-switches", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    run(a.steps, a.out, index_paths=a.index_steps, raw_root=a.raw_root, map_path=a.map, judge_labels=a.judge_labels,
        switch_splits=a.switch_splits, min_switch_steps=a.min_switch_steps, nulls=a.nulls, n_perm=a.n_perm,
        n_boot=a.n_boot, seed=a.seed, min_proofs=a.min_proofs, figures=not a.no_figures,
        switches=not a.no_switches)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
