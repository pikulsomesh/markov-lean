"""A static results explorer: one HTML file to browse proofs, area paths and the model.

``markov-lean explore --results results/pilot3 --out site``

writes ``site/index.html``, a single self-contained page (inline CSS, JS and
data; nothing is fetched when it is viewed) that opens from disk, so it can ship
as supplementary material. It shows:

* **Proofs**: a searchable list; for each proof, its goal state before every
  step with the tactic, the judges' MSC area and the Markov-model macrostate,
  the two area paths drawn as strips, and links to proofs with the same switch
  pattern or sharing a switch.
* **Model**: the run's headline numbers, the coarse-grained transition matrix,
  the PCCA+ sets (size, stationary share, stay probability, membership) and
  which judge areas fall in each set.
* **Patterns**: the area paths that switch at least once, most common first.

Inputs are read from the results folder and any of them may be missing, so the
page can be built from a run in progress:

| input | default location | used for |
|---|---|---|
| assignments, ``coarse_T.npy``, ``msm_report.json`` | ``<results>/msm/`` | macrostates, the matrix and the sets |
| steps (``docs/schema.md``) | ``<results>/judge/sample_steps.parquet``, or ``--steps`` | goal states and tactics |
| labels | ``<results>/judge/labels.parquet``, or ``--labels`` | judge areas (the consensus, as in ``bridges``) |
| ``pilot_report.json`` | ``<results>/`` | gate 1 and agreement numbers |

On synthetic steps (a ``true_area`` column) with no labels, the hidden area
stands in for the judges, as in the pilot.

The page holds no names of people, no machine paths and no timestamps: only
Lean/Mathlib names, goal states, tactics, model numbers and file names relative
to the results folder. ``--max-proofs`` and ``--max-state-chars`` keep it small.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import msc

KEY = ["proof_id", "step"]
SWITCH_SEP = ">"
MAX_RELATED = 6
MAX_POSTING = 300
MAX_PATTERNS = 300

REPORT_FIELDS = {
    "data": ["source", "split", "n_proofs", "n_steps"],
    "msm": ["n_microstates", "lag", "n_macrostates", "all_checks_passed"],
}


# --------------------------------------------------------------------------- inputs


def _first(*paths: Path | None) -> Path | None:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def _read(path: Path) -> pd.DataFrame:
    from .bridges import read_table

    return read_table(path)


def _keyed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["proof_id"] = df["proof_id"].astype(str)
    df["step"] = pd.to_numeric(df["step"]).astype("int64")
    return df


def _rel(path: Path | None, base: Path) -> str | None:
    """A path as shown on the page: relative to the results folder, else the file name only."""
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def _clean(x: Any) -> Any:
    """JSON-safe values: numpy scalars to Python, NaN and inf to None."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


def load_inputs(results: Path, steps_path: Path | None = None, labels_path: Path | None = None,
                msm_dir: Path | None = None) -> dict:
    """Whatever of the run exists, as tables plus a note of what was found."""
    results = Path(results)
    msm_dir = _first(msm_dir, results / "msm", results if (results / "assignments.parquet").exists() else None)
    found: dict[str, Any] = {"msm": _rel(msm_dir, results) if msm_dir else None}
    out: dict[str, Any] = {"found": found}

    out["assignments"] = None
    out["coarse_T"] = None
    out["msm_report"] = {}
    out["memberships"] = None
    if msm_dir is not None:
        ap = _first(msm_dir / "assignments.parquet", msm_dir / "assignments.csv")
        if ap is not None:
            out["assignments"] = _keyed(_read(ap))
        tp = msm_dir / "coarse_T.npy"
        if tp.exists():
            out["coarse_T"] = np.load(tp)
        rp = msm_dir / "msm_report.json"
        if rp.exists():
            out["msm_report"] = json.loads(rp.read_text())

    sp = _first(steps_path, results / "judge" / "sample_steps.parquet", results / "judge" / "sample_steps.csv",
                results / "steps.parquet")
    out["steps"] = _keyed(_read(sp)) if sp is not None else None
    found["steps"] = _rel(sp, results)

    lp = _first(labels_path, results / "judge" / "labels.parquet", results / "judge" / "labels.csv",
                results / "labels.parquet")
    out["labels"] = _keyed(_read(lp)) if lp is not None else None
    found["labels"] = _rel(lp, results)

    pr = results / "pilot_report.json"
    out["pilot_report"] = json.loads(pr.read_text()) if pr.exists() else {}
    found["pilot_report"] = "pilot_report.json" if pr.exists() else None
    return out


# --------------------------------------------------------------------------- areas


def step_areas(steps: pd.DataFrame | None, labels: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
    """One area per labelled step (``proof_id``, ``step``, ``area``, ``confidence``) and where it came from."""
    empty = pd.DataFrame(columns=KEY + ["area", "confidence"])
    if labels is not None and len(labels):
        from .bridges import judge_areas

        lab = labels[labels["step"] >= 0]
        if lab.empty:
            return empty, "no step-level labels"
        areas, used = judge_areas(lab)
        judges = [j for j in dict.fromkeys(lab["judge"].astype(str)) if j != "consensus"]
        conf = lab.groupby(KEY)["confidence"].mean().rename("confidence").reset_index() \
            if "confidence" in lab.columns else None
        if conf is not None:
            areas = areas.merge(conf, on=KEY, how="left")
        else:
            areas["confidence"] = np.nan
        if used.startswith("consensus of"):
            note = f"consensus of {len(judges)} judges (steps where they agree)"
        elif used == "consensus":
            note = "judge consensus"
        else:
            note = "one judge"
        return areas, note
    if steps is not None and "true_area" in steps.columns and steps["true_area"].notna().any():
        t = steps[steps["true_area"].notna()]
        areas = pd.DataFrame({"proof_id": t["proof_id"].values, "step": t["step"].values,
                              "area": t["true_area"].astype(str).str.zfill(2).values, "confidence": np.nan})
        return areas.drop_duplicates(KEY), "hidden true area (synthetic data)"
    return empty, "none yet"


def judge_votes(labels: pd.DataFrame | None) -> dict[tuple[str, int], str]:
    """Per step, the raw judges' codes joined with ' / ' when they disagree (judges are not named)."""
    if labels is None or labels.empty:
        return {}
    lab = labels[(labels["step"] >= 0) & (labels["judge"].astype(str) != "consensus")]
    if lab["judge"].nunique() < 2:
        return {}
    lab = lab.assign(msc=lab["msc"].astype(str).str.zfill(2))
    out = {}
    for (pid, step), codes in lab.groupby(KEY)["msc"]:
        uniq = list(dict.fromkeys(codes))
        if len(uniq) > 1:
            out[(str(pid), int(step))] = " / ".join(uniq)
    return out


def proof_labels(labels: pd.DataFrame | None) -> dict[str, str]:
    """Whole-proof labels (step -1): the code the judges agree on, or their codes joined."""
    if labels is None or labels.empty:
        return {}
    lab = labels[labels["step"] < 0]
    out = {}
    for pid, codes in lab.assign(msc=lab["msc"].astype(str).str.zfill(2)).groupby("proof_id")["msc"]:
        uniq = list(dict.fromkeys(codes))
        out[str(pid)] = uniq[0] if len(uniq) == 1 else " / ".join(uniq)
    return out


def runs(seq: list) -> list:
    """Collapse repeats and drop gaps: [26, 26, None, 11, 11, 26] -> [26, 11, 26]."""
    out: list = []
    for v in seq:
        if v is None:
            continue
        if not out or out[-1] != v:
            out.append(v)
    return out


def switches(seq: list) -> list[str]:
    """Distinct area moves of a path, in order of first appearance, as 'a>b'."""
    path = runs(seq)
    return list(dict.fromkeys(f"{a}{SWITCH_SEP}{b}" for a, b in zip(path, path[1:])))


def related(paths: list[list], max_related: int = MAX_RELATED) -> list[list[list]]:
    """For each proof, other proofs with the same switch path first, then the most shared switches.

    Returns, per proof, ``[[other_index, n_shared_switches, same_path], ...]``.
    """
    sw = [switches(p) for p in paths]
    keys = [SWITCH_SEP.join(map(str, runs(p))) for p in paths]
    posting: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sw):
        for pair in s:
            if len(posting[pair]) < MAX_POSTING:
                posting[pair].append(i)
    out = []
    for i, s in enumerate(sw):
        if not s:
            out.append([])
            continue
        shared: Counter = Counter()
        for pair in s:
            for j in posting[pair]:
                if j != i:
                    shared[j] += 1
        ranked = sorted(shared.items(), key=lambda kv: (keys[kv[0]] != keys[i], -kv[1],
                                                         abs(len(paths[kv[0]]) - len(paths[i])), kv[0]))
        out.append([[j, n, keys[j] == keys[i]] for j, n in ranked[:max_related]])
    return out


# --------------------------------------------------------------------------- model


def stationary(T: np.ndarray) -> np.ndarray | None:
    try:
        w, v = np.linalg.eig(T.T)
        pi = np.real(v[:, np.argmin(np.abs(w - 1))])
        pi = pi / pi.sum()
        return pi if np.all(np.isfinite(pi)) else None
    except np.linalg.LinAlgError:
        return None


def model_section(inp: dict, areas: pd.DataFrame, top_areas: int = 4) -> dict | None:
    a, T, rep = inp["assignments"], inp["coarse_T"], inp["msm_report"]
    if a is None and T is None:
        return None
    k = int(T.shape[0]) if T is not None else int(a["macrostate"].max()) + 1
    sets = []
    a_ok = a[a["macrostate"] >= 0] if a is not None else None
    pi = stationary(T) if T is not None else None
    crosstab_codes: list[str] = []
    crosstab: list[list[int]] = []
    if a_ok is not None and len(areas):
        m = a_ok.merge(areas[KEY + ["area"]], on=KEY, how="inner")
        crosstab_codes = list(m["area"].value_counts().index[:15])
        ct = pd.crosstab(m["macrostate"], m["area"]).reindex(index=range(k), columns=crosstab_codes, fill_value=0)
        crosstab = ct.astype(int).values.tolist()
    else:
        m = None
    for s in range(k):
        row: dict[str, Any] = {"id": s}
        if a_ok is not None:
            sub = a_ok[a_ok["macrostate"] == s]
            row["n_steps"] = int(len(sub))
            row["share_steps"] = float(len(sub) / max(len(a_ok), 1))
            row["n_microstates"] = int(sub["microstate"].nunique()) if "microstate" in sub else None
            row["mean_membership"] = float(sub["membership"].mean()) if "membership" in sub and len(sub) else None
            row["n_proofs"] = int(sub["proof_id"].nunique())
        if T is not None:
            row["stay"] = float(T[s, s])
        if pi is not None:
            row["stationary"] = float(pi[s])
        if m is not None:
            vc = m.loc[m["macrostate"] == s, "area"].value_counts(normalize=True)
            row["top_areas"] = [[str(c), float(p)] for c, p in vc.iloc[:top_areas].items()]
        sets.append(row)
    checks = {name: bool(c.get("passed")) for name, c in (rep.get("checks") or {}).items() if isinstance(c, dict)}
    return _clean({
        "k": k, "lag": rep.get("lag"), "n_microstates": rep.get("n_microstates"),
        "all_checks_passed": rep.get("all_checks_passed"), "checks": checks,
        "T": T if T is not None else None, "sets": sets,
        "crosstab": {"areas": crosstab_codes, "rows": crosstab} if crosstab_codes else None,
        "n_unassigned_steps": int((a["macrostate"] < 0).sum()) if a is not None else None,
    })


def run_summary(report: dict) -> dict:
    """Whitelisted numbers from pilot_report.json; never its file paths."""
    out: dict[str, Any] = {}
    for sec, keys in REPORT_FIELDS.items():
        if isinstance(report.get(sec), dict):
            out[sec] = {k: report[sec][k] for k in keys if k in report[sec]}
    if report.get("msm_checks_failed") is not None:
        out["msm_checks_failed"] = list(report["msm_checks_failed"])
    ev = report.get("eval")
    if isinstance(ev, dict):
        g = ev.get("gate1") or {}
        out["gate1"] = {k: g.get(k) for k in ("passed", "k", "macrostate_nmi", "folder_nmi", "threshold", "rule")}
        out["n_labelled_steps"] = ev.get("n_labelled_steps")
        for name in ("macrostates", "microstates", "folder_baseline"):
            if isinstance(ev.get(name), dict):
                out.setdefault("agreement", {})[name] = {k: ev[name].get(k) for k in ("nmi", "ari") if k in ev[name]}
        sweep = ev.get("macrostate_sweep") or {}
        if sweep:
            out["sweep"] = {str(k): {"nmi": v.get("nmi"), "ari": v.get("ari")} for k, v in sweep.items()}
        nxt = ev.get("next_area") or {}
        out["next_area"] = {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, bool))}
                            for k, v in nxt.items() if isinstance(v, dict)}
    elif report.get("gate1"):
        out["gate1_status"] = str(report["gate1"])
    agree = report.get("judge_agreement")
    if isinstance(agree, dict):
        out["judge_agreement"] = {lvl: {k: agree[lvl].get(k) for k in ("n", "agreement", "kappa")}
                                  for lvl in ("step", "proof") if isinstance(agree.get(lvl), dict)}
        out["n_judges"] = len(agree.get("judges") or [])
    return _clean(out)


# --------------------------------------------------------------------------- build


def _theorem(pid: str) -> str:
    return pid.split(":", 1)[1] if ":" in pid else pid


def _trim(text: Any, n: int) -> tuple[str, bool]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return "", False
    s = str(text)
    return (s, False) if n <= 0 or len(s) <= n else (s[:n].rstrip() + " …", True)


def pick_proofs(ids: list[str], has_text: set[str], has_area: set[str], max_proofs: int, seed: int) -> list[str]:
    """Proofs with goal text and areas first, then text only, then the rest; random within each tier."""
    rng = random.Random(seed)
    tiers: list[list[str]] = [[], [], []]
    for pid in ids:
        tiers[0 if (pid in has_text and pid in has_area) else 1 if pid in has_text else 2].append(pid)
    chosen: list[str] = []
    for t in tiers:
        t = sorted(t)
        rng.shuffle(t)
        chosen.extend(t)
    if max_proofs and max_proofs > 0:
        chosen = chosen[:max_proofs]
    return sorted(chosen)


def build_data(inp: dict, max_proofs: int = 500, max_state_chars: int = 600, seed: int = 0,
               title: str | None = None) -> dict:
    steps, a, labels = inp["steps"], inp["assignments"], inp["labels"]
    areas, area_note = step_areas(steps, labels)
    votes = judge_votes(labels)
    whole = proof_labels(labels)

    ids: set[str] = set()
    for t in (steps, a):
        if t is not None:
            ids |= set(t["proof_id"].astype(str))
    has_text = set(steps["proof_id"]) if steps is not None else set()
    has_area = set(areas["proof_id"])
    chosen = pick_proofs(sorted(ids), has_text, has_area, max_proofs, seed)
    keep = set(chosen)

    area_of = {(r.proof_id, int(r.step)): (r.area, r.confidence) for r in areas.itertuples(index=False)
               if r.proof_id in keep}
    macro_of = {}
    if a is not None:
        sub = a[a["proof_id"].isin(keep)]
        memb = sub["membership"] if "membership" in sub else pd.Series(np.nan, index=sub.index)
        for pid, st, mac, mb in zip(sub["proof_id"], sub["step"], sub["macrostate"], memb):
            macro_of[(pid, int(st))] = (int(mac) if int(mac) >= 0 else None, mb)
    text_of: dict[tuple[str, int], tuple] = {}
    meta: dict[str, tuple[str, str]] = {}
    if steps is not None:
        sub = steps[steps["proof_id"].isin(keep) & (steps["step"] >= 0)]
        cols = {c: sub[c] if c in sub else pd.Series("", index=sub.index)
                for c in ("tactic", "state_before", "theorem", "file_path")}
        for pid, st, tac, state, th, fp in zip(sub["proof_id"], sub["step"], cols["tactic"], cols["state_before"],
                                               cols["theorem"], cols["file_path"]):
            text_of[(pid, int(st))] = (tac, state)
            if pid not in meta:
                meta[pid] = (str(th) if isinstance(th, str) and th else _theorem(pid),
                             str(fp) if isinstance(fp, str) else "")

    step_keys: dict[str, set[int]] = defaultdict(set)
    for (pid, st) in list(text_of) + list(macro_of) + list(area_of):
        step_keys[pid].add(st)

    proofs, area_paths, macro_paths = [], [], []
    truncated = 0
    for pid in chosen:
        rows = []
        for st in sorted(step_keys[pid]):
            tac, state = text_of.get((pid, st), ("", ""))
            tac, _ = _trim(tac, 300)
            state, cut = _trim(state, max_state_chars)
            truncated += cut
            ar, conf = area_of.get((pid, st), (None, None))
            mac, mb = macro_of.get((pid, st), (None, None))
            rows.append([st, tac, state, ar, conf, votes.get((pid, st)), mac, mb])
        th, fp = meta.get(pid, (_theorem(pid), ""))
        ap = [r[3] for r in rows]
        mp = [r[6] for r in rows]
        area_paths.append(ap)
        macro_paths.append(mp)
        proofs.append({"id": pid, "th": th, "fp": fp, "pl": whole.get(pid), "st": rows,
                       "ap": runs(ap), "mp": runs(mp), "sa": switches(ap), "sm": switches(mp)})
    for p, ra, rm in zip(proofs, related(area_paths), related(macro_paths)):
        p["ra"], p["rm"] = ra, rm

    pat: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(proofs):
        if len(p["ap"]) > 1:
            pat[SWITCH_SEP.join(p["ap"])].append(i)
    patterns = [{"key": k, "n": len(v), "proofs": v[:200]}
                for k, v in sorted(pat.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:MAX_PATTERNS]]

    codes = sorted({c for p in proofs for c in p["ap"]} | {c for p in proofs if p["pl"] for c in p["pl"].split(" / ")}
                   | set(areas["area"].unique()))
    area_titles = {c: msc.MSC2020_TOP.get(c, "") for c in codes}

    n_steps = sum(len(p["st"]) for p in proofs)
    data = {
        "meta": {
            "title": title or "Area transitions in Lean proofs: results explorer",
            "n_proofs_total": len(ids), "n_proofs_shown": len(proofs), "n_steps_shown": n_steps,
            "area_source": area_note, "has_text": bool(text_of), "has_model": a is not None,
            "max_state_chars": max_state_chars, "n_truncated_states": int(truncated),
            "inputs": {k: v for k, v in inp["found"].items()},
            "time_unit": "proof steps",
        },
        "run": run_summary(inp["pilot_report"]),
        "areas": area_titles,
        "model": model_section(inp, areas),
        "proofs": proofs,
        "patterns": patterns,
    }
    return _clean(data)


def embed(data: dict) -> str:
    """JSON safe to place inside a <script> element: no '<' survives, so no '</script>' either."""
    s = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return s.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _html_text(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(data: dict) -> str:
    from ._explorer_page import PAGE

    return (PAGE.replace("__TITLE__", _html_text(data["meta"]["title"]))
                .replace("__DATA__", embed(data)))


def run(results: str | Path, out: str | Path, steps: str | Path | None = None, labels: str | Path | None = None,
        msm_dir: str | Path | None = None, max_proofs: int = 500, max_state_chars: int = 600, seed: int = 0,
        title: str | None = None, verbose: bool = True) -> Path:
    say = print if verbose else (lambda *a, **k: None)
    results = Path(results)
    if not results.exists():
        raise FileNotFoundError(f"{results} does not exist")
    inp = load_inputs(results, Path(steps) if steps else None, Path(labels) if labels else None,
                      Path(msm_dir) if msm_dir else None)
    if inp["steps"] is None and inp["assignments"] is None:
        raise FileNotFoundError(f"nothing to show in {results}: no msm/assignments.parquet and no steps table "
                                "(pass --steps, or run `markov-lean pilot` first)")
    data = build_data(inp, max_proofs=max_proofs, max_state_chars=max_state_chars, seed=seed, title=title)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    page = out / "index.html"
    page.write_text(render(data), encoding="utf-8")
    m = data["meta"]
    say(f"explore: {m['n_proofs_shown']} of {m['n_proofs_total']} proofs, {m['n_steps_shown']} steps, "
        f"areas from {m['area_source']}; model {'yes' if data['model'] else 'not yet'}")
    say(f"wrote {page} ({page.stat().st_size / 1e6:.2f} MB); open it in a browser, no server needed")
    return page


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="markov-lean explore",
                                description="Write a static HTML results explorer (one file, no server).")
    p.add_argument("--results", required=True, help="a run folder, e.g. results/pilot3 or results/full")
    p.add_argument("--out", default="site", help="folder for index.html (default: site)")
    p.add_argument("--steps", default=None, help="steps table with goal states (default: <results>/judge/sample_steps.parquet)")
    p.add_argument("--labels", default=None, help="labels table (default: <results>/judge/labels.parquet)")
    p.add_argument("--msm", default=None, help="model folder (default: <results>/msm)")
    p.add_argument("--max-proofs", type=int, default=500, help="proofs on the page; 0 keeps all (default: 500)")
    p.add_argument("--max-state-chars", type=int, default=600,
                   help="cut each goal state to this many characters; 0 keeps it whole (default: 600)")
    p.add_argument("--seed", type=int, default=0, help="which proofs are kept when there are more than --max-proofs")
    p.add_argument("--title", default=None)
    a = p.parse_args(argv)
    run(a.results, a.out, steps=a.steps, labels=a.labels, msm_dir=a.msm, max_proofs=a.max_proofs,
        max_state_chars=a.max_state_chars, seed=a.seed, title=a.title)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
