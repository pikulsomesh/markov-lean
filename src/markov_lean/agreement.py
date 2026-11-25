"""Agreement between two or more MSC judges, across model families, and each judge's NMI with the macrostates.

With a third judge from another family (a local open model, see
``markov_lean.ollama_judge``) the paper can report whether the labels hold up
beyond one vendor's models. For every level (single proof steps, and whole
proofs) this computes, on the items **every** judge labelled:

* pairwise Cohen's kappa and raw agreement for each pair of judges,
* Fleiss' kappa across all judges, and the share of items where all agree,

and, when a pilot run is given, each judge's NMI and ARI with the gate
macrostates (labels joined to steps exactly as ``pilot.score`` and
``significance`` join them), both on the steps that judge labelled and on the
steps all judges labelled, next to the Mathlib-folder baseline on those steps.
Pairwise kappa on each pair's full overlap is reported too.

Intervals are 95% percentile bootstraps over **proofs**, like the rest of the
significance code: steps of one proof are dependent.

    markov-lean judge-agreement --run results/pilot3 \\
        --labels results/pilot3/judge/labels.parquet results/pilot3/judge/local/qwen2.5_7b-instruct/labels.parquet

writes ``<run>/agreement/judge_agreement.json`` and ``judge_agreement.md``.
No API call and no refit.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import significance as sig
from .judge import CONSENSUS_JUDGE, PROOF_LEVEL_STEP

KEY = ["proof_id", "step"]
LEVELS = (("step", lambda s: s >= 0), ("proof", lambda s: s == PROOF_LEVEL_STEP))


# --------------------------------------------------------------------------- metrics


def fleiss_from_sums(n_items: np.ndarray, p_sum: np.ndarray, cat: np.ndarray, n_raters: int) -> np.ndarray:
    """Fleiss' kappa from (weighted) per-resample sums.

    ``n_items`` (B,) items, ``p_sum`` (B,) the sum over items of
    ``P_i = (sum_j n_ij^2 - n) / (n (n - 1))``, ``cat`` (B, K) the category
    counts summed over items, with ``n`` raters per item.
    """
    n_items = np.asarray(n_items, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_bar = p_sum / n_items
        p_j = cat / (n_items[:, None] * n_raters)
        p_e = (p_j ** 2).sum(1)
        kappa = np.where(np.isclose(p_e, 1.0), 1.0, (p_bar - p_e) / (1.0 - p_e))
    return np.where(n_items > 0, kappa, np.nan)


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' kappa for an (items, raters) array of category labels."""
    ratings = np.asarray(ratings)
    if ratings.size == 0:
        return float("nan")
    cats = {c: i for i, c in enumerate(sorted(set(ratings.ravel().tolist())))}
    counts = np.zeros((ratings.shape[0], len(cats)))
    for r in range(ratings.shape[1]):
        np.add.at(counts, (np.arange(ratings.shape[0]), [cats[c] for c in ratings[:, r]]), 1)
    n = ratings.shape[1]
    p_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    return float(fleiss_from_sums(np.array([len(counts)]), np.array([p_i.sum()]), counts.sum(0)[None], n)[0])


def _boot_weights(n_proofs: int, n_boot: int, seed: int):
    if n_boot <= 0 or n_proofs == 0:
        return
    yield from sig._weights(np.random.default_rng(seed), n_proofs, n_boot, 250)


def _ci(values: list[np.ndarray]) -> list[float]:
    return sig.interval(np.concatenate(values)) if values else [float("nan"), float("nan")]


def wide_labels(labels: pd.DataFrame, judges: list[str]) -> pd.DataFrame:
    """One row per (proof_id, step) labelled by every judge, one column per judge."""
    lab = labels[labels["judge"].astype(str).isin(judges)].copy()
    lab["proof_id"] = lab["proof_id"].astype(str)
    lab["msc"] = lab["msc"].astype(str)
    lab = lab.drop_duplicates(KEY + ["judge"])
    w = lab.pivot(index=KEY, columns="judge", values="msc")
    w = w.reindex(columns=judges).dropna()
    return w.reset_index()


def _pair_stats(a: pd.Series, b: pd.Series, proofs: pd.Series, n_boot: int, seed: int) -> dict:
    cats = sorted(set(a) | set(b))
    code = {c: i for i, c in enumerate(cats)}
    xa, xb = a.map(code).to_numpy(), b.map(code).to_numpy()
    pidx, pnames = sig._codes(proofs)
    K = len(cats)
    M = sig.proof_tables(pidx, len(pnames), xa, xb, K, K)
    kap, po = sig.kappa_from_tables(np.asarray(M.sum(0)).reshape(K, K))
    bk, bp = [], []
    for W in _boot_weights(len(pnames), n_boot, seed):
        k_, p_ = sig.kappa_from_tables(sig._resampled_tables(M, W, (K, K)))
        bk.append(k_)
        bp.append(p_)
    return {"n": int(len(a)), "n_proofs": int(len(pnames)), "kappa": float(kap[0]), "kappa_ci95": _ci(bk),
            "raw_agreement": float(po[0]), "raw_agreement_ci95": _ci(bp)}


def _fleiss_stats(w: pd.DataFrame, judges: list[str], n_boot: int, seed: int) -> dict:
    R = w[judges].to_numpy()
    cats = {c: i for i, c in enumerate(sorted(set(R.ravel().tolist())))}
    K, n = len(cats), len(judges)
    counts = np.zeros((len(R), K))
    for r in range(n):
        np.add.at(counts, (np.arange(len(R)), [cats[c] for c in R[:, r]]), 1)
    p_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    all_agree = (counts.max(1) == n).astype(float)
    pidx, pnames = sig._codes(w["proof_id"])
    P = len(pnames)
    # per-proof sums, so a bootstrap resample is a weighted sum over proofs
    items = np.bincount(pidx, minlength=P).astype(float)
    psum = np.bincount(pidx, weights=p_i, minlength=P)
    agree = np.bincount(pidx, weights=all_agree, minlength=P)
    cat = np.zeros((P, K))
    np.add.at(cat, pidx, counts)
    kappa = fleiss_from_sums(items.sum()[None], psum.sum()[None], cat.sum(0)[None], n)[0]
    bk, ba = [], []
    for W in _boot_weights(P, n_boot, seed):
        ni = W @ items
        bk.append(fleiss_from_sums(ni, W @ psum, W @ cat, n))
        with np.errstate(divide="ignore", invalid="ignore"):
            ba.append((W @ agree) / ni)
    return {"n": int(len(R)), "n_proofs": int(P), "fleiss_kappa": float(kappa), "fleiss_kappa_ci95": _ci(bk),
            "all_agree": float(all_agree.mean()) if len(R) else float("nan"), "all_agree_ci95": _ci(ba)}


def judge_agreement(labels: pd.DataFrame, judges: list[str] | None = None, n_boot: int = 1000,
                    seed: int = 0) -> dict:
    """Pairwise Cohen's kappa and Fleiss' kappa at the step and whole-proof levels."""
    present = [j for j in dict.fromkeys(labels["judge"].astype(str)) if j != CONSENSUS_JUDGE]
    judges = judges or present
    missing = [j for j in judges if j not in present]
    if missing:
        raise ValueError(f"judges not in the labels: {missing}; present: {present}")
    if len(judges) < 2:
        raise ValueError("need at least two judges")
    out: dict[str, Any] = {"judges": judges, "settings": {"n_boot": n_boot, "seed": seed, "resampling_unit": "proof"}}
    for li, (level, sel) in enumerate(LEVELS):
        lab = labels[sel(labels["step"].astype("int64"))]
        w = wide_labels(lab, judges)
        res: dict[str, Any] = {"common": {"n": int(len(w)), "n_proofs": int(w["proof_id"].nunique())}}
        if len(w):
            if len(judges) > 2:
                res["common"].update(_fleiss_stats(w, judges, n_boot, seed + 10 * li))
            res["common"]["pairwise"] = [
                {"a": a, "b": b, **_pair_stats(w[a], w[b], w["proof_id"], n_boot, seed + 10 * li + 1 + k)}
                for k, (a, b) in enumerate(itertools.combinations(judges, 2))
            ]
        overlap = []
        for k, (a, b) in enumerate(itertools.combinations(judges, 2)):
            pw = wide_labels(lab, [a, b])
            overlap.append({"a": a, "b": b, **(_pair_stats(pw[a], pw[b], pw["proof_id"], n_boot,
                                                            seed + 10 * li + 5 + k) if len(pw) else {"n": 0})})
        res["pair_overlap"] = overlap
        out[level] = res
    return out


# --------------------------------------------------------------------------- NMI with the macrostates


def _nmi_stats(t: pd.DataFrame, cols: list[str], n_boot: int, seed: int) -> dict[str, dict]:
    if t.empty:
        return {c: {"n_steps": 0} for c in cols}
    pidx, pnames = sig._codes(t["proof_id"])
    tables = sig._partition_tables(t, cols, pidx, len(pnames))
    out = {}
    for c in cols:
        M, shape, _, _ = tables[c]
        nmi, ari = sig.nmi_ari_from_tables(np.asarray(M.sum(0)).reshape(shape))
        bn, ba = [], []
        for W in _boot_weights(len(pnames), n_boot, seed):
            n_, a_ = sig.nmi_ari_from_tables(sig._resampled_tables(M, W, shape))
            bn.append(n_)
            ba.append(a_)
        out[c] = {"n_steps": int(len(t)), "n_proofs": int(len(pnames)), "nmi": float(nmi[0]), "nmi_ci95": _ci(bn),
                  "ari": float(ari[0]), "ari_ci95": _ci(ba)}
    return out


def macrostate_nmi(labels: pd.DataFrame, judges: list[str], res, steps: pd.DataFrame, n_boot: int = 1000,
                   seed: int = 0) -> dict:
    """Each judge's NMI/ARI with the gate macrostates, on its own steps and on the steps all judges share."""
    tables = {}
    for j in judges:
        t, _ = sig.scored_table(res, steps, labels[labels["judge"].astype(str) == j], "msc", [])
        tables[j] = t
    own = {}
    for i, j in enumerate(judges):
        s = _nmi_stats(tables[j], ["macrostate", "folder"], n_boot, seed + 100 + i)
        own[j] = {**s["macrostate"], "folder_nmi": s["folder"].get("nmi", float("nan"))}
    keys = None
    for t in tables.values():
        k = set(zip(t["proof_id"], t["step"]))
        keys = k if keys is None else keys & k
    common = {}
    for i, j in enumerate(judges):
        t = tables[j]
        t = t[[k in keys for k in zip(t["proof_id"], t["step"])]] if keys else t.iloc[:0]
        s = _nmi_stats(t, ["macrostate"] + (["folder"] if i == 0 else []), n_boot, seed + 200)
        common[j] = s["macrostate"]
        if i == 0:
            common["Mathlib folder"] = s["folder"]
    return {"macrostates": int(res.report.get("n_macrostates", 0)), "own_steps": own, "common_steps": common}


# --------------------------------------------------------------------------- report


def report(labels: pd.DataFrame, judges: list[str] | None = None, run: str | Path | None = None,
           steps: pd.DataFrame | None = None, n_boot: int = 1000, seed: int = 0) -> dict:
    rep = judge_agreement(labels, judges, n_boot, seed)
    if run is not None:
        run = Path(run)
        if steps is None:
            steps = pd.read_parquet(run / "judge" / "sample_steps.parquet")
        steps = steps.assign(proof_id=steps["proof_id"].astype(str))
        res = sig.load_msm(run / "msm")
        rep["nmi"] = macrostate_nmi(labels, rep["judges"], res, steps, n_boot, seed)
        rep["run"] = str(run)
    return rep


def _f(x, d=3):
    return sig._f(x, d)


def _ci_s(ci):
    return sig._ci(ci)


def markdown(rep: dict) -> str:
    judges = rep["judges"]
    lines = ["# Judge agreement across model families", "",
             f"Judges: {', '.join(f'`{j}`' for j in judges)}. Intervals: 95% bootstrap over proofs "
             f"({rep['settings']['n_boot']} resamples, seed {rep['settings']['seed']}).", ""]
    for level in ("step", "proof"):
        c = rep[level]["common"]
        lines += [f"## {level.capitalize()} labels", "",
                  f"Items labelled by every judge: {c['n']} ({c['n_proofs']} proofs)."]
        if not c["n"]:
            lines.append("")
            continue
        if "fleiss_kappa" in c:
            lines += ["", f"Fleiss' kappa ({len(judges)} judges): **{_f(c['fleiss_kappa'])}** "
                          f"{_ci_s(c['fleiss_kappa_ci95'])}; all judges agree on {_f(c['all_agree'])} "
                          f"{_ci_s(c['all_agree_ci95'])} of items."]
        lines += ["", "| pair | items | Cohen's kappa [95% CI] | raw agreement [95% CI] | kappa on the pair's full overlap (items) |",
                  "|---|---|---|---|---|"]
        overlap = {(o["a"], o["b"]): o for o in rep[level]["pair_overlap"]}
        for p in c["pairwise"]:
            o = overlap[(p["a"], p["b"])]
            lines.append(f"| {p['a']} vs {p['b']} | {p['n']} | {_f(p['kappa'])} {_ci_s(p['kappa_ci95'])} | "
                         f"{_f(p['raw_agreement'])} {_ci_s(p['raw_agreement_ci95'])} | "
                         f"{_f(o.get('kappa'))} ({o['n']}) |")
        lines.append("")
    nmi = rep.get("nmi")
    if nmi:
        lines += [f"## NMI with the gate macrostates (k = {nmi['macrostates']})", "",
                  "Whole-proof labels count for the proof's unlabelled steps, as in the gate.", "",
                  "| labels | steps (all judges) | NMI [95% CI] | ARI [95% CI] | own steps | NMI on own steps [95% CI] | folder NMI on own steps |",
                  "|---|---|---|---|---|---|---|"]
        for j in judges:
            c, o = nmi["common_steps"][j], nmi["own_steps"][j]
            lines.append(f"| {j} | {c.get('n_steps', 0)} | {_f(c.get('nmi'))} {_ci_s(c.get('nmi_ci95', [None, None]))} | "
                         f"{_f(c.get('ari'))} {_ci_s(c.get('ari_ci95', [None, None]))} | {o.get('n_steps', 0)} | "
                         f"{_f(o.get('nmi'))} {_ci_s(o.get('nmi_ci95', [None, None]))} | {_f(o.get('folder_nmi'))} |")
        f = nmi["common_steps"].get("Mathlib folder", {})
        lines.append(f"| Mathlib folder (baseline) | {f.get('n_steps', 0)} | {_f(f.get('nmi'))} "
                     f"{_ci_s(f.get('nmi_ci95', [None, None]))} | {_f(f.get('ari'))} "
                     f"{_ci_s(f.get('ari_ci95', [None, None]))} | | | |")
        lines.append("")
    return "\n".join(lines)


def _read(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, dtype={"msc": str})
    df["proof_id"] = df["proof_id"].astype(str)
    return df


def load_labels(paths: list[str]) -> pd.DataFrame:
    """Concatenate labels tables; a judge that appears in two files keeps the first file's rows."""
    frames, seen = [], set()
    for p in paths:
        df = _read(p)
        df = df[~df["judge"].astype(str).isin(seen) & (df["judge"].astype(str) != CONSENSUS_JUDGE)]
        seen |= set(df["judge"].astype(str))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="markov-lean judge-agreement", description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", nargs="+", required=True, help="one or more labels tables (parquet or csv)")
    ap.add_argument("--judges", nargs="+", help="judges to compare, in order (default: all in the labels)")
    ap.add_argument("--run", help="pilot run folder (msm/ and judge/sample_steps.parquet) for NMI with macrostates")
    ap.add_argument("--steps", help="steps table (default <run>/judge/sample_steps.parquet)")
    ap.add_argument("--out", help="output folder (default <run>/agreement, or the current folder)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    labels = load_labels(a.labels)
    steps = pd.read_parquet(a.steps) if a.steps else None
    rep = report(labels, a.judges, a.run, steps, a.n_boot, a.seed)
    out = Path(a.out) if a.out else (Path(a.run) / "agreement" if a.run else Path("."))
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge_agreement.json").write_text(json.dumps(rep, indent=2, default=sig._jsonable))
    md = markdown(rep)
    (out / "judge_agreement.md").write_text(md)
    print(md)
    print(f"wrote {out / 'judge_agreement.json'} and {out / 'judge_agreement.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
