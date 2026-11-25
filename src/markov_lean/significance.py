"""Significance tests for gate 1: do MSM macrostates match the MSC labels better than the folder baseline?

``markov-lean significance --run results/pilot3`` reads a finished pilot run
(the second pass, with judge labels) and writes
``<run>/significance/significance.json`` and ``significance.md``.  It makes no
API call and refits nothing: the gate macrostates come from
``msm/assignments.parquet`` and the sweep macrostates are recomputed with
``pilot.macrostate_sweep`` from the saved transition matrix, centers and
connected set.  Point estimates come from ``pilot.score`` itself.

Every resampling unit is a **proof**, never a step: steps of one proof are
strongly dependent, so a step-level bootstrap would give intervals that are
far too narrow.

1. **Paired bootstrap of the NMI/ARI difference** (MSM minus folder baseline).
   Each resample draws proofs with replacement and recomputes both scores on
   the same resampled steps, so the difference is paired.  Reported: a 95%
   percentile interval and bootstrap p-values from the null-centered
   distribution (``delta* - delta_hat``): one-sided for "MSM beats the folder"
   (the gate's direction) and two-sided.  A per-proof swap permutation test is
   not used: the two partitions have different label sets, so a proof-level
   swap builds mixed partitions with extra clusters, which biases NMI/ARI.
2. **Within-proof shuffle null**: macrostate labels are permuted among the
   labelled steps of each proof, which keeps every proof's macrostate mix and
   breaks only the step-level alignment with the labels.  It asks whether the
   MSM tracks *where in a proof* the area changes, beyond which proof it is.
   When every proof has a single label (whole-proof judging), this null is
   degenerate (p = 1) and the report says so.  A global shuffle (all labelled
   steps at once) is reported next to it as the chance level.
3. **Judge agreement**: Cohen's kappa and raw agreement between the two judges
   (e.g. Haiku vs Sonnet), at the step and whole-proof levels, with 95%
   bootstrap intervals over proofs.
4. **Multiple comparisons across macrostate counts**: the tests in (1) and (2)
   are repeated for every k of the sweep (plus the gate's k) and corrected
   with Holm (family-wise error) and Benjamini-Hochberg (false discovery rate).

Everything is seeded.  Resampled scores are computed from weighted contingency
tables (proof-by-cell count matrices times multinomial proof weights), which
gives the same NMI/ARI as scikit-learn on the duplicated steps and runs in
seconds for the pilot's ~8k proofs on the Mac Mini.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from . import evaluate as ev
from . import pilot
from .msm import MSMResult

KEY = ["proof_id", "step"]
DEFAULT_SWEEP = [2, 4, 6, 8, 10, 12]


# --------------------------------------------------------------------------- metrics from contingency tables


def _xlogx(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    pos = x > 0
    out[pos] = x[pos] * np.log(x[pos])
    return out


def _comb2(x: np.ndarray) -> np.ndarray:
    return x * (x - 1) / 2.0


def nmi_ari_from_tables(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NMI (arithmetic normalization, as scikit-learn) and ARI for a stack of contingency tables.

    ``C`` has shape ``(B, n_pred, n_true)`` with (possibly weighted) counts; a
    2-D table is treated as ``B = 1``.  Empty rows or columns are ignored, so a
    resample that misses a cluster scores as if the cluster were absent.
    """
    C = np.asarray(C, dtype=float)
    if C.ndim == 2:
        C = C[None]
    n = C.sum(axis=(1, 2))
    a = C.sum(axis=2)
    b = C.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # entropies and mutual information in nats: H = log n - sum x log x / n
        ha = np.log(n) - _xlogx(a).sum(1) / n
        hb = np.log(n) - _xlogx(b).sum(1) / n
        hab = np.log(n) - _xlogx(C).sum((1, 2)) / n
        mi = np.maximum(ha + hb - hab, 0.0)
        norm = np.maximum((ha + hb) / 2.0, np.finfo(float).eps)
        nmi = mi / norm
    ka, kb = (a > 0).sum(1), (b > 0).sum(1)
    nmi = np.where((ka == 1) & (kb == 1), 1.0, nmi)
    nmi = np.where(np.isclose(mi, 0.0, atol=1e-15) & ~((ka == 1) & (kb == 1)), 0.0, nmi)

    s_ij = _comb2(C).sum((1, 2))
    s_a = _comb2(a).sum(1)
    s_b = _comb2(b).sum(1)
    total = _comb2(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = s_a * s_b / total
        maximum = (s_a + s_b) / 2.0
        ari = (s_ij - expected) / (maximum - expected)
    ari = np.where(np.isclose(maximum, expected), 1.0, ari)
    nmi = np.where(n > 0, nmi, np.nan)
    ari = np.where(n > 0, ari, np.nan)
    return nmi, ari


def kappa_from_tables(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cohen's kappa and raw agreement for a stack of square confusion tables ``(B, k, k)``."""
    C = np.asarray(C, dtype=float)
    if C.ndim == 2:
        C = C[None]
    n = C.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        po = np.trace(C, axis1=1, axis2=2) / n
        pe = (C.sum(2) * C.sum(1)).sum(1) / n ** 2
        kappa = np.where(np.isclose(pe, 1.0), 1.0, (po - pe) / (1.0 - pe))
    return np.where(n > 0, kappa, np.nan), np.where(n > 0, po, np.nan)


# --------------------------------------------------------------------------- per-proof count matrices


def _codes(values: pd.Series) -> tuple[np.ndarray, list]:
    codes, uniques = pd.factorize(pd.Series(values).astype(str), sort=True)
    return codes.astype(np.int64), list(uniques)


def proof_tables(proof_idx: np.ndarray, n_proofs: int, x: np.ndarray, y: np.ndarray,
                 nx: int, ny: int) -> sparse.csr_matrix:
    """Sparse ``(n_proofs, nx * ny)`` matrix: each proof's contingency table of (x, y), flattened."""
    cells = x * ny + y
    return sparse.csr_matrix((np.ones(len(cells)), (proof_idx, cells)), shape=(n_proofs, nx * ny))


def _resampled_tables(M: sparse.csr_matrix, W: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Contingency tables of every resample: proof weights ``W`` (B, P) times proof tables ``M``."""
    T = np.asarray((M.T @ W.T).T)
    return T.reshape(W.shape[0], *shape)


def _weights(rng: np.random.Generator, n_proofs: int, n_boot: int, chunk: int):
    """Multinomial proof weights in chunks, so memory stays at ``chunk * n_proofs`` floats."""
    done = 0
    p = np.full(n_proofs, 1.0 / n_proofs)
    while done < n_boot:
        b = min(chunk, n_boot - done)
        yield rng.multinomial(n_proofs, p, size=b).astype(float)
        done += b


# --------------------------------------------------------------------------- p-values and corrections


def bootstrap_pvalues(obs: float, boot: np.ndarray) -> dict:
    """p-values for H0: difference = 0 from the null-centered bootstrap distribution."""
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    if not np.isfinite(obs) or len(boot) == 0:
        return {"p_greater": float("nan"), "p_two_sided": float("nan")}
    centered = boot - obs
    B = len(boot)
    return {
        "p_greater": float((1 + np.sum(centered >= obs)) / (B + 1)),
        "p_two_sided": float((1 + np.sum(np.abs(centered) >= abs(obs))) / (B + 1)),
    }


def interval(boot: np.ndarray, level: float = 0.95) -> list[float]:
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return [float("nan"), float("nan")]
    lo, hi = np.quantile(boot, [(1 - level) / 2, (1 + level) / 2])
    return [float(lo), float(hi)]


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def benjamini_hochberg(p: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for rank, i in enumerate(order):
        running = min(running, p[i] * m / (m - rank))
        adj[i] = running
    return adj.tolist()


# --------------------------------------------------------------------------- loading a run


def load_msm(msm_dir: Path) -> MSMResult:
    """Rebuild enough of an ``MSMResult`` from ``MSMResult.save`` output to score and sweep k."""
    from deeptime.markov.msm import MarkovStateModel

    msm_dir = Path(msm_dir)
    assignments = pd.read_parquet(msm_dir / "assignments.parquet")
    assignments["proof_id"] = assignments["proof_id"].astype(str)
    report = json.loads((msm_dir / "msm_report.json").read_text())
    memberships = msm_dir / "memberships.npy"

    def csv(name: str) -> pd.DataFrame:
        p = msm_dir / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    return MSMResult(
        assignments=assignments,
        msm=MarkovStateModel(np.load(msm_dir / "micro_T.npy")),
        microstates=assignments["microstate"].to_numpy(),
        centers=np.load(msm_dir / "centers.npy"),
        connected=np.load(msm_dir / "connected.npy"),
        micro_to_macro=np.load(msm_dir / "micro_to_macro.npy"),
        coarse_T=np.load(msm_dir / "coarse_T.npy"),
        timescales=csv("implied_timescales.csv"),
        ck=csv("ck_test.csv"),
        microstate_cv=csv("microstate_cv.csv"),
        report=report,
        memberships=np.load(memberships) if memberships.exists() else None,
    )


def load_inputs(run: Path, steps_path: Path | None = None, labels_path: Path | None = None,
                config_path: Path | None = None, judges: list[str] | None = None,
                label_col: str | None = None) -> dict:
    """Steps, labels (consensus of two judges, or synthetic ``true_area``) and raw judge labels of a run."""
    run = Path(run)
    config = yaml.safe_load(Path(config_path).read_text()) if config_path else {}
    pilot_report = {}
    if (run / "pilot_report.json").exists():
        pilot_report = json.loads((run / "pilot_report.json").read_text())

    if steps_path is None and (run / "judge" / "sample_steps.parquet").exists():
        steps_path = run / "judge" / "sample_steps.parquet"
    if steps_path is not None:
        steps = pd.read_parquet(steps_path)
    elif config:
        data = config.get("data") or {}
        steps = pilot.sample_proofs(pilot.load_steps(data), data.get("max_proofs"),
                                    int(data.get("min_steps", 2)), int(data.get("seed", 0)))
    else:
        raise FileNotFoundError(f"no steps table: pass --steps, or --config to rebuild it, for {run}")
    steps = steps.assign(proof_id=steps["proof_id"].astype(str))

    if labels_path is None:
        jlab = (config.get("judge") or {}).get("labels")
        labels_path = (run / jlab) if jlab else run / "judge" / "labels.parquet"
    raw = None
    if Path(labels_path).exists():
        raw = pd.read_parquet(labels_path)
        raw["proof_id"] = raw["proof_id"].astype(str)
        judges = judges or (config.get("judge") or {}).get("judges") \
            or (pilot_report.get("judge_agreement") or {}).get("judges")
        if judges == ["consensus"]:
            judges = None
        labels, _ = pilot.consensus_labels(raw, judges)
        col = label_col or "msc"
        source = str(labels_path)
    elif "true_area" in steps.columns:
        labels, col, source = steps, label_col or "true_area", "true_area (synthetic)"
    else:
        raise FileNotFoundError(f"{labels_path} not found and the steps have no true_area column")

    ecfg = config.get("eval") or pilot_report.get("eval_config") or {}
    sweep = ecfg.get("macrostate_sweep")
    if sweep is None and isinstance(pilot_report.get("eval"), dict):
        sweep = [int(k) for k in pilot_report["eval"].get("macrostate_sweep", {})] or None
    return {"steps": steps, "labels": labels, "label_col": col, "raw_labels": raw, "labels_source": source,
            "sweep": sweep, "gate_nmi": float(ecfg.get("gate_nmi", 0.15)), "topk": list(ecfg.get("topk", [1, 3])),
            "pilot_report": pilot_report}


def scored_table(res: MSMResult, steps: pd.DataFrame, labels: pd.DataFrame, label_col: str,
                 sweep: list[int]) -> tuple[pd.DataFrame, list[int]]:
    """One row per labelled step with the label, folder, gate macrostate and each sweep macrostate.

    Labels are joined as ``pilot.score`` joins them (whole-proof labels count
    for every step of the proof).  Rows are kept where every partition is
    defined, so all tests are paired on the same steps.
    """
    a = res.assignments.copy()
    a["proof_id"] = a["proof_id"].astype(str)
    lab = labels[KEY + [label_col]].copy()
    lab["proof_id"] = lab["proof_id"].astype(str)
    lab["step"] = lab["step"].astype("int64")
    whole = lab[lab["step"] == -1]
    if len(whole):
        expanded = a[KEY].merge(whole[["proof_id", label_col]], on="proof_id")
        lab = pd.concat([lab[lab["step"] != -1], expanded]).drop_duplicates(KEY, keep="first")
    rows = a[KEY + ["macrostate"]].reset_index(drop=True)
    ks = []
    for k, pred in pilot.macrostate_sweep(res, sweep).items():
        rows[f"k{k}"] = pred
        ks.append(k)
    t = rows.merge(lab, on=KEY, how="inner")
    t = t.merge(steps[KEY + ["file_path"]], on=KEY, how="left")
    t["folder"] = ev.folder_labels(t)
    t = t.rename(columns={label_col: "label"})
    ok = t["label"].notna() & (t["label"].astype(str) != "") & t["folder"].notna()
    for c in ["macrostate"] + [f"k{k}" for k in ks]:
        ok &= t[c] >= 0
    t = t[ok].sort_values(KEY, kind="stable").reset_index(drop=True)
    return t, ks


# --------------------------------------------------------------------------- the tests


def _partition_tables(t: pd.DataFrame, cols: list[str], proof_idx: np.ndarray, n_proofs: int):
    y, y_names = _codes(t["label"])
    out = {}
    for c in cols:
        x, x_names = _codes(t[c])
        out[c] = (proof_tables(proof_idx, n_proofs, x, y, len(x_names), len(y_names)), (len(x_names), len(y_names)),
                  x, y)
    return out


def gate_tests(t: pd.DataFrame, ks: list[int], gate_k: int, n_boot: int, n_perm: int, seed: int,
               chunk: int = 250) -> dict:
    """Bootstrap intervals, paired differences against the folder, and shuffle nulls, for each partition."""
    proof_idx, proof_names = _codes(t["proof_id"])
    P = len(proof_names)
    parts = {"msm": "macrostate", "folder": "folder", **{f"k={k}": f"k{k}" for k in ks}}
    tables = _partition_tables(t, list(parts.values()), proof_idx, P)

    observed = {}
    for name, c in parts.items():
        M, shape, _, _ = tables[c]
        nmi, ari = nmi_ari_from_tables(np.asarray(M.sum(0)).reshape(shape))
        observed[name] = {"nmi": float(nmi[0]), "ari": float(ari[0])}

    rng = np.random.default_rng(seed)
    boot = {name: {"nmi": [], "ari": []} for name in parts}
    for W in _weights(rng, P, n_boot, chunk):
        for name, c in parts.items():
            M, shape, _, _ = tables[c]
            nmi, ari = nmi_ari_from_tables(_resampled_tables(M, W, shape))
            boot[name]["nmi"].append(nmi)
            boot[name]["ari"].append(ari)
    boot = {n: {m: np.concatenate(v) for m, v in d.items()} for n, d in boot.items()}

    partitions = {}
    for name in parts:
        partitions[name] = {m: {"value": observed[name][m], "ci95": interval(boot[name][m])} for m in ("nmi", "ari")}

    differences = {}
    for name in [p for p in parts if p != "folder"]:
        d = {}
        for m in ("nmi", "ari"):
            obs = observed[name][m] - observed["folder"][m]
            diff = boot[name][m] - boot["folder"][m]
            d[m] = {"difference": float(obs), "ci95": interval(diff), **bootstrap_pvalues(obs, diff)}
        differences[name] = d

    # shuffle nulls: one permutation per draw, applied to every partition
    y_all = {c: tables[c][3] for c in tables}
    within_constant = bool((t.groupby("proof_id")["label"].nunique() <= 1).all())
    null_names = [p for p in parts if p != "folder"]
    within = {n: [] for n in null_names}
    glob = {n: [] for n in null_names}
    rng = np.random.default_rng(seed + 1)
    for _ in range(n_perm):
        order_w = np.lexsort((rng.random(len(t)), proof_idx))   # rows are sorted by proof, so this permutes within proofs
        order_g = rng.permutation(len(t))
        for name in null_names:
            c = parts[name]
            _, (nx, ny), x, _ = tables[c]
            y = y_all[c]
            for order, store in ((order_w, within), (order_g, glob)):
                C = np.bincount(x[order] * ny + y, minlength=nx * ny).reshape(nx, ny)
                store[name].append(nmi_ari_from_tables(C)[0][0])
    nulls = {}
    for name in null_names:
        obs = observed[name]["nmi"]
        out = {}
        for label, store in (("within_proof", within), ("global", glob)):
            v = np.asarray(store[name])
            sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            out[label] = {
                "null_mean": float(v.mean()) if len(v) else float("nan"),
                "null_sd": sd,
                "z": float((obs - v.mean()) / sd) if len(v) > 1 and sd > 0 else float("nan"),
                "p": float((1 + np.sum(v >= obs - 1e-12)) / (len(v) + 1)) if len(v) else float("nan"),
            }
        nulls[name] = {"nmi": obs, **out}

    # multiple comparisons across macrostate counts
    family = [f"k={k}" for k in ks]
    if f"k={gate_k}" not in family:
        family.append("msm")
    corrections = {}
    if family:
        for test, getter in (
            ("nmi_beats_folder", lambda n: differences[n]["nmi"]["p_greater"]),
            ("ari_beats_folder", lambda n: differences[n]["ari"]["p_greater"]),
            ("nmi_within_proof_null", lambda n: nulls[n]["within_proof"]["p"]),
        ):
            p = [getter(n) for n in family]
            corrections[test] = {
                n: {"p": p[i], "holm": h, "bh": b}
                for i, (n, h, b) in enumerate(zip(family, holm(p), benjamini_hochberg(p)))
            }

    return {
        "n_proofs": int(P),
        "n_steps": int(len(t)),
        "n_labels": int(t["label"].nunique()),
        "labels_constant_within_proofs": within_constant,
        "partitions": partitions,
        "vs_folder": differences,
        "shuffle_nulls": nulls,
        "family": family,
        "corrected": corrections,
    }


def judge_tests(raw: pd.DataFrame | None, judges: list[str] | None, n_boot: int, seed: int,
                chunk: int = 250) -> dict | None:
    """Cohen's kappa between two judges with 95% bootstrap intervals over proofs, per level."""
    if raw is None:
        return None
    from .judge import paired

    present = [j for j in dict.fromkeys(raw["judge"].astype(str)) if j != "consensus"]
    judges = [j for j in (judges or []) if j != "consensus"] or present[:2]
    if len(judges) < 2:
        return {"judges": judges, "note": "fewer than two judges in the labels"}
    pairs = paired(raw, judges[0], judges[1])
    out: dict[str, Any] = {"judges": judges}
    for level, sel in (("step", pairs["step"] >= 0), ("proof", pairs["step"] == -1)):
        p = pairs[sel]
        if p.empty:
            out[level] = {"n": 0}
            continue
        cats = sorted(set(p["msc_a"].astype(str)) | set(p["msc_b"].astype(str)))
        code = {c: i for i, c in enumerate(cats)}
        xa = p["msc_a"].astype(str).map(code).to_numpy()
        xb = p["msc_b"].astype(str).map(code).to_numpy()
        pidx, pnames = _codes(p["proof_id"])
        K = len(cats)
        M = proof_tables(pidx, len(pnames), xa, xb, K, K)
        kap, po = kappa_from_tables(np.asarray(M.sum(0)).reshape(K, K))
        rng = np.random.default_rng(seed + 2 + (level == "proof"))
        bk, bp = [], []
        for W in _weights(rng, len(pnames), n_boot, chunk):
            k_, p_ = kappa_from_tables(_resampled_tables(M, W, (K, K)))
            bk.append(k_)
            bp.append(p_)
        out[level] = {
            "n": int(len(p)), "n_proofs": int(len(pnames)),
            "kappa": float(kap[0]), "kappa_ci95": interval(np.concatenate(bk)),
            "raw_agreement": float(po[0]), "raw_agreement_ci95": interval(np.concatenate(bp)),
        }
    return out


# --------------------------------------------------------------------------- report


def run(run_dir: str | Path, out_dir: str | Path | None = None, *, steps_path=None, labels_path=None,
        config_path=None, judges: list[str] | None = None, ks: list[int] | None = None, n_boot: int = 2000,
        n_perm: int = 1000, seed: int = 0, alpha: float = 0.05, verbose: bool = True) -> dict:
    """Run every test on a pilot run directory and write ``significance.json`` and ``significance.md``."""
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "significance"
    say = print if verbose else (lambda *a, **k: None)
    inp = load_inputs(run_dir, steps_path, labels_path, config_path, judges)
    res = load_msm(run_dir / "msm")
    gate_k = int(res.report["n_macrostates"])
    sweep = [int(k) for k in (ks or inp["sweep"] or DEFAULT_SWEEP)]

    # point estimates exactly as the pilot reports them
    point = pilot.score(res, inp["steps"], inp["labels"], inp["label_col"], inp["topk"], inp["gate_nmi"], sweep)
    point.pop("next_area", None)

    t, swept = scored_table(res, inp["steps"], inp["labels"], inp["label_col"], sweep)
    if t.empty:
        raise ValueError("no labelled steps with both a macrostate and a Mathlib folder")
    say(f"significance: {t['proof_id'].nunique()} proofs, {len(t)} labelled steps, k in {swept}, "
        f"{n_boot} bootstrap resamples, {n_perm} permutations")
    gate = gate_tests(t, swept, gate_k, n_boot, n_perm, seed)
    jt = judge_tests(inp["raw_labels"], judges or (inp["pilot_report"].get("judge_agreement") or {}).get("judges"),
                     n_boot, seed)

    d = gate["vs_folder"]["msm"]
    report = {
        "run": str(run_dir),
        "labels": inp["labels_source"],
        "label_col": inp["label_col"],
        "settings": {"n_boot": n_boot, "n_perm": n_perm, "seed": seed, "alpha": alpha, "resampling_unit": "proof",
                     "sweep": swept, "gate_k": gate_k},
        "pilot_score": point,
        "gate1": {
            "k": gate_k,
            "point_passed": point["gate1"]["passed"],
            "nmi_difference": d["nmi"]["difference"],
            "nmi_difference_ci95": d["nmi"]["ci95"],
            "p_nmi_beats_folder": d["nmi"]["p_greater"],
            "significant": bool(d["nmi"]["p_greater"] < alpha and d["nmi"]["ci95"][0] > 0),
        },
        **gate,
        "judge_agreement": jt,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "significance.json").write_text(json.dumps(report, indent=2, default=_jsonable))
    (out / "significance.md").write_text(markdown(report))
    g = report["gate1"]
    say(f"gate 1 at k={g['k']}: NMI difference {g['nmi_difference']:+.3f} "
        f"[{g['nmi_difference_ci95'][0]:+.3f}, {g['nmi_difference_ci95'][1]:+.3f}], "
        f"p = {g['p_nmi_beats_folder']:.4f} ({'significant' if g['significant'] else 'not significant'} at {alpha})")
    say(f"wrote {out / 'significance.json'} and {out / 'significance.md'}")
    return report


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _f(x: float, d: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _ci(ci: list[float], d: int = 3) -> str:
    return f"[{_f(ci[0], d)}, {_f(ci[1], d)}]"


def _p(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else (f"{x:.4f}" if x >= 1e-4 else f"{x:.1e}")


def markdown(rep: dict) -> str:
    s = rep["settings"]
    g = rep["gate1"]
    lines = [
        "# Gate 1 significance",
        "",
        f"Run `{rep['run']}`, labels `{rep['labels']}`. {rep['n_proofs']} proofs, {rep['n_steps']} labelled steps, "
        f"{rep['n_labels']} labels. Resampling unit: proof. {s['n_boot']} bootstrap resamples, "
        f"{s['n_perm']} permutations, seed {s['seed']}.",
        "",
        f"**Gate 1 (k = {g['k']})**: MSM minus folder NMI = {_f(g['nmi_difference'])} "
        f"{_ci(g['nmi_difference_ci95'])}, one-sided p = {_p(g['p_nmi_beats_folder'])}; "
        f"{'significant' if g['significant'] else 'not significant'} at alpha = {s['alpha']}.",
        "",
        "## Agreement with the labels, and the difference from the folder baseline",
        "",
        "| partition | NMI [95% CI] | ARI [95% CI] | ΔNMI vs folder [95% CI] | p (Δ>0) | ΔARI vs folder [95% CI] | p (Δ>0) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, part in rep["partitions"].items():
        label = f"MSM, gate k = {g['k']}" if name == "msm" else ("Mathlib folder" if name == "folder" else f"MSM, {name}")
        row = [label, f"{_f(part['nmi']['value'])} {_ci(part['nmi']['ci95'])}",
               f"{_f(part['ari']['value'])} {_ci(part['ari']['ci95'])}"]
        d = rep["vs_folder"].get(name)
        if d:
            row += [f"{_f(d['nmi']['difference'])} {_ci(d['nmi']['ci95'])}", _p(d["nmi"]["p_greater"]),
                    f"{_f(d['ari']['difference'])} {_ci(d['ari']['ci95'])}", _p(d["ari"]["p_greater"])]
        else:
            row += ["", "", "", ""]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Shuffle nulls (NMI)",
        "",
        "Within-proof: macrostates permuted among the steps of each proof. Global: permuted across all steps.",
    ]
    if rep["labels_constant_within_proofs"]:
        lines += ["", "Every proof has one label, so the within-proof null cannot move NMI (p = 1 by construction)."]
    lines += ["", "| partition | NMI | within-proof null mean ± sd | p | global null mean ± sd | p |",
              "|---|---|---|---|---|---|"]
    for name, nl in rep["shuffle_nulls"].items():
        w, gl = nl["within_proof"], nl["global"]
        lines.append(f"| {name} | {_f(nl['nmi'])} | {_f(w['null_mean'])} ± {_f(w['null_sd'])} | {_p(w['p'])} | "
                     f"{_f(gl['null_mean'])} ± {_f(gl['null_sd'])} | {_p(gl['p'])} |")
    if rep["corrected"]:
        lines += ["", f"## Corrected across macrostate counts ({', '.join(rep['family'])})", "",
                  "| test | " + " | ".join(rep["family"]) + " |", "|---|" + "---|" * len(rep["family"])]
        for test, by in rep["corrected"].items():
            for kind in ("p", "holm", "bh"):
                label = {"p": "raw p", "holm": "Holm", "bh": "BH"}[kind]
                lines.append(f"| {test} ({label}) | " + " | ".join(_p(by[n][kind]) for n in rep["family"]) + " |")
    jt = rep.get("judge_agreement")
    if jt and "step" in jt:
        lines += ["", f"## Judge agreement ({jt['judges'][0]} vs {jt['judges'][1]})", "",
                  "| level | pairs | proofs | Cohen's kappa [95% CI] | raw agreement [95% CI] |", "|---|---|---|---|---|"]
        for level in ("step", "proof"):
            v = jt[level]
            if not v.get("n"):
                lines.append(f"| {level} | 0 | | | |")
                continue
            lines.append(f"| {level} | {v['n']} | {v['n_proofs']} | {_f(v['kappa'])} {_ci(v['kappa_ci95'])} | "
                         f"{_f(v['raw_agreement'])} {_ci(v['raw_agreement_ci95'])} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="markov-lean significance",
                                description="Proof-level significance tests for gate 1 on a finished pilot run.")
    p.add_argument("--run", required=True, help="pilot output directory (holds msm/ and judge/labels.parquet)")
    p.add_argument("--out", default=None, help="output directory (default <run>/significance)")
    p.add_argument("--config", default=None, help="pilot config; gives the sweep, judges, and rebuilds the steps "
                                                  "when <run>/judge/sample_steps.parquet is missing")
    p.add_argument("--steps", default=None, help="steps parquet (default <run>/judge/sample_steps.parquet)")
    p.add_argument("--labels", default=None, help="judge labels parquet (default <run>/judge/labels.parquet)")
    p.add_argument("--judges", nargs=2, default=None, metavar=("A", "B"), help="the two judges to compare")
    p.add_argument("--ks", type=int, nargs="+", default=None, help="macrostate counts to test (default: the sweep)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    a = p.parse_args(argv)
    run(a.run, a.out, steps_path=a.steps, labels_path=a.labels, config_path=a.config, judges=a.judges, ks=a.ks,
        n_boot=a.n_boot, n_perm=a.n_perm, seed=a.seed, alpha=a.alpha)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
