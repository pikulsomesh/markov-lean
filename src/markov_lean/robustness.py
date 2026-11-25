"""Robustness checks for gate 1 and the paper's area claims.

``markov-lean robustness --pilot results/pilot3`` reads what ``markov-lean
pilot`` saved (``msm/``, ``judge/labels.parquet`` and the steps the judges
saw) and answers the questions a reviewer will ask about the gate 1 result:

1. **Baselines on the same labelled steps.**  The pilot's macrostates against
   groupings with no dynamics: the Mathlib folder (one and two levels deep),
   the theorem namespace (one and two levels), k-means on the same goal-state
   features, k-means over the same microstates' centers (the same microstates
   grouped statically instead of by PCCA+), the judges' own whole-proof label
   broadcast to every step, and a random partition.  Every score comes with a
   95% bootstrap interval over proofs, and each baseline gets a paired
   bootstrap of the difference from the macrostates.  NMI rises with the
   number of clusters, so AMI is reported too.
2. **Permutation nulls.**  NMI of the macrostates when their sequences are
   shuffled across steps, and when whole proofs' sequences are swapped between
   proofs of the same length (which keeps within-proof correlation, the
   honest null for step labels that are correlated inside a proof).
3. **Judge ceiling.**  How well the two judges agree with each other (NMI, ARI,
   kappa), and the macrostates scored against each judge alone.
4. **Sweeps.**  Lag, microstate count, macrostate count and featurizer.  Each
   cell reports agreement with the labels and the model checks that failed at
   the pilot's setting (Chapman-Kolmogorov, implied-timescale convergence,
   spectral gap, PCCA+ crispness), so the paper can show where they pass.
5. **Memory.**  Held-out log-likelihood of zeroth-, first-, second- and
   third-order Markov chains over the judges' area sequence, the macrostates
   and the microstates.  If the second-order model predicts held-out proofs
   better than the first-order one, the process has memory at lag 1, which is
   what a failing Chapman-Kolmogorov test suggests.

Nothing here calls an API.  Sweep cells run in parallel with ``jobs`` in spawned
processes, so a script that calls ``run`` with ``jobs > 1`` needs the usual
``if __name__ == "__main__":`` guard (the ``markov-lean`` command has it).
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

KEY = ["proof_id", "step"]
PROOF_LEVEL_STEP = -1


# ============================================================================ config


@dataclass
class RobustnessConfig:
    lags: list[int] = field(default_factory=lambda: [1, 2, 3, 5])
    # PCCA+ time grows steeply with k (about 80 s per model at k = 20), so above 12 the grid is sparser
    macrostates: list[int] = field(default_factory=lambda: list(range(2, 13)) + [14, 16, 18, 20])
    microstates: list[int] = field(default_factory=lambda: [50, 100, 200, 400])
    # "name" or "name:n_components"; tfidf_svd variants differ in SVD dimension
    featurizers: list[str] = field(default_factory=lambda: ["tfidf_svd:32", "tfidf_svd:64", "tfidf_svd:128",
                                                            "tfidf_svd:256"])
    ck_mlags: int = 4
    ck_tol: float = 0.10
    its_tol: float = 0.15
    min_gap_ratio: float = 1.5
    min_membership: float = 0.6
    n_permutations: int = 1000
    n_bootstrap: int = 1000
    memory_orders: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    memory_folds: int = 5
    memory_betas: list[float] = field(default_factory=lambda: [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0,
                                                                1000.0])
    judges: list[str] | None = None
    seed: int = 0
    jobs: int = 1
    figures: bool = True
    sweep: bool = True

    @classmethod
    def quick(cls, **kw) -> "RobustnessConfig":
        """A small grid for a first look (minutes on the Mac Mini)."""
        base = dict(lags=[1, 2, 3], macrostates=[2, 4, 6, 8, 10, 12, 16, 20], microstates=[100, 200],
                    featurizers=["tfidf_svd:64", "tfidf_svd:128"], n_permutations=200, n_bootstrap=200)
        base.update(kw)
        return cls(**base)


def parse_featurizer(spec: str) -> tuple[str, dict]:
    """``tfidf_svd:128`` -> ("tfidf_svd", {"n_components": 128}); ``sentence_embedding`` -> (name, {})."""
    name, _, arg = spec.partition(":")
    if not arg:
        return name, {}
    if name == "tfidf_svd":
        return name, {"n_components": int(arg)}
    return name, {"model_name": arg}


def featurizer_spec(name: str, kwargs: dict) -> str:
    if name == "tfidf_svd" and "n_components" in kwargs:
        return f"tfidf_svd:{int(kwargs['n_components'])}"
    if "model_name" in kwargs:
        return f"{name}:{kwargs['model_name']}"
    return name


# ============================================================================ inputs


@dataclass
class PilotInputs:
    steps: pd.DataFrame          # sorted by proof then step, one row per assignment
    assignments: pd.DataFrame    # the pilot's saved assignments, same row order as steps
    report: dict                 # msm_report.json
    labels: pd.DataFrame         # raw labels table (all judges)
    judges: list[str]            # the two judges the consensus is built from (or one)
    pilot_dir: Path | None = None


def _read_labels(path: Path) -> pd.DataFrame:
    lab = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    lab = lab.astype({"proof_id": str})
    lab["step"] = lab["step"].astype("int64")
    lab["msc"] = lab["msc"].astype(str)
    return lab


def load_pilot(pilot_dir: str | Path, labels: str | Path | None = None, steps: str | Path | None = None,
               judges: list[str] | None = None) -> PilotInputs:
    """Read a pilot output folder.

    ``steps`` defaults to ``<pilot>/judge/sample_steps.parquet``, the table the
    judges labelled, which is every pilot proof when ``judge.sample_proofs`` is
    at least the number of proofs (as in the 8,423-proof pilot).  Steps the
    saved assignments do not cover are dropped.
    """
    pdir = Path(pilot_dir)
    msm_dir = pdir / "msm"
    a_path = msm_dir / "assignments.parquet"
    if not a_path.exists():
        raise FileNotFoundError(f"{a_path} not found; run `markov-lean pilot --out {pdir}` first")
    assignments = pd.read_parquet(a_path)
    assignments["proof_id"] = assignments["proof_id"].astype(str)
    report = json.loads((msm_dir / "msm_report.json").read_text())
    lab_path = Path(labels) if labels else pdir / "judge" / "labels.parquet"
    if not lab_path.exists():
        raise FileNotFoundError(f"{lab_path} not found; the judges' labels are needed")
    steps_path = Path(steps) if steps else pdir / "judge" / "sample_steps.parquet"
    if not steps_path.exists():
        raise FileNotFoundError(f"{steps_path} not found; pass --steps with the pilot's steps table")
    st = pd.read_parquet(steps_path)
    return make_inputs(st, assignments, report, _read_labels(lab_path), judges, pdir)


def make_inputs(steps: pd.DataFrame, assignments: pd.DataFrame, report: dict, labels: pd.DataFrame,
                judges: list[str] | None = None, pilot_dir: Path | None = None) -> PilotInputs:
    a = assignments.copy()
    a["proof_id"] = a["proof_id"].astype(str)
    a["step"] = a["step"].astype("int64")
    st = steps.copy()
    st["proof_id"] = st["proof_id"].astype(str)
    st["step"] = st["step"].astype("int64")
    st = st.drop_duplicates(KEY)
    # steps the model saw, in the assignments' order (proof, then step)
    a = a.sort_values(KEY, kind="stable").reset_index(drop=True)
    merged = a[KEY].merge(st, on=KEY, how="left", indicator=True)
    missing = int((merged["_merge"] != "both").sum())
    if missing == len(a):
        raise ValueError("the steps table shares no (proof_id, step) with the saved assignments")
    if missing:
        # keep whole proofs only, so sequences stay intact
        bad = set(merged.loc[merged["_merge"] != "both", "proof_id"])
        keep = ~a["proof_id"].isin(bad)
        a, merged = a[keep].reset_index(drop=True), merged[keep.to_numpy()].reset_index(drop=True)
    st = merged.drop(columns="_merge")
    lab = labels.copy()
    lab["proof_id"] = lab["proof_id"].astype(str)
    present = [j for j in dict.fromkeys(lab["judge"].astype(str)) if j != "consensus"]
    js = list(judges or present[:2])
    if not js:
        raise ValueError("the labels table has no judge")
    return PilotInputs(steps=st, assignments=a, report=report, labels=lab, judges=js, pilot_dir=pilot_dir)


def consensus(labels: pd.DataFrame, judges: list[str]) -> pd.DataFrame:
    """Rows both judges agree on (or one judge's rows), as in the pilot's gate."""
    if len(judges) >= 2:
        from .judge import consensus as _cons

        return _cons(labels, judges[0], judges[1])
    return labels[labels["judge"] == judges[0]]


def expand_labels(rows: pd.DataFrame, labels: pd.DataFrame, col: str = "msc") -> pd.Series:
    """One label per row of ``rows`` (proof_id, step), as the pilot scores them.

    A step's own label wins; a step with none takes its proof's whole-proof
    label (step -1).  Rows with neither are NaN.
    """
    lab = labels[KEY + [col]].copy()
    lab["proof_id"] = lab["proof_id"].astype(str)
    lab["step"] = lab["step"].astype("int64")
    per_step = lab[lab["step"] != PROOF_LEVEL_STEP].drop_duplicates(KEY)
    whole = lab[lab["step"] == PROOF_LEVEL_STEP].drop_duplicates("proof_id")
    m = rows[KEY].merge(per_step, on=KEY, how="left")
    if len(whole):
        fill = rows[["proof_id"]].merge(whole[["proof_id", col]], on="proof_id", how="left")[col]
        m[col] = m[col].where(m[col].notna(), fill.to_numpy())
    return m[col].reset_index(drop=True)


def proof_labels(rows: pd.DataFrame, labels: pd.DataFrame, col: str = "msc") -> pd.Series:
    """The whole-proof label (step -1) broadcast to each row; NaN where a proof has none."""
    whole = labels[labels["step"] == PROOF_LEVEL_STEP][["proof_id", col]].drop_duplicates("proof_id")
    whole = whole.assign(proof_id=whole["proof_id"].astype(str))
    return rows[["proof_id"]].merge(whole, on="proof_id", how="left")[col].reset_index(drop=True)


# ---------------------------------------------------------------------------- static groupings


def folder_path(file_path: Any, depth: int = 1) -> str | None:
    """Mathlib folder ``depth`` levels deep: depth 2 of ``Mathlib/NumberTheory/Padics/X.lean`` is
    ``NumberTheory/Padics``.  A file above that depth gives its folder path plus stem."""
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    parts = [p for p in file_path.replace("\\", "/").split("/") if p and p != "."]
    if "Mathlib" not in parts:
        return None
    parts = parts[parts.index("Mathlib") + 1:]
    if not parts:
        return None
    parts = parts[:depth]
    parts[-1] = parts[-1][:-5] if parts[-1].endswith(".lean") else parts[-1]
    return "/".join(parts)


def namespace(theorem: Any, depth: int = 1) -> str | None:
    """Leading ``depth`` namespace components of a theorem name; ``_root_`` when it has none."""
    if not isinstance(theorem, str) or not theorem.strip():
        return None
    parts = theorem.split(".")
    if len(parts) == 1:
        return "_root_"
    return ".".join(parts[:min(depth, len(parts) - 1)])


# ============================================================================ metrics


def codes(values) -> tuple[np.ndarray, int]:
    """Integer codes with -1 for missing values (NaN, None, "" or a -1 cluster)."""
    s = pd.Series(values).reset_index(drop=True)
    bad = s.isna() | (s.astype(str) == "") | (s.astype(str) == "None")
    if pd.api.types.is_numeric_dtype(s):
        bad |= s == -1
    c, uniq = pd.factorize(s.where(~bad).astype(object))
    c = np.where(bad.to_numpy(), -1, c).astype(np.int64)
    return c, int(len(uniq))


def table(pred: np.ndarray, n_pred: int, lab: np.ndarray, n_lab: int, weights=None) -> np.ndarray:
    """Contingency counts of (pred, lab) over rows where both are set."""
    ok = (pred >= 0) & (lab >= 0)
    w = None if weights is None else np.asarray(weights)[ok]
    t = np.bincount(pred[ok] * n_lab + lab[ok], weights=w, minlength=n_pred * n_lab)
    return t.reshape(n_pred, n_lab).astype(float)


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def nmi_table(t: np.ndarray) -> float:
    """NMI with arithmetic-mean normalization (sklearn's default) from a contingency table."""
    t = t[t.sum(1) > 0][:, t.sum(0) > 0]
    n = t.sum()
    if n == 0:
        return float("nan")
    if t.shape[0] == t.shape[1] == 1:
        return 1.0
    pa, pb = t.sum(1) / n, t.sum(0) / n
    nz = t > 0
    pij = t[nz] / n
    mi = float((pij * (np.log(pij) - np.log(np.outer(pa, pb)[nz]))).sum())
    denom = 0.5 * (_entropy(pa) + _entropy(pb))
    if denom <= 0:
        return 0.0
    return float(np.clip(mi / denom, 0.0, 1.0))


def ari_table(t: np.ndarray) -> float:
    """Adjusted Rand index from a contingency table (integer or bootstrap-weighted counts)."""
    n = t.sum()
    if n < 2:
        return float("nan")

    def c2(x):
        return x * (x - 1) / 2.0

    s_ij = c2(t).sum()
    s_a, s_b = c2(t.sum(1)).sum(), c2(t.sum(0)).sum()
    expected = s_a * s_b / c2(n)
    mx = 0.5 * (s_a + s_b)
    if mx == expected:
        return 1.0
    return float((s_ij - expected) / (mx - expected))


def ami(pred: np.ndarray, lab: np.ndarray) -> float:
    from sklearn.metrics import adjusted_mutual_info_score

    ok = (pred >= 0) & (lab >= 0)
    if ok.sum() == 0:
        return float("nan")
    return float(adjusted_mutual_info_score(lab[ok], pred[ok]))


def scores(pred: np.ndarray, n_pred: int, lab: np.ndarray, n_lab: int, with_ami: bool = True) -> dict:
    t = table(pred, n_pred, lab, n_lab)
    out = {"nmi": nmi_table(t), "ari": ari_table(t), "n": int(t.sum()),
           "n_groups": int((t.sum(1) > 0).sum())}
    if with_ami:
        out["ami"] = ami(pred, lab)
    return out


def _ci(x: np.ndarray, center: float | None = None) -> tuple[float, float]:
    """95% interval of bootstrap draws ``x``.

    With ``center`` (the estimate on the full data) the interval is the draws'
    spread around their median, moved onto the estimate.  Resampling proofs
    with replacement duplicates rows, which pushes plug-in NMI up for
    groupings with many small groups; centering keeps that bias out of the
    interval (the bias itself is reported next to it).
    """
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan")
    lo, hi = float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))
    if center is None or not np.isfinite(center):
        return lo, hi
    med = float(np.median(x))
    return center - (med - lo), center + (hi - med)


def bootstrap_proofs(preds: dict[str, tuple[np.ndarray, int]], lab: np.ndarray, n_lab: int, groups: np.ndarray,
                     n_boot: int, seed: int, reference: str | None = None) -> dict:
    """95% intervals over proofs for NMI and ARI of each predictor, and paired differences.

    Each resample draws proofs with replacement; a row's weight is how many
    times its proof was drawn, so every predictor sees the same resample.
    ``reference`` (e.g. "macrostate") gets a paired difference against every
    other predictor, with the share of (centered) resamples where the
    reference is not ahead, a one-sided bootstrap p-value.
    """
    g, n_groups = codes(groups)
    rng = np.random.default_rng(seed)
    names = list(preds)
    obs_nmi = np.array([nmi_table(table(*preds[nm], lab, n_lab)) for nm in names])
    obs_ari = np.array([ari_table(table(*preds[nm], lab, n_lab)) for nm in names])
    nmi = np.full((n_boot, len(names)), np.nan)
    ari = np.full((n_boot, len(names)), np.nan)
    for b in range(n_boot):
        w = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)[g]
        for j, nm in enumerate(names):
            p, k = preds[nm]
            t = table(p, k, lab, n_lab, w)
            nmi[b, j], ari[b, j] = nmi_table(t), ari_table(t)
    out: dict[str, Any] = {"n_bootstrap": n_boot, "n_proofs": n_groups, "interval": "median-centered percentile",
                           "ci": {}, "paired": {}}
    for j, nm in enumerate(names):
        out["ci"][nm] = {"nmi": _ci(nmi[:, j], obs_nmi[j]), "ari": _ci(ari[:, j], obs_ari[j]),
                         "nmi_bootstrap_bias": float(np.nanmedian(nmi[:, j]) - obs_nmi[j])}
    if reference in names:
        r = names.index(reference)
        for j, nm in enumerate(names):
            if j == r:
                continue
            res = {}
            for metric, arr, obs in (("nmi", nmi, obs_nmi), ("ari", ari, obs_ari)):
                d = arr[:, r] - arr[:, j]
                d_obs = float(obs[r] - obs[j])
                shifted = d - (np.nanmedian(d) - d_obs)
                res[f"{metric}_diff_mean"] = d_obs
                res[f"{metric}_diff_ci"] = _ci(d, d_obs)
                res[f"p_{metric}_not_ahead"] = float(np.nanmean(shifted <= 0))
            out["paired"][nm] = res
    return out


def proof_offsets(proof: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For rows sorted by proof: proof code, position within proof, and each proof's start row."""
    p, _ = codes(proof)
    start = np.r_[0, np.flatnonzero(np.diff(p)) + 1]
    lens = np.diff(np.r_[start, len(p)])
    pos = np.arange(len(p)) - np.repeat(start, lens)
    return p, pos, start


def permutation_null(pred: np.ndarray, n_pred: int, lab: np.ndarray, n_lab: int, proof: np.ndarray,
                     n_perm: int, seed: int) -> dict:
    """NMI under two nulls, against the observed value.

    * ``row``: the predicted labels shuffled over all rows (sizes kept).
    * ``proof``: whole proofs' predicted sequences swapped between proofs of
      the same length, so runs of one macrostate inside a proof survive.

    ``pred``, ``lab`` and ``proof`` are row-aligned and sorted by proof then step.
    """
    obs_t = table(pred, n_pred, lab, n_lab)
    obs = {"nmi": nmi_table(obs_t), "ari": ari_table(obs_t)}
    rng = np.random.default_rng(seed)
    pc, pos, start = proof_offsets(proof)
    lens = np.diff(np.r_[start, len(pc)])
    by_len: dict[int, np.ndarray] = {}
    for L in np.unique(lens):
        by_len[int(L)] = np.flatnonzero(lens == L)
    out: dict[str, Any] = {"observed": obs, "n_permutations": n_perm}
    for kind in ("row", "proof"):
        vals = np.empty((n_perm, 2))
        for i in range(n_perm):
            if kind == "row":
                pp = rng.permutation(pred)
            else:
                perm = np.arange(len(start))
                for idx in by_len.values():
                    perm[idx] = idx[rng.permutation(len(idx))]
                pp = pred[start[perm[pc]] + pos]
            t = table(pp, n_pred, lab, n_lab)
            vals[i] = nmi_table(t), ari_table(t)
        out[kind] = {
            "nmi_mean": float(vals[:, 0].mean()), "nmi_q95": float(np.percentile(vals[:, 0], 95)),
            "nmi_max": float(vals[:, 0].max()),
            "p_nmi": float((1 + (vals[:, 0] >= obs["nmi"]).sum()) / (n_perm + 1)),
            "ari_mean": float(vals[:, 1].mean()), "ari_q95": float(np.percentile(vals[:, 1], 95)),
            "p_ari": float((1 + (vals[:, 1] >= obs["ari"]).sum()) / (n_perm + 1)),
        }
    return out


# ============================================================================ features


def featurize(texts: list[str], spec: str, seed: int = 0) -> np.ndarray:
    from .featurize import make_featurizer

    name, kw = parse_featurizer(spec)
    if name == "tfidf_svd":
        kw.setdefault("seed", seed)
    return np.asarray(make_featurizer(name, **kw).fit_transform(texts), dtype=np.float32)


def pilot_featurizer_spec(report: dict) -> str:
    cfg = report.get("config", {})
    return featurizer_spec(cfg.get("featurizer", "tfidf_svd"), cfg.get("featurizer_kwargs", {}) or {})


def _kmeans(X: np.ndarray, k: int, seed: int, sample_weight=None, n_init: int = 3) -> np.ndarray:
    from sklearn.cluster import KMeans, MiniBatchKMeans

    k = min(k, len(X))
    if len(X) <= 5000:
        return KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(X, sample_weight=sample_weight).labels_
    return MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=n_init).fit(
        X, sample_weight=sample_weight).labels_


# ============================================================================ baselines


def scored_frame(inp: PilotInputs) -> pd.DataFrame:
    """Every pilot step with its consensus label and every static grouping."""
    rows = inp.assignments.copy()
    cons = consensus(inp.labels, inp.judges)
    rows["label"] = expand_labels(rows, cons)
    for j in inp.judges:
        rows[f"label_{j}"] = expand_labels(rows, inp.labels[inp.labels["judge"] == j])
    rows["proof_label"] = proof_labels(rows, cons)
    st = inp.steps
    fp = st["file_path"] if "file_path" in st else pd.Series([None] * len(st))
    th = st["theorem"] if "theorem" in st else pd.Series([None] * len(st))
    rows["folder"] = fp.map(lambda x: folder_path(x, 1)).to_numpy()
    rows["folder2"] = fp.map(lambda x: folder_path(x, 2)).to_numpy()
    rows["namespace"] = th.map(lambda x: namespace(x, 1)).to_numpy()
    rows["namespace2"] = th.map(lambda x: namespace(x, 2)).to_numpy()
    return rows


BASELINE_NOTES = {
    "macrostate": "PCCA+ macrostates of the pilot model (the gate 1 partition)",
    "microstate": "k-means microstates the model is built on",
    "folder": "top-level Mathlib folder (the gate 1 baseline)",
    "folder2": "Mathlib folder two levels deep",
    "namespace": "first component of the theorem name",
    "namespace2": "first two components of the theorem name",
    "kmeans_steps": "k-means on the same goal-state features, k = macrostates (no dynamics)",
    "kmeans_microstates": "k-means on the same microstate centers weighted by visits, k = macrostates (no dynamics)",
    "proof_label": "the judges' own whole-proof label on every step (best any one-label-per-proof grouping can do)",
    "random": "random partition with the macrostates' sizes (one draw; see the permutation null)",
}


def baselines(inp: PilotInputs, cfg: RobustnessConfig, X: np.ndarray | None = None,
              log: Callable[[str], None] = print) -> dict:
    """Every grouping against the consensus labels on the same steps, with intervals and nulls."""
    rows = scored_frame(inp)
    k = int(inp.report["n_macrostates"])
    lab, n_lab = codes(rows["label"])
    keep = lab >= 0
    if not keep.any():
        raise ValueError("no pilot step has a consensus label")
    preds: dict[str, tuple[np.ndarray, int]] = {}
    for col in ("macrostate", "microstate", "folder", "folder2", "namespace", "namespace2", "proof_label"):
        c, n = codes(rows[col])
        if (c[keep] >= 0).any():
            preds[col] = (c, n)
    if X is not None:
        log(f"baselines: k-means with k={k} on {X.shape[0]} goal states")
        preds["kmeans_steps"] = (_kmeans(X, k, cfg.seed).astype(np.int64), k)
    micro = rows["microstate"].to_numpy()
    centers = _load_centers(inp)
    if centers is not None and len(centers) >= k:
        visits = np.bincount(micro, minlength=len(centers)).astype(float)
        km = _kmeans(centers, k, cfg.seed, sample_weight=np.maximum(visits, 1e-9), n_init=10)
        preds["kmeans_microstates"] = (km[micro].astype(np.int64), k)
    rng = np.random.default_rng(cfg.seed)
    preds["random"] = (rng.permutation(preds["macrostate"][0]), preds["macrostate"][1])

    # every predictor on the same rows: labelled, and set for every predictor
    common = keep.copy()
    for c, _ in preds.values():
        common &= c >= 0
    lab_c = np.where(common, lab, -1)
    table_rows = []
    for nm, (c, n) in preds.items():
        s = scores(c, n, lab_c, n_lab)
        own = scores(c, n, lab, n_lab, with_ami=False)  # on the predictor's own labelled rows
        table_rows.append({"grouping": nm, **s, "nmi_own_rows": own["nmi"], "n_own_rows": own["n"],
                           "what": BASELINE_NOTES.get(nm, "")})
    log(f"baselines: bootstrap over proofs ({cfg.n_bootstrap} resamples)")
    boot = bootstrap_proofs(preds, lab_c, n_lab, rows["proof_id"].to_numpy(), cfg.n_bootstrap, cfg.seed,
                            reference="macrostate")
    tab = pd.DataFrame(table_rows)
    tab["nmi_lo"] = tab["grouping"].map(lambda g: boot["ci"][g]["nmi"][0])
    tab["nmi_hi"] = tab["grouping"].map(lambda g: boot["ci"][g]["nmi"][1])
    tab["ari_lo"] = tab["grouping"].map(lambda g: boot["ci"][g]["ari"][0])
    tab["ari_hi"] = tab["grouping"].map(lambda g: boot["ci"][g]["ari"][1])
    log(f"baselines: permutation nulls ({cfg.n_permutations} shuffles)")
    mp, mk = preds["macrostate"]
    null = permutation_null(mp, mk, lab_c, n_lab, rows["proof_id"].to_numpy(), cfg.n_permutations, cfg.seed)

    # per-judge: is the result an artifact of the consensus filter?
    per_judge = []
    for j in inp.judges:
        lj, nj = codes(rows[f"label_{j}"])
        for nm in ("macrostate", "folder", "kmeans_steps", "kmeans_microstates"):
            if nm in preds:
                c, n = preds[nm]
                per_judge.append({"judge": j, "grouping": nm, **scores(c, n, lj, nj, with_ami=False)})
    coverage = {
        "n_steps": int(len(rows)),
        "n_steps_labelled": int(keep.sum()),
        "n_steps_scored": int(common.sum()),
        "n_proofs_scored": int(rows.loc[common, "proof_id"].nunique()),
        "n_areas": int(len(np.unique(lab[common]))),
        "k": k,
    }
    return {"table": tab, "bootstrap": boot, "null": null, "per_judge": pd.DataFrame(per_judge),
            "coverage": coverage, "preds": preds, "lab": (lab_c, n_lab), "rows": rows}


def _load_centers(inp: PilotInputs) -> np.ndarray | None:
    if inp.pilot_dir is not None and (inp.pilot_dir / "msm" / "centers.npy").exists():
        return np.load(inp.pilot_dir / "msm" / "centers.npy")
    return getattr(inp, "_centers", None)


def judge_ceiling(inp: PilotInputs) -> dict:
    """Agreement between the two judges on the steps both labelled: the most any partition can expect."""
    from sklearn.metrics import cohen_kappa_score

    if len(inp.judges) < 2:
        return {"judges": inp.judges, "note": "one judge only; no ceiling"}
    a, b = inp.judges[:2]
    rows = inp.assignments[KEY]
    la = expand_labels(rows, inp.labels[inp.labels["judge"] == a])
    lb = expand_labels(rows, inp.labels[inp.labels["judge"] == b])
    ok = la.notna() & lb.notna()
    out: dict[str, Any] = {"judges": [a, b], "n_steps": int(ok.sum())}
    if ok.any():
        ca, na = codes(la[ok])
        cb, nb = codes(lb[ok])
        out.update(scores(ca, na, cb, nb))
        out["kappa"] = float(cohen_kappa_score(la[ok].astype(str), lb[ok].astype(str)))
        out["raw_agreement"] = float((la[ok].astype(str) == lb[ok].astype(str)).mean())
    pl = inp.labels[inp.labels["step"] == PROOF_LEVEL_STEP]
    pa = pl[pl["judge"] == a].drop_duplicates("proof_id").set_index("proof_id")["msc"]
    pb = pl[pl["judge"] == b].drop_duplicates("proof_id").set_index("proof_id")["msc"]
    both = pa.index.intersection(pb.index)
    if len(both):
        out["proof_level"] = {"n_proofs": int(len(both)),
                              "raw_agreement": float((pa[both] == pb[both]).mean()),
                              "kappa": float(cohen_kappa_score(pa[both], pb[both]))}
    return out


def label_dynamics(rows: pd.DataFrame, col: str = "label") -> dict:
    """How often the judges' area changes inside a proof: the effect the macrostates must capture."""
    r = rows[["proof_id", "step", col]].dropna().sort_values(KEY)
    nxt = r.groupby("proof_id")[col].shift(-1)
    pairs = r[col][nxt.notna()].astype(str).to_numpy(), nxt[nxt.notna()].astype(str).to_numpy()
    distinct = r.groupby("proof_id")[col].nunique()
    return {
        "n_transitions": int(len(pairs[0])),
        "change_rate": float(np.mean(pairs[0] != pairs[1])) if len(pairs[0]) else float("nan"),
        "share_proofs_with_change": float((distinct > 1).mean()) if len(distinct) else float("nan"),
        "mean_distinct_areas_per_proof": float(distinct.mean()) if len(distinct) else float("nan"),
    }


# ============================================================================ sweeps


def _limit_threads(n):
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(n) if n else None
    except Exception:  # pragma: no cover
        return None


def _cluster_cell(args: dict) -> dict:
    """Phase 1: k-means microstates for one featurizer x microstate count, saved for the lag tasks."""
    limiter = _limit_threads(args.get("threads"))
    from .msm import cluster_microstates

    t0 = time.perf_counter()
    X = np.asarray(np.load(args["features"], mmap_mode="r"))
    km = cluster_microstates(X, args["n_micro"], args["seed"], args.get("batch_size", 2048), args.get("n_init", 3))
    np.save(args["micro_path"], km.labels_.astype(np.int64))
    np.save(args["centers_path"], km.cluster_centers_)
    if limiter is not None:
        limiter.restore_original_limits()
    return {"spec": args["spec"], "n_micro": args["n_micro"], "seconds": round(time.perf_counter() - t0, 1)}


def _lag_cell(args: dict) -> list[dict]:
    """Phase 2: one featurizer x microstate count x lag, over every macrostate count.

    PCCA+ is the slow step (it grows steeply with k), so it runs once per k and
    the same memberships feed the Chapman-Kolmogorov test, exactly as
    deeptime's ``MarkovStateModel.ck_test`` would build them.
    """
    limiter = _limit_threads(args.get("threads"))
    from deeptime.markov._base import MembershipsObservable
    from deeptime.util.validation import ck_test

    from .msm import estimate_msm, map_microstates

    t0 = time.perf_counter()
    micro = np.load(args["micro_path"])
    centers = np.load(args["centers_path"])
    n_micro = centers.shape[0]
    proof, lab, n_lab, lag = args["proof"], args["lab"], args["n_lab"], int(args["lag"])
    dtrajs = [np.asarray(t, dtype=np.int64) for t in np.split(micro, np.flatnonzero(np.diff(proof)) + 1)]
    lens = np.array([len(d) for d in dtrajs])

    def model(L: int):
        try:
            return estimate_msm(dtrajs, L, n_micro, args.get("reversible", True))[0]
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            return None

    msm = model(lag)
    if msm is None:
        return []
    nxt = model(lag + 1)
    ck_models = [msm] + [m for m in (model(j * lag) for j in range(2, args["ck_mlags"] + 1)) if m is not None]
    connected = np.asarray(msm.count_model.state_symbols, dtype=np.int64)
    kept = float(np.isin(micro, connected).mean())
    micro_nmi = scores(micro, n_micro, lab, n_lab, with_ami=False)["nmi"]
    ts_all = msm.timescales(min(max(args["ks"]) + 1, msm.n_states - 1))
    ts_next = nxt.timescales(min(max(args["ks"]) + 1, nxt.n_states - 1)) if nxt is not None else None
    out = []
    for k in args["ks"]:
        if k < 2 or k > msm.n_states - 1:
            continue
        t_k = time.perf_counter()
        row = {"featurizer": args["spec"], "n_microstates": n_micro, "lag": lag, "k": k,
               "fraction_steps_connected": kept,
               "share_proofs_longer_than_ck_window": float((lens > lag * args["ck_mlags"]).mean()),
               "microstate_nmi": micro_nmi}
        try:
            pcca = msm.pcca(k)
            macro_c, member, _ = map_microstates(n_micro, connected, pcca.assignments, pcca.memberships,
                                                 centers, "nearest")
        except Exception as exc:  # PCCA+ can fail on degenerate or complex spectra
            row["error"] = f"pcca: {type(exc).__name__}"
            out.append(row)
            continue
        pred = macro_c[micro]
        sc = scores(pred, k, lab, n_lab)
        row.update({"nmi": sc["nmi"], "ari": sc["ari"], "ami": sc["ami"], "n_scored": sc["n"],
                    "n_groups_used": sc["n_groups"], "pcca_membership": float(np.nanmean(member[micro]))})
        n_slow = k - 1
        row["timescale_slowest_used"] = float(ts_all[n_slow - 1]) if len(ts_all) >= n_slow else float("nan")
        row["gap_ratio"] = float(ts_all[n_slow - 1] / ts_all[n_slow]) if len(ts_all) > n_slow else float("nan")
        if ts_next is not None and len(ts_next) >= n_slow and len(ts_all) >= n_slow:
            row["its_rel_change"] = float(np.max(np.abs(ts_next[:n_slow] - ts_all[:n_slow]) / ts_all[:n_slow]))
        else:
            row["its_rel_change"] = float("nan")
        row["ck_max_error"] = row["ck_max_error_2lags"] = float("nan")
        if len(ck_models) >= 2:
            try:
                obs = MembershipsObservable(msm, pcca.memberships, initial_distribution=msm.stationary_distribution)
                ck = ck_test(ck_models, obs, test_model=msm, include_lag0=False)
                diff = np.abs(np.real(np.asarray(ck.predictions)) - np.real(np.asarray(ck.estimates)))
                row["ck_max_error"] = float(diff.max())
                row["ck_max_error_2lags"] = float(diff[:2].max())
            except Exception as exc:
                row["error"] = f"ck: {type(exc).__name__}"
        row["ck_lags_tested"] = len(ck_models)
        row["seconds"] = round(time.perf_counter() - t_k, 2)
        out.append(row)
    for r in out:
        r["seconds_task"] = round(time.perf_counter() - t0, 1)
    if limiter is not None:
        limiter.restore_original_limits()
    return out


def _run_tasks(fn, tasks: list[dict], jobs: int, label: str, log: Callable[[str], None]) -> list:
    """Run ``fn`` over ``tasks``, in ``jobs`` spawned processes when jobs > 1, logging as each finishes."""
    t0 = time.perf_counter()
    results = []
    if jobs > 1 and len(tasks) > 1:
        import multiprocessing as mp
        from concurrent.futures import as_completed

        # spawn, not fork: the parent holds BLAS threads, and spawn behaves the same on macOS and Linux
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(fn, t) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                results.append(f.result())
                log(f"sweep: {label} {i}/{len(tasks)} ({time.perf_counter() - t0:.0f}s)")
    else:
        for i, t in enumerate(tasks, 1):
            results.append(fn(t))
            log(f"sweep: {label} {i}/{len(tasks)} ({time.perf_counter() - t0:.0f}s)")
    return results


def run_sweep(inp: PilotInputs, cfg: RobustnessConfig, work: Path, lab: np.ndarray, n_lab: int,
              log: Callable[[str], None] = print) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """The full grid.  Features are computed once per featurizer, microstates once per featurizer x count,
    and the PCCA+-heavy part runs one task per featurizer x count x lag."""
    work.mkdir(parents=True, exist_ok=True)
    texts = inp.steps["state_before"].fillna("").astype(str).tolist()
    proof, _ = codes(inp.assignments["proof_id"])
    arrays: dict[str, np.ndarray] = {}
    rc = inp.report.get("config", {})
    threads = 1 if cfg.jobs > 1 else None
    clusters = []
    for spec in cfg.featurizers:
        safe = spec.replace(":", "_").replace("/", "_")
        path = work / f"features_{safe}.npy"
        t0 = time.perf_counter()
        X = featurize(texts, spec, cfg.seed)
        np.save(path, X)
        arrays[spec] = X
        log(f"sweep: features {spec} {X.shape} in {time.perf_counter() - t0:.0f}s")
        for m in cfg.microstates:
            clusters.append({"features": str(path), "spec": spec, "n_micro": int(m),
                             "micro_path": str(work / f"micro_{safe}_{m}.npy"),
                             "centers_path": str(work / f"centers_{safe}_{m}.npy"),
                             "seed": int(rc.get("seed", cfg.seed)), "batch_size": int(rc.get("kmeans_batch_size", 2048)),
                             "n_init": int(rc.get("kmeans_n_init", 3)), "threads": threads})
    log(f"sweep: clustering {len(clusters)} featurizer x microstate cells, {cfg.jobs} at a time")
    _run_tasks(_cluster_cell, clusters, cfg.jobs, "microstates", log)
    # the slowest tasks (large k, many microstates) first, so the pool stays busy to the end
    tasks = [{"micro_path": c["micro_path"], "centers_path": c["centers_path"], "spec": c["spec"], "lag": int(L),
              "ks": list(cfg.macrostates), "ck_mlags": cfg.ck_mlags, "proof": proof, "lab": lab, "n_lab": n_lab,
              "reversible": bool(rc.get("reversible", True)), "threads": threads}
             for c in sorted(clusters, key=lambda c: -c["n_micro"]) for L in cfg.lags]
    log(f"sweep: {len(tasks)} model tasks (x {len(cfg.lags)} lags), each over k {cfg.macrostates[0]}.."
        f"{cfg.macrostates[-1]}")
    rows = [r for res in _run_tasks(_lag_cell, tasks, cfg.jobs, "models", log) for r in res]
    return add_checks(pd.DataFrame(rows), cfg), arrays


def add_checks(sweep: pd.DataFrame, cfg: RobustnessConfig) -> pd.DataFrame:
    """Pass/fail columns for the model checks, with the pilot's thresholds."""
    s = sweep.copy()
    if s.empty:
        return s
    s["ck_passed"] = s["ck_max_error"] <= cfg.ck_tol
    s["its_converged"] = s["its_rel_change"] < cfg.its_tol
    s["spectral_gap_passed"] = s["gap_ratio"] >= cfg.min_gap_ratio
    s["pcca_crisp"] = s["pcca_membership"] >= cfg.min_membership
    s["timescales_resolved"] = s["timescale_slowest_used"] > s["lag"]
    checks = ["ck_passed", "its_converged", "spectral_gap_passed", "pcca_crisp", "timescales_resolved"]
    s["n_checks_passed"] = s[checks].sum(axis=1)
    s["all_checks_passed"] = s[checks].all(axis=1)
    return s


def sweep_summary(sweep: pd.DataFrame, report: dict, folder_nmi: float) -> dict:
    """Headline numbers from the grid for the paper."""
    if sweep.empty or "nmi" not in sweep:
        return {}
    s = sweep.dropna(subset=["nmi"])
    k0, lag0, m0 = int(report["n_macrostates"]), int(report["lag"]), int(report["n_microstates"])
    spec0 = pilot_featurizer_spec(report)
    at_pilot = s[(s["k"] == k0) & (s["lag"] == lag0) & (s["n_microstates"] == m0) & (s["featurizer"] == spec0)]
    at_k = s[s["k"] == k0]
    out = {
        "cells": int(len(s)),
        "pilot_setting": {"featurizer": spec0, "n_microstates": m0, "lag": lag0, "k": k0},
        "pilot_setting_nmi": float(at_pilot["nmi"].iloc[0]) if len(at_pilot) else None,
        "at_pilot_k": {
            "n_settings": int(len(at_k)),
            "nmi_min": float(at_k["nmi"].min()) if len(at_k) else None,
            "nmi_median": float(at_k["nmi"].median()) if len(at_k) else None,
            "nmi_max": float(at_k["nmi"].max()) if len(at_k) else None,
            "share_above_folder": float((at_k["nmi"] > folder_nmi).mean()) if len(at_k) else None,
        },
        "share_all_settings_above_folder": float((s["nmi"] > folder_nmi).mean()),
        "best": s.sort_values("nmi", ascending=False).head(1)[
            ["featurizer", "n_microstates", "lag", "k", "nmi", "ari", "ami"]].to_dict("records"),
        "check_pass_rates": {c: float(s[c].mean()) for c in
                             ("ck_passed", "its_converged", "spectral_gap_passed", "pcca_crisp",
                              "timescales_resolved", "all_checks_passed") if c in s},
        "ck_pass_rate_by_lag": {int(k): float(v) for k, v in s.groupby("lag")["ck_passed"].mean().items()},
    }
    ok = s[s["all_checks_passed"]]
    out["all_checks_settings"] = {
        "n": int(len(ok)),
        "best": ok.sort_values("nmi", ascending=False).head(5)[
            ["featurizer", "n_microstates", "lag", "k", "nmi", "ari", "ck_max_error"]].to_dict("records"),
    }
    return out


def kmeans_curve(X: np.ndarray, ks: list[int], lab: np.ndarray, n_lab: int, seed: int) -> pd.DataFrame:
    """k-means on the features with no dynamics, at every k of the macrostate sweep."""
    rows = []
    for k in ks:
        p = _kmeans(X, k, seed).astype(np.int64)
        rows.append({"k": k, **scores(p, k, lab, n_lab)})
    return pd.DataFrame(rows)


# ============================================================================ memory (higher-order)


def sequences(proof: np.ndarray, values: np.ndarray, step: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Contiguous runs of set values inside each proof (a missing or skipped step ends a run).

    Returns the runs and the proof code of each run.  Rows must be sorted by proof then step.
    """
    runs, owner = [], []
    p, _ = codes(proof)
    brk = np.r_[True, (np.diff(p) != 0) | (np.diff(step) != 1)]
    bad = values < 0
    cur: list[int] = []
    cur_p = -1
    for i in range(len(values)):
        if brk[i] or bad[i]:
            if len(cur) > 1:
                runs.append(np.array(cur, dtype=np.int64))
                owner.append(cur_p)
            cur = []
        if not bad[i]:
            cur.append(int(values[i]))
            cur_p = int(p[i])
    if len(cur) > 1:
        runs.append(np.array(cur, dtype=np.int64))
        owner.append(cur_p)
    return runs, np.array(owner, dtype=np.int64)


def _positions(runs: list[np.ndarray], order: int, S: int) -> dict:
    """Every position of every run, with its symbol and its length-n contexts (-1 where too short)."""
    if not runs:
        z = np.zeros(0, dtype=np.int64)
        return {"x": z, "t": z, "run": z, "ctx": {n: z for n in range(1, order + 1)}}
    x = np.concatenate(runs).astype(np.int64)
    lens = np.array([len(r) for r in runs])
    run = np.repeat(np.arange(len(runs)), lens)
    t = np.arange(len(x)) - np.repeat(np.r_[0, np.cumsum(lens)[:-1]], lens)
    ctx: dict[int, np.ndarray] = {}
    prev = np.zeros(len(x), dtype=np.int64)
    for n in range(1, order + 1):
        shifted = np.full(len(x), -1, dtype=np.int64)
        shifted[n:] = x[:-n]
        c = prev + shifted * (S ** (n - 1))
        c[t < n] = -1
        ctx[n] = c
        prev = np.where(t >= n, c, 0)
    return {"x": x, "t": t, "run": run, "ctx": ctx}


def _lookup(keys: np.ndarray, counts: np.ndarray, q: np.ndarray) -> np.ndarray:
    if not len(keys):
        return np.zeros(len(q))
    i = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
    return np.where(keys[i] == q, counts[i], 0).astype(float)


class BackoffChain:
    """Order-n Markov chain with hierarchical Dirichlet smoothing.

    P_n(x | c) = (N(c, x) + beta_n * P_{n-1}(x | c')) / (N(c) + beta_n), where c'
    drops the oldest symbol of c, down to an add-``alpha`` unigram.  ``beta``
    is one value per level (a scalar is used at every level).  An unseen
    context falls back to the lower order exactly, and as beta_n grows the
    order-n model becomes the order-(n-1) one, so a higher order can only win
    on held-out data if the data support it.
    """

    def __init__(self, order: int, n_states: int, beta: float | list[float] = 1.0, alpha: float = 0.5):
        if n_states ** max(order, 1) * n_states >= 2 ** 62:
            raise ValueError("too many states for this order")
        betas = list(beta) if isinstance(beta, (list, tuple, np.ndarray)) else [beta] * order
        if len(betas) < order:
            raise ValueError("need one beta per level")
        self.order, self.S, self.betas, self.alpha = order, n_states, [float(b) for b in betas], alpha

    def fit(self, runs: list[np.ndarray] | dict) -> "BackoffChain":
        P = runs if isinstance(runs, dict) else _positions(runs, self.order, self.S)
        x = P["x"]
        self.uni = np.bincount(x, minlength=self.S).astype(float) + self.alpha
        self.uni /= self.uni.sum()
        self.tables = {}
        for n in range(1, self.order + 1):
            c = P["ctx"][n]
            ok = c >= 0
            ck, cc = np.unique(c[ok], return_counts=True)
            pk, pc = np.unique(c[ok] * self.S + x[ok], return_counts=True)
            self.tables[n] = (ck, cc, pk, pc)
        return self

    def probs(self, P: dict) -> np.ndarray:
        x = P["x"]
        p = self.uni[x]
        alive = np.ones(len(x), dtype=bool)
        for n in range(1, self.order + 1):
            c = P["ctx"][n]
            ck, cc, pk, pc = self.tables[n]
            nc = _lookup(ck, cc, c)
            nx = _lookup(pk, pc, c * self.S + x)
            alive &= (c >= 0) & (nc > 0)
            b = self.betas[n - 1]
            p = np.where(alive, (nx + b * p) / (nc + b), p)
        return p

    def loglik(self, runs: list[np.ndarray] | dict, start: int, n_runs: int | None = None
               ) -> tuple[float, int, np.ndarray]:
        """Sum of log P(x_t | past) for t >= start in each run; also the per-run sums."""
        P = runs if isinstance(runs, dict) else _positions(runs, self.order, self.S)
        n_runs = n_runs if n_runs is not None else (len(runs) if not isinstance(runs, dict)
                                                    else int(P["run"].max()) + 1 if len(P["run"]) else 0)
        sel = P["t"] >= start
        lp = np.log(self.probs(P))[sel]
        per_run = np.bincount(P["run"][sel], weights=lp, minlength=n_runs)
        return float(lp.sum()), int(sel.sum()), per_run


def _pick_beta(runs: list[np.ndarray], lower: list[float], S: int, betas: list[float], start: int,
               seed: int) -> float:
    """Smoothing for the top level of an order-(len(lower)+1) chain, by 2-fold CV on training runs only."""
    order = len(lower) + 1
    if len(runs) < 4:
        return betas[len(betas) // 2]
    rng = np.random.default_rng(seed)
    half = rng.permutation(len(runs)) % 2
    folds = []
    for f in (0, 1):
        tr = [r for r, h in zip(runs, half) if h != f]
        te = [r for r, h in zip(runs, half) if h == f]
        folds.append((_positions(tr, order, S), _positions(te, order, S)))
    best, best_ll = betas[0], -np.inf
    for b in betas:
        ll = sum(BackoffChain(order, S, lower + [b]).fit(tr).loglik(te, start)[0] for tr, te in folds)
        if ll > best_ll:
            best, best_ll = b, ll
    return best


def memory_test(runs: list[np.ndarray], owner: np.ndarray, n_states: int, orders: list[int], folds: int,
                betas: list[float], n_boot: int, seed: int, name: str = "") -> dict:
    """Held-out log-likelihood per transition of each order, with K-fold CV over proofs.

    All orders are scored on the same targets: positions >= max(orders) in
    each run, so a higher order never wins by scoring easier steps.  The
    smoothing strength of each order is picked by a 2-fold split of the
    training proofs only.  The difference between orders gets a bootstrap
    interval over test proofs.
    """
    start = max(orders)
    runs_ok = [r for r in runs if len(r) > start]
    own = owner[[i for i, r in enumerate(runs) if len(r) > start]]
    if len(runs_ok) < max(folds, 4):
        return {"sequence": name, "error": f"too few runs longer than {start} steps ({len(runs_ok)})"}
    rng = np.random.default_rng(seed)
    proofs = np.unique(own)
    fold_of_proof = dict(zip(proofs, rng.permutation(len(proofs)) % folds))
    fold = np.array([fold_of_proof[p] for p in own])
    per_run = {o: np.zeros(len(runs_ok)) for o in orders}
    n_targets = 0
    betas_used: dict[int, list[list[float]]] = {o: [] for o in orders}
    for f in range(folds):
        tr = [r for r, ff in zip(runs_ok, fold) if ff != f]
        te_idx = np.flatnonzero(fold == f)
        te = [runs_ok[i] for i in te_idx]
        chain: list[float] = []  # smoothing of each level, grown one order at a time
        for o in range(max(orders) + 1):
            if o > 0:
                chain = chain + [_pick_beta(tr, chain, n_states, betas, start, seed + f)]
            if o not in per_run:
                continue
            betas_used[o].append(list(chain))
            _, n, pr = BackoffChain(o, n_states, chain).fit(tr).loglik(te, start)
            per_run[o][te_idx] = pr
            if o == orders[0]:
                n_targets += n
    targets = np.array([len(r) - start for r in runs_ok], dtype=float)
    out: dict[str, Any] = {"sequence": name, "n_states": int(n_states), "n_runs": int(len(runs_ok)),
                           "n_targets": int(n_targets), "score_from_position": start, "folds": folds,
                           "orders": {}}
    for o in orders:
        out["orders"][str(o)] = {"loglik_per_step": float(per_run[o].sum() / targets.sum()),
                                 "betas": betas_used[o]}
    # paired bootstrap over runs for successive orders
    g_own, n_g = codes(own)
    for a, b in zip(orders[:-1], orders[1:]):
        d = per_run[b] - per_run[a]
        boots = np.empty(n_boot)
        for i in range(n_boot):
            w = np.bincount(rng.integers(n_g, size=n_g), minlength=n_g)[g_own]
            boots[i] = (w * d).sum() / max((w * targets).sum(), 1)
        out[f"gain_{b}_vs_{a}"] = {"mean": float(d.sum() / targets.sum()), "ci": _ci(boots),
                                   "p_not_positive": float((boots <= 0).mean())}
    return out


def memory_tests(rows: pd.DataFrame, cfg: RobustnessConfig, log: Callable[[str], None] = print) -> list[dict]:
    """Memory test on the judges' consensus areas, the macrostates and the microstates."""
    r = rows.sort_values(KEY, kind="stable").reset_index(drop=True)
    proof = r["proof_id"].to_numpy()
    step = r["step"].to_numpy()
    out = []
    for name, col in (("judge areas (consensus)", "label"), ("macrostates", "macrostate"),
                      ("microstates", "microstate")):
        if col not in r:
            continue
        c, n = codes(r[col])
        runs, owner = sequences(proof, c, step)
        log(f"memory: {name}: {len(runs)} runs over {n} states")
        t0 = time.perf_counter()
        res = memory_test(runs, owner, n, list(cfg.memory_orders), cfg.memory_folds, list(cfg.memory_betas),
                          min(cfg.n_bootstrap, 1000), cfg.seed, name)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        out.append(res)
    return out


def memory_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        if "orders" not in r:
            continue
        for o, v in r["orders"].items():
            g = r.get(f"gain_{o}_vs_{int(o) - 1}", {})
            rows.append({"sequence": r["sequence"], "order": int(o), "loglik_per_step": v["loglik_per_step"],
                         "gain_vs_previous": g.get("mean"), "gain_lo": (g.get("ci") or [None, None])[0],
                         "gain_hi": (g.get("ci") or [None, None])[1], "n_targets": r["n_targets"],
                         "n_states": r["n_states"]})
    return pd.DataFrame(rows)


# ============================================================================ entry point


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    raise TypeError(type(o))


def _fmt(x, nd=3) -> str:
    try:
        return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def write_markdown(summary: dict, path: Path) -> None:
    b = summary["baselines"]
    tab = pd.DataFrame(b["table"])
    lines = ["# Gate 1 robustness", ""]
    cov = b["coverage"]
    lines += [f"{cov['n_steps_scored']} labelled steps in {cov['n_proofs_scored']} proofs, "
              f"{cov['n_areas']} MSC areas, k = {cov['k']} macrostates.", ""]
    v = summary["verdict"]
    lines += ["## Verdict", "", v["text"], ""]
    lines += ["## Groupings against the consensus labels (same steps)", "",
              "| grouping | groups | NMI [95% CI] | AMI | ARI [95% CI] |", "|---|---|---|---|---|"]
    for _, r in tab.sort_values("nmi", ascending=False).iterrows():
        lines.append(f"| {r['grouping']} | {int(r['n_groups'])} | {_fmt(r['nmi'])} [{_fmt(r['nmi_lo'])}, "
                     f"{_fmt(r['nmi_hi'])}] | {_fmt(r['ami'])} | {_fmt(r['ari'])} [{_fmt(r['ari_lo'])}, "
                     f"{_fmt(r['ari_hi'])}] |")
    lines += ["", "Paired bootstrap over proofs, macrostates minus each grouping:", "",
              "| grouping | NMI difference [95% CI] | P(macrostates not ahead) |", "|---|---|---|"]
    for nm, d in b["bootstrap"]["paired"].items():
        lines.append(f"| {nm} | {_fmt(d['nmi_diff_mean'])} [{_fmt(d['nmi_diff_ci'][0])}, "
                     f"{_fmt(d['nmi_diff_ci'][1])}] | {_fmt(d['p_nmi_not_ahead'])} |")
    n = b["null"]
    lines += ["", "## Permutation nulls for the macrostates", "",
              f"Observed NMI {_fmt(n['observed']['nmi'])}. Row shuffle: mean {_fmt(n['row']['nmi_mean'])}, "
              f"95th percentile {_fmt(n['row']['nmi_q95'])}, p = {_fmt(n['row']['p_nmi'], 4)}. "
              f"Proof swap (same length): mean {_fmt(n['proof']['nmi_mean'])}, 95th percentile "
              f"{_fmt(n['proof']['nmi_q95'])}, p = {_fmt(n['proof']['p_nmi'], 4)}.", ""]
    c = summary.get("judge_ceiling", {})
    if "nmi" in c:
        lines += ["## Judge ceiling", "",
                  f"{c['judges'][0]} vs {c['judges'][1]} on {c['n_steps']} steps: NMI {_fmt(c['nmi'])}, "
                  f"ARI {_fmt(c['ari'])}, kappa {_fmt(c['kappa'])}, raw agreement {_fmt(c['raw_agreement'])}.", ""]
    pj = pd.DataFrame(b.get("per_judge", []))
    if len(pj):
        lines += ["Against each judge alone (compare with the judge-vs-judge ceiling, not with the consensus scores):",
                  "", "| judge | grouping | NMI | ARI |", "|---|---|---|---|"]
        for _, r in pj.iterrows():
            lines.append(f"| {r['judge']} | {r['grouping']} | {_fmt(r['nmi'])} | {_fmt(r['ari'])} |")
        lines.append("")
    d = summary.get("label_dynamics", {})
    if d:
        lines += [f"Judges' areas change on {_fmt(d['change_rate'])} of within-proof steps; "
                  f"{_fmt(d['share_proofs_with_change'])} of proofs change area at least once.", ""]
    s = summary.get("sweep_summary") or {}
    if s:
        a = s["at_pilot_k"]
        lines += ["## Sweeps", "",
                  f"{s['cells']} settings. At k = {s['pilot_setting']['k']}: NMI {_fmt(a['nmi_min'])} to "
                  f"{_fmt(a['nmi_max'])} (median {_fmt(a['nmi_median'])}) over {a['n_settings']} settings; "
                  f"{_fmt(a['share_above_folder'])} of them beat the folder baseline. The sweep's cell at the "
                  f"pilot's own setting gives NMI {_fmt(s['pilot_setting_nmi'])} (saved pilot model: "
                  f"{_fmt(float(tab.set_index('grouping').loc['macrostate', 'nmi']))}).", "",
                  "Check pass rates over all settings: " + ", ".join(
                      f"{k} {_fmt(v, 2)}" for k, v in s["check_pass_rates"].items()) + ".", "",
                  "Chapman-Kolmogorov pass rate by lag: " + ", ".join(
                      f"lag {k}: {_fmt(v, 2)}" for k, v in s["ck_pass_rate_by_lag"].items()) + ".", "",
                  f"Settings where every check passes: {s['all_checks_settings']['n']}.", ""]
        best = s["all_checks_settings"]["best"]
        if best:
            lines += ["Best of those by NMI:", "", "| featurizer | microstates | lag | k | NMI | ARI | max CK error |",
                      "|---|---|---|---|---|---|---|"]
            lines += [f"| {r['featurizer']} | {r['n_microstates']} | {r['lag']} | {r['k']} | {_fmt(r['nmi'])} | "
                      f"{_fmt(r['ari'])} | {_fmt(r['ck_max_error'])} |" for r in best]
            lines.append("")
    mem = summary.get("memory", [])
    if mem:
        lines += ["## Memory (held-out log-likelihood per step, nats)", "",
                  "| sequence | order 0 | order 1 | order 2 | order 3 | gain 2 vs 1 [95% CI] |",
                  "|---|---|---|---|---|---|"]
        for m in mem:
            if "orders" not in m:
                lines.append(f"| {m['sequence']} | {m.get('error', '')} | | | | |")
                continue
            o = m["orders"]
            g = m.get("gain_2_vs_1", {})
            lines.append(f"| {m['sequence']} | " + " | ".join(_fmt(o.get(str(i), {}).get("loglik_per_step"))
                                                         for i in range(4))
                         + f" | {_fmt(g.get('mean'), 4)} [{_fmt((g.get('ci') or [None])[0], 4)}, "
                           f"{_fmt((g.get('ci') or [None, None])[1], 4)}] |")
        lines.append("")
    path.write_text("\n".join(lines))


def verdict(summary: dict, gate_nmi: float) -> dict:
    b = summary["baselines"]
    tab = pd.DataFrame(b["table"]).set_index("grouping")
    macro = tab.loc["macrostate"]
    paired = b["bootstrap"]["paired"]
    static = [g for g in ("folder", "folder2", "namespace", "namespace2", "kmeans_steps", "kmeans_microstates")
              if g in paired]
    beats = {g: paired[g]["nmi_diff_ci"][0] > 0 for g in static}
    folder_ok = bool(beats.get("folder", False))
    above = bool(macro["nmi_lo"] >= gate_nmi) if np.isfinite(macro["nmi_lo"]) else False
    null_ok = b["null"]["proof"]["p_nmi"] < 0.05
    parts = [f"Macrostate NMI {_fmt(macro['nmi'])} [{_fmt(macro['nmi_lo'])}, {_fmt(macro['nmi_hi'])}] "
             f"against a {gate_nmi} threshold."]
    parts.append("It beats the folder baseline with a bootstrap interval above zero." if folder_ok else
                 "Its lead over the folder baseline is not clear of zero (or it trails).")
    lost = [g for g, ok in beats.items() if not ok]
    if lost:
        parts.append("Not clearly ahead of: " + ", ".join(lost) + ".")
    else:
        parts.append("It is clearly ahead of every static grouping.")
    parts.append("It is above the proof-swap permutation null." if null_ok else
                 "It is not above the proof-swap permutation null.")
    return {"robust_pass": bool(folder_ok and above and null_ok), "beats": beats, "lower_ci_above_threshold": above,
            "above_null": bool(null_ok), "text": " ".join(parts)}


def run(pilot_dir: str | Path | None = None, out_dir: str | Path | None = None, cfg: RobustnessConfig | None = None,
        labels: str | Path | None = None, steps: str | Path | None = None, inputs: PilotInputs | None = None,
        gate_nmi: float = 0.15, verbose: bool = True) -> dict:
    """Run every check and write tables, figures, ``robustness.json`` and ``robustness.md`` to ``out_dir``."""
    cfg = cfg or RobustnessConfig()
    say = print if verbose else (lambda *a, **k: None)
    t_start = time.perf_counter()
    inp = inputs or load_pilot(pilot_dir, labels, steps, cfg.judges)
    out = Path(out_dir) if out_dir else Path(pilot_dir) / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    say(f"robustness: {inp.assignments['proof_id'].nunique()} proofs, {len(inp.assignments)} steps, "
        f"judges {inp.judges}")

    # features of the pilot's own featurizer, refit on the same texts (identical to the pilot's run)
    spec0 = pilot_featurizer_spec(inp.report)
    specs = list(dict.fromkeys(list(cfg.featurizers) + [spec0])) if cfg.sweep else [spec0]
    X0 = None
    arrays: dict[str, np.ndarray] = {}
    sweep = pd.DataFrame()
    rows_for_lab = scored_frame(inp)
    lab_all, n_lab_all = codes(rows_for_lab["label"])
    if cfg.sweep:
        sweep, arrays = run_sweep(inp, RobustnessConfig(**{**asdict(cfg), "featurizers": specs}), out / "cache",
                                  lab_all, n_lab_all, say)
        sweep.to_csv(out / "sweep.csv", index=False)
        X0 = arrays.get(spec0)
    if X0 is None:
        say(f"features: {spec0}")
        X0 = featurize(inp.steps["state_before"].fillna("").astype(str).tolist(), spec0, cfg.seed)

    base = baselines(inp, cfg, X0, say)
    base["table"].to_csv(out / "baselines.csv", index=False)
    base["per_judge"].to_csv(out / "baselines_per_judge.csv", index=False)
    lab_c, n_lab = base["lab"]
    say("k-means curve over k")
    kcurve = kmeans_curve(X0, list(cfg.macrostates), lab_c, n_lab, cfg.seed)
    kcurve.to_csv(out / "kmeans_by_k.csv", index=False)
    ceiling = judge_ceiling(inp)
    dyn = label_dynamics(base["rows"])
    mem = memory_tests(base["rows"], cfg, say)
    memory_table(mem).to_csv(out / "memory.csv", index=False)
    tab = base["table"].set_index("grouping")
    folder_nmi = float(tab.loc["folder", "nmi"]) if "folder" in tab.index else float("nan")
    summary: dict[str, Any] = {
        "config": asdict(cfg),
        "pilot": {k: inp.report.get(k) for k in ("n_microstates", "lag", "n_macrostates", "all_checks_passed")},
        "pilot_checks_failed": [k for k, c in inp.report.get("checks", {}).items() if not c.get("passed")],
        "baselines": {"table": base["table"], "bootstrap": base["bootstrap"], "null": base["null"],
                      "per_judge": base["per_judge"], "coverage": base["coverage"]},
        "judge_ceiling": ceiling,
        "label_dynamics": dyn,
        "kmeans_by_k": kcurve,
        "sweep_summary": sweep_summary(sweep, inp.report, folder_nmi) if len(sweep) else {},
        "memory": mem,
    }
    summary["verdict"] = verdict(summary, gate_nmi)
    summary["seconds"] = round(time.perf_counter() - t_start, 1)
    (out / "robustness.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
    write_markdown(summary, out / "robustness.md")
    if cfg.figures:
        try:
            from . import robustness_figures as rf

            paths = rf.save_figures(summary, sweep, out / "figures")
            say(f"figures: {len(paths)} written to {out / 'figures'}")
        except ImportError as exc:  # matplotlib/seaborn are the optional "figures" extra
            say(f"figures skipped ({exc}); pip install -e '.[figures]'")
    say(summary["verdict"]["text"])
    say(f"wrote {out / 'robustness.md'} in {summary['seconds']:.0f}s")
    return summary


def save_synthetic_pilot(out: Path, n_proofs: int = 150, seed: int = 0, noise: float = 0.1,
                         msm_overrides: dict | None = None) -> Path:
    """A tiny pilot folder from synthetic data with two fake judges, for tests and dry runs."""
    from .msm import MSMConfig
    from .msm import run as run_msm
    from .synthetic import generate

    steps, _ = generate(n_proofs=n_proofs, seed=seed)
    cfg = MSMConfig(**{"n_microstates": 30, "lag": 1, "n_macrostates": 6, "error_samples": 0,
                       "featurizer_kwargs": {"n_components": 16}, **(msm_overrides or {})})
    res = run_msm(steps, cfg)
    res.save(out / "msm")
    (out / "judge").mkdir(parents=True, exist_ok=True)
    steps.drop(columns=["true_area"]).to_parquet(out / "judge" / "sample_steps.parquet", index=False)
    rng = np.random.default_rng(seed)
    areas = sorted(steps["true_area"].unique())
    lab = []
    for j, p_flip in (("judge-a", noise / 2), ("judge-b", noise)):
        flip = rng.random(len(steps)) < p_flip
        msc = np.where(flip, rng.choice(areas, len(steps)), steps["true_area"].to_numpy())
        lab.append(pd.DataFrame({"proof_id": steps["proof_id"].astype(str), "step": steps["step"], "judge": j,
                                 "msc": msc, "confidence": 0.9}))
        whole = steps.groupby("proof_id")["true_area"].agg(lambda s: s.mode().iloc[0]).reset_index()
        lab.append(pd.DataFrame({"proof_id": whole["proof_id"].astype(str), "step": PROOF_LEVEL_STEP, "judge": j,
                                 "msc": whole["true_area"], "confidence": 0.9}))
    pd.concat(lab).to_parquet(out / "judge" / "labels.parquet", index=False)
    return out
