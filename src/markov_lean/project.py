"""Place proofs from any dataset on a fitted Markov state model and measure area moves.

Given a saved model directory (``MSMResult.save``: ``featurizer.pkl``,
``centers.npy``, ``micro_to_macro.npy``, ``memberships.npy``, ``coarse_T.npy``)
and a ``steps`` table (docs/schema.md), each goal state goes to its nearest
k-means microstate and from there to that microstate's PCCA+ macrostate.
Nothing is refitted, so every dataset is read in the training set's areas.

Per dataset this module computes, with time counted in proof steps:

* the **macrostate transition matrix**: moves from the macrostate at step t
  to the one at step t + lag, counted within each proof and row-normalized;
* **area-switch rates** per proof: the share of a proof's moves that change
  macrostate (``switch_rate``), and the count of switches;
* the **dwell-time distribution**: lengths of runs of consecutive steps in
  one macrostate, with flags for runs cut off by the start or end of the
  proof (censored runs; their true dwell is at least the length shown);
* **occupancy**: the share of steps in each macrostate;
* **bootstrap confidence intervals over whole proofs** for all of the above,
  and for the difference of each dataset from a reference dataset.

Statement-only datasets (one row per theorem, ``step = -1``, as the processed
miniF2F, ProofNet, PutnamBench and Lean Workbook files are) have no moves, so
they get occupancy only and ``has_transitions = False``.  Proofs found by
``markov_lean.search`` (its ``steps.parquet``) give those benchmarks real
trajectories.

The statistics are plain numpy and pandas, so they run anywhere; only
``place_states`` needs the fitted featurizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

KEY = ["proof_id", "step"]
PLACED_COLUMNS = ["proof_id", "step", "microstate", "macrostate", "membership"]


# --------------------------------------------------------------------------- the fitted model


@dataclass
class FittedModel:
    """What projection needs from a saved model directory."""

    featurizer: Any
    centers: np.ndarray
    micro_to_macro: np.ndarray
    memberships: np.ndarray | None
    coarse_T: np.ndarray | None
    lag: int
    n_macrostates: int
    report: dict = field(default_factory=dict)


def load_model(msm_dir: str | Path) -> FittedModel:
    """Read a directory written by ``MSMResult.save``.

    ``featurizer.pkl`` is a pickle, and loading a pickle runs code from the
    file, so load only a model directory you produced yourself.
    """
    import pickle

    d = Path(msm_dir)
    with open(d / "featurizer.pkl", "rb") as f:
        featurizer = pickle.load(f)  # noqa: S301 - a file this project wrote
    centers = np.load(d / "centers.npy")
    micro_to_macro = np.load(d / "micro_to_macro.npy")
    memberships = np.load(d / "memberships.npy") if (d / "memberships.npy").exists() else None
    coarse_T = np.load(d / "coarse_T.npy") if (d / "coarse_T.npy").exists() else None
    report = json.loads((d / "msm_report.json").read_text()) if (d / "msm_report.json").exists() else {}
    if len(micro_to_macro) != len(centers):
        raise ValueError(f"{d}: {len(centers)} centers but {len(micro_to_macro)} microstate labels")
    n_macro = int(report.get("n_macrostates") or (memberships.shape[1] if memberships is not None
                                                  else micro_to_macro.max() + 1))
    return FittedModel(featurizer, centers, micro_to_macro, memberships, coarse_T,
                       int(report.get("lag", 1)), n_macro, report)


def nearest_center(X: np.ndarray, centers: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Index of the nearest center (squared Euclidean) for each row, in chunks to bound memory."""
    X = np.asarray(X, dtype=np.float32)
    C = np.asarray(centers, dtype=np.float32)
    c2 = (C * C).sum(1)
    out = np.empty(len(X), dtype=np.int64)
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        d2 = c2[None, :] - 2.0 * (x @ C.T)
        out[i:i + chunk] = np.argmin(d2, axis=1)
    return out


def assign(X: np.ndarray, model: FittedModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(microstate, macrostate, membership) for feature rows ``X``."""
    micro = nearest_center(X, model.centers)
    macro = model.micro_to_macro[micro].astype(np.int64)
    if model.memberships is not None:
        mem = model.memberships[micro, np.clip(macro, 0, None)]
        mem = np.where(macro < 0, np.nan, mem)
    else:
        mem = np.full(len(micro), np.nan)
    return micro, macro, mem


def place_states(steps: pd.DataFrame, model: FittedModel,
                 transform: Callable[[list[str]], np.ndarray] | None = None,
                 batch_rows: int = 50000) -> pd.DataFrame:
    """Microstate, macrostate and membership for every row of a ``steps`` table.

    ``transform`` maps a list of goal states to features; the default is the
    model's own featurizer, one batch at a time (``markov_lean.full`` passes a
    parallel version).
    """
    df = steps.sort_values(KEY, kind="stable").reset_index(drop=True)
    texts = df["state_before"].fillna("").astype(str)
    fn = transform or model.featurizer.transform
    parts = [assign(np.asarray(fn(texts.iloc[i:i + batch_rows].tolist())), model)
             for i in range(0, len(df), batch_rows)]
    if parts:
        micro, macro, mem = (np.concatenate(p) for p in zip(*parts))
    else:
        micro = macro = np.zeros(0, np.int64)
        mem = np.zeros(0)
    return pd.DataFrame({"proof_id": df["proof_id"].astype(str).values, "step": df["step"].astype("int64").values,
                         "microstate": micro, "macrostate": macro, "membership": mem})[PLACED_COLUMNS]


# --------------------------------------------------------------------------- per-proof statistics


def _sorted(placed: pd.DataFrame) -> pd.DataFrame:
    df = placed[placed["macrostate"] >= 0]
    return df.sort_values(KEY, kind="stable").reset_index(drop=True)


@dataclass
class ProofTables:
    """Per-proof sufficient statistics, so the bootstrap is a weighted sum."""

    proof_ids: np.ndarray            # (n,)
    n_steps: np.ndarray              # (n,)
    n_moves: np.ndarray              # (n,) pairs (t, t + lag) inside the proof
    n_switches: np.ndarray           # (n,) of those pairs whose macrostate differs
    transitions: np.ndarray          # (n, m, m) move counts
    occupancy: np.ndarray            # (n, m) step counts per macrostate
    dwell_sum: np.ndarray            # (n,) total length of all runs (= n_steps)
    dwell_runs: np.ndarray           # (n,) number of runs
    dwell_sum_uncensored: np.ndarray  # (n,) runs that start and end inside the proof
    dwell_runs_uncensored: np.ndarray
    runs: pd.DataFrame               # one row per run

    @property
    def n_proofs(self) -> int:
        return len(self.proof_ids)

    @property
    def has_transitions(self) -> bool:
        return bool(self.n_moves.sum() > 0)


def proof_tables(placed: pd.DataFrame, n_macro: int, lag: int = 1) -> ProofTables:
    """Sufficient statistics per proof from a placed table (rows with macrostate -1 are dropped)."""
    df = _sorted(placed)
    codes, ids = pd.factorize(df["proof_id"], sort=False)
    n = len(ids)
    m = int(n_macro)
    macro = df["macrostate"].to_numpy(np.int64)
    n_steps = np.bincount(codes, minlength=n)
    occ = np.zeros((n, m))
    np.add.at(occ, (codes, macro), 1)

    trans = np.zeros((n, m, m))
    if len(df) > lag:
        same = codes[:-lag] == codes[lag:]
        a, b, c = macro[:-lag][same], macro[lag:][same], codes[:-lag][same]
        np.add.at(trans, (c, a, b), 1)
    n_moves = trans.sum((1, 2))
    n_switch = n_moves - np.einsum("pii->p", trans)

    # runs of a constant macrostate, consecutive rows of one proof
    if len(df):
        new_run = np.ones(len(df), bool)
        new_run[1:] = (codes[1:] != codes[:-1]) | (macro[1:] != macro[:-1])
        run_id = np.cumsum(new_run) - 1
        starts = np.flatnonzero(new_run)
        lengths = np.bincount(run_id)
        run_proof = codes[starts]
        run_macro = macro[starts]
        first_in_proof = np.ones(len(starts), bool)
        first_in_proof[1:] = run_proof[1:] != run_proof[:-1]
        last_in_proof = np.ones(len(starts), bool)
        last_in_proof[:-1] = run_proof[1:] != run_proof[:-1]
    else:
        lengths = run_proof = run_macro = np.zeros(0, np.int64)
        first_in_proof = last_in_proof = np.zeros(0, bool)
    uncensored = ~first_in_proof & ~last_in_proof
    runs = pd.DataFrame({"proof_id": np.asarray(ids)[run_proof] if len(run_proof) else np.zeros(0, object),
                         "macrostate": run_macro, "dwell": lengths,
                         "left_censored": first_in_proof, "right_censored": last_in_proof})
    return ProofTables(
        proof_ids=np.asarray(ids, dtype=object), n_steps=n_steps, n_moves=n_moves, n_switches=n_switch,
        transitions=trans, occupancy=occ,
        dwell_sum=np.bincount(run_proof, weights=lengths, minlength=n),
        dwell_runs=np.bincount(run_proof, minlength=n).astype(float),
        dwell_sum_uncensored=np.bincount(run_proof, weights=lengths * uncensored, minlength=n),
        dwell_runs_uncensored=np.bincount(run_proof, weights=uncensored.astype(float), minlength=n),
        runs=runs,
    )


def proof_stats_frame(t: ProofTables) -> pd.DataFrame:
    """One row per proof: length, moves, switches and switch rate, distinct macrostates visited."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(t.n_moves > 0, t.n_switches / t.n_moves, np.nan)
    return pd.DataFrame({
        "proof_id": t.proof_ids.astype(str), "n_steps": t.n_steps.astype(np.int64),
        "n_moves": t.n_moves.astype(np.int64), "n_switches": t.n_switches.astype(np.int64),
        "switch_rate": rate, "n_macrostates_visited": (t.occupancy > 0).sum(1).astype(np.int64),
        "dominant_macrostate": t.occupancy.argmax(1).astype(np.int64) if t.n_proofs else np.zeros(0, np.int64),
    })


# --------------------------------------------------------------------------- statistics and the bootstrap


def _row_normalize(C: np.ndarray) -> np.ndarray:
    s = C.sum(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, C / s, np.nan)


def _ratio(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / b, np.nan)


def statistics(t: ProofTables, w: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Dataset-level statistics with proof weights ``w`` (``None``: every proof once).

    Weights make the bootstrap a weighted sum over proofs: a resample that
    draws proof ``p`` twice has ``w[p] = 2``.  With a 2-D ``w`` of shape
    ``(B, n)`` every statistic gets a leading axis of length ``B``.
    """
    if w is None:
        w = np.ones(t.n_proofs)
    w = np.asarray(w, float)
    has = (t.n_moves > 0).astype(float)
    rate = _ratio(t.n_switches, t.n_moves)
    rate0 = np.nan_to_num(rate)
    C = np.tensordot(w, t.transitions, axes=(-1, 0))
    occ = np.tensordot(w, t.occupancy, axes=(-1, 0))
    return {
        "n_proofs": w.sum(-1),
        "pooled_switch_rate": _ratio(w @ t.n_switches, w @ t.n_moves),
        "mean_switch_rate": _ratio(w @ rate0, w @ has),
        "mean_switches_per_proof": _ratio(w @ t.n_switches, w @ has),
        # dwell over proofs with at least one move; a one-state proof has a trivial dwell of 1
        "mean_dwell": _ratio(w @ (t.dwell_sum * has), w @ (t.dwell_runs * has)),
        "mean_dwell_uncensored": _ratio(w @ t.dwell_sum_uncensored, w @ t.dwell_runs_uncensored),
        "share_proofs_with_switch": _ratio(w @ ((t.n_switches > 0) * has), w @ has),
        "occupancy": _row_normalize(occ),
        "transition_matrix": _row_normalize(C),
        "stay_probability": np.diagonal(_row_normalize(C), axis1=-2, axis2=-1),
    }


SCALARS = ["pooled_switch_rate", "mean_switch_rate", "mean_switches_per_proof", "mean_dwell",
           "mean_dwell_uncensored", "share_proofs_with_switch"]


def bootstrap_draws(t: ProofTables, n_boot: int, seed: int, chunk: int = 100) -> dict[str, np.ndarray]:
    """``n_boot`` resamples of whole proofs (seeded); each statistic gets a leading axis of length ``n_boot``."""
    rng = np.random.default_rng(seed)
    n = t.n_proofs
    out: dict[str, list] = {}
    if n == 0 or n_boot <= 0:
        return {}
    p = np.full(n, 1.0 / n)
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        w = rng.multinomial(n, p, size=b).astype(float)
        for k, v in statistics(t, w).items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def interval(draws: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval over the leading (bootstrap) axis; all-NaN entries give NaN."""
    import warnings

    a = (1 - level) / 2 * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(draws, a, axis=0), np.nanpercentile(draws, 100 - a, axis=0)


# --------------------------------------------------------------------------- one dataset, many datasets


@dataclass
class DatasetResult:
    name: str
    placed: pd.DataFrame
    tables: ProofTables
    point: dict[str, np.ndarray]
    draws: dict[str, np.ndarray]

    @property
    def has_transitions(self) -> bool:
        return self.tables.has_transitions


def analyze(name: str, placed: pd.DataFrame, n_macro: int, lag: int = 1, n_boot: int = 1000,
            seed: int = 0) -> DatasetResult:
    t = proof_tables(placed, n_macro, lag)
    return DatasetResult(name, placed, t, statistics(t), bootstrap_draws(t, n_boot, seed))


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()] if x.ndim else _jsonable(x.item())
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def summary_row(r: DatasetResult, level: float = 0.95) -> dict:
    """Point estimate and interval of every scalar statistic, flat for a parquet row."""
    row: dict[str, Any] = {"dataset": r.name, "n_proofs": int(r.tables.n_proofs),
                           "n_steps": int(r.tables.n_steps.sum()), "n_moves": int(r.tables.n_moves.sum()),
                           "has_transitions": r.has_transitions,
                           "median_proof_length": float(np.median(r.tables.n_steps)) if r.tables.n_proofs else np.nan}
    for k in SCALARS:
        est = float(r.point[k]) if np.ndim(r.point[k]) == 0 else np.nan
        lo, hi = interval(r.draws[k], level) if k in r.draws else (np.nan, np.nan)
        row[k], row[f"{k}_lo"], row[f"{k}_hi"] = est, float(lo), float(hi)
    return row


def transition_rows(r: DatasetResult, level: float = 0.95) -> pd.DataFrame:
    """Long table: dataset, from, to, count, probability and its interval."""
    C = r.tables.transitions.sum(0)
    P = r.point["transition_matrix"]
    lo, hi = interval(r.draws["transition_matrix"], level) if "transition_matrix" in r.draws else (P * np.nan,) * 2
    m = C.shape[0]
    i, j = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    return pd.DataFrame({"dataset": r.name, "from_macrostate": i.ravel(), "to_macrostate": j.ravel(),
                         "count": C.ravel().astype(np.int64), "probability": P.ravel(),
                         "lo": np.ravel(lo), "hi": np.ravel(hi)})


def occupancy_rows(r: DatasetResult, level: float = 0.95) -> pd.DataFrame:
    occ = r.point["occupancy"]
    lo, hi = interval(r.draws["occupancy"], level) if "occupancy" in r.draws else (occ * np.nan,) * 2
    return pd.DataFrame({"dataset": r.name, "macrostate": np.arange(len(occ)),
                         "n_steps": r.tables.occupancy.sum(0).astype(np.int64),
                         "share": occ, "lo": lo, "hi": hi})


def dwell_histogram(r: DatasetResult) -> pd.DataFrame:
    runs = r.tables.runs
    if runs.empty:
        return pd.DataFrame(columns=["dataset", "dwell", "n_runs", "n_uncensored", "n_left_censored",
                                     "n_right_censored"])
    unc = ~runs["left_censored"] & ~runs["right_censored"]
    g = runs.assign(unc=unc).groupby("dwell")
    return pd.DataFrame({"dataset": r.name, "dwell": g.size().index.astype(np.int64), "n_runs": g.size().values,
                         "n_uncensored": g["unc"].sum().values.astype(np.int64),
                         "n_left_censored": g["left_censored"].sum().values.astype(np.int64),
                         "n_right_censored": g["right_censored"].sum().values.astype(np.int64)})


def differences(results: list[DatasetResult], reference: str, level: float = 0.95) -> pd.DataFrame:
    """Each dataset minus the reference, per scalar statistic, from independent bootstraps."""
    ref = next((r for r in results if r.name == reference), None)
    cols = ["dataset", "reference", "statistic", "difference", "lo", "hi"]
    if ref is None or not ref.draws:
        return pd.DataFrame(columns=cols)
    rows = []
    for r in results:
        if r.name == reference or not r.draws:
            continue
        for k in SCALARS:
            a, b = r.draws[k], ref.draws[k]
            nb = min(len(a), len(b))
            if not np.isfinite(r.point[k]) or not np.isfinite(ref.point[k]):
                continue
            lo, hi = interval(a[:nb] - b[:nb], level)
            rows.append({"dataset": r.name, "reference": reference, "statistic": k,
                         "difference": float(r.point[k] - ref.point[k]), "lo": float(lo), "hi": float(hi)})
    return pd.DataFrame(rows, columns=cols)


def write_results(results: list[DatasetResult], out_dir: str | Path, reference: str | None = None,
                  level: float = 0.95, model: FittedModel | None = None) -> dict:
    """Write per-dataset tables and the cross-dataset summary; returns the summary dict.

    Layout under ``out_dir``::

        <dataset>/placed.parquet       proof_id, step, microstate, macrostate, membership
        <dataset>/proof_stats.parquet  one row per proof (switches, switch rate, ...)
        <dataset>/dwell_runs.parquet   one row per run of a constant macrostate
        summary.parquet                one row per dataset, every scalar with its interval
        transitions.parquet            dataset, from, to, count, probability, lo, hi
        occupancy.parquet              dataset, macrostate, share, lo, hi
        dwell_hist.parquet             dataset, dwell, run counts (all, uncensored, censored)
        differences.parquet            dataset minus reference, per statistic, with interval
        summary.json                   all of the above as nested JSON (matrices included)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for r in results:
        d = out / r.name
        d.mkdir(parents=True, exist_ok=True)
        r.placed.to_parquet(d / "placed.parquet", index=False)
        proof_stats_frame(r.tables).to_parquet(d / "proof_stats.parquet", index=False)
        r.tables.runs.assign(proof_id=lambda x: x["proof_id"].astype(str)).to_parquet(
            d / "dwell_runs.parquet", index=False)
    summary = pd.DataFrame([summary_row(r, level) for r in results])
    trans = pd.concat([transition_rows(r, level) for r in results], ignore_index=True) if results else pd.DataFrame()
    occ = pd.concat([occupancy_rows(r, level) for r in results], ignore_index=True) if results else pd.DataFrame()
    dwell = pd.concat([dwell_histogram(r) for r in results], ignore_index=True) if results else pd.DataFrame()
    diff = differences(results, reference, level) if reference else pd.DataFrame()
    summary.to_parquet(out / "summary.parquet", index=False)
    trans.to_parquet(out / "transitions.parquet", index=False)
    occ.to_parquet(out / "occupancy.parquet", index=False)
    dwell.to_parquet(out / "dwell_hist.parquet", index=False)
    diff.to_parquet(out / "differences.parquet", index=False)

    js: dict[str, Any] = {"confidence_level": level, "reference": reference, "time_unit": "proof steps",
                          "lag": model.lag if model else None, "datasets": {}}
    if model is not None and model.coarse_T is not None:
        js["model_coarse_T"] = _jsonable(model.coarse_T)
    for r, row in zip(results, summary.to_dict("records")):
        entry = {k: _jsonable(v) for k, v in row.items()}
        for k in ("transition_matrix", "occupancy", "stay_probability"):
            entry[k] = _jsonable(r.point[k])
            if k in r.draws:
                lo, hi = interval(r.draws[k], level)
                entry[f"{k}_lo"], entry[f"{k}_hi"] = _jsonable(lo), _jsonable(hi)
        entry["n_bootstrap"] = int(len(next(iter(r.draws.values())))) if r.draws else 0
        js["datasets"][r.name] = entry
    js["differences"] = [{k: _jsonable(v) for k, v in d.items()} for d in diff.to_dict("records")]
    (out / "summary.json").write_text(json.dumps(js, indent=2))
    return js
