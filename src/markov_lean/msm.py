"""Markov state model over goal-state trajectories, fully automatic.

Time is counted in **proof steps**: the lag tau and every implied timescale
are numbers of tactic steps within one proof (the ``step`` column of the
``steps`` table, or another ordering column via ``MSMConfig.time_col``).  An
implied timescale of 6 means the slowest area-switching process relaxes over
about 6 tactic steps.

``run(steps)`` does every stage and every standard check without hand tuning:

1. featurize ``state_before`` (``featurize.py``);
2. choose the number of k-means microstates by cross-validated VAMP-2 score
   over held-out *proofs* (deeptime ``MarkovStateModel.score``);
3. cluster with scikit-learn ``MiniBatchKMeans``;
4. count transitions at each lag between consecutive steps *within each
   proof* (each proof is its own discrete trajectory, so nothing is counted
   from the end of one proof to the start of the next);
5. keep the largest strongly connected set and report what was dropped;
6. estimate a reversible maximum-likelihood MSM at each candidate lag and
   choose the lag where the slow implied timescales stop changing;
7. choose the number of macrostates from the largest spectral gap and
   coarse-grain with PCCA+;
8. validate: connectivity, sampling per microstate, lag against proof length,
   implied-timescale convergence, resolved timescales, spectral gap, PCCA+
   crispness, a Chapman-Kolmogorov test, and timescale error bars (a seeded
   bootstrap over proofs by default, so the checks are reproducible).

Every check lands in ``report["checks"]`` with its value, threshold and
pass/fail; ``report["all_checks_passed"]`` summarizes them.

deeptime calls used (checked against deeptime 0.5.0):

* ``deeptime.markov.TransitionCountEstimator(lagtime, "sliding", n_states).fit(dtrajs).fetch_model()``
* ``TransitionCountModel.submodel_largest(directed=True)``
* ``deeptime.markov.msm.MaximumLikelihoodMSM(reversible=True).fit(counts).fetch_model()``
* ``MarkovStateModel.score(test_dtrajs, r=2, dim=d)`` (VAMP-2, for choosing microstates)
* ``MarkovStateModel.timescales(k)``, ``MarkovStateModel.eigenvalues(k)``
* ``MarkovStateModel.pcca(n)`` -> ``assignments``, ``memberships``,
  ``coarse_grained_transition_matrix``, ``coarse_grained_stationary_probability``
* ``MarkovStateModel.ck_test(models, n)`` -> ``predictions``, ``estimates``, ``lagtimes``
* ``deeptime.markov.msm.BayesianMSM(n_samples).fit(counts).fetch_model().gather_stats("timescales", k=k)``
  (only with ``error_method="bayesian"``; its sampler runs in C++ with no seed, so it is not reproducible)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM, MarkovStateModel, MaximumLikelihoodMSM
from sklearn.cluster import MiniBatchKMeans

from .featurize import make_featurizer

ASSIGNMENT_COLUMNS = ["proof_id", "step", "microstate", "macrostate", "membership"]
TIME_UNIT = "proof steps"


@dataclass
class MSMConfig:
    # what "time" is: the column that orders goal states within a proof
    time_col: str = "step"
    featurizer: str = "tfidf_svd"
    featurizer_kwargs: dict = field(default_factory=lambda: {"n_components": 64})
    # microstates: an int fixes k; a list is searched by cross-validated VAMP-2
    n_microstates: int | list[int] = field(default_factory=lambda: [25, 50, 100, 200])
    cv_folds: int = 3
    cv_lag: int = 1
    # lag in proof steps: an int fixes it; "auto" picks from its_lags by timescale convergence
    lag: int | str = "auto"
    its_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    its_tol: float = 0.15             # max relative change of the slow timescales between successive lags
    max_lag_fraction: float = 0.5     # auto lag: lag * ck_mlags stays within this fraction of the median proof length
    n_its: int = 10
    # macrostates: an int fixes it; None picks the largest spectral gap up to max_macrostates
    n_macrostates: int | None = None
    max_macrostates: int = 12
    ck_mlags: int = 4                 # CK test at lag, 2*lag, ..., ck_mlags*lag
    reversible: bool = True
    disconnected: str = "nearest"     # dropped microstates: "nearest" kept centroid's macrostate, or "drop" (-1)
    error_samples: int = 50           # resamples for timescale error bars; 0 skips them
    error_method: str = "bootstrap"   # "bootstrap" over proofs (seeded) or "bayesian" (deeptime BayesianMSM, unseeded)
    bayesian_samples: int | None = None  # deprecated alias for error_samples
    # check thresholds
    min_counts_kept: float = 0.95
    min_visits: int = 10
    max_sparse_fraction: float = 0.10
    min_gap_ratio: float = 1.5
    min_membership: float = 0.6
    ck_tol: float = 0.10
    kmeans_batch_size: int = 2048
    kmeans_n_init: int = 3
    seed: int = 0


@dataclass
class MSMResult:
    assignments: pd.DataFrame
    msm: MarkovStateModel
    microstates: np.ndarray            # microstate per row of the sorted steps table
    centers: np.ndarray                # k-means centers, (n_microstates, d)
    connected: np.ndarray              # microstate ids in the largest connected set
    micro_to_macro: np.ndarray         # (n_microstates,), -1 where unassigned
    coarse_T: np.ndarray               # PCCA+ coarse-grained transition matrix at the chosen lag
    timescales: pd.DataFrame           # lag, index, timescale (all in proof steps)
    ck: pd.DataFrame                   # lag, from_set, to_set, predicted, estimated
    microstate_cv: pd.DataFrame        # n_microstates, fold, vamp2
    report: dict

    @property
    def lag(self) -> int:
        return int(self.report["lag"])

    @property
    def checks(self) -> dict:
        return self.report["checks"]

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.assignments.to_parquet(out / "assignments.parquet", index=False)
        self.timescales.to_csv(out / "implied_timescales.csv", index=False)
        self.ck.to_csv(out / "ck_test.csv", index=False)
        self.microstate_cv.to_csv(out / "microstate_cv.csv", index=False)
        np.save(out / "coarse_T.npy", self.coarse_T)
        np.save(out / "micro_T.npy", self.msm.transition_matrix)
        (out / "msm_report.json").write_text(json.dumps(self.report, indent=2, default=_jsonable))


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------- building blocks

def sort_steps(steps: pd.DataFrame, time_col: str = "step") -> pd.DataFrame:
    """Steps ordered by proof then time, with a fresh index."""
    missing = {"proof_id", "step", "state_before", time_col} - set(steps.columns)
    if missing:
        raise ValueError(f"steps table is missing columns {sorted(missing)}")
    return steps.sort_values(["proof_id", time_col], kind="stable").reset_index(drop=True)


def cluster_microstates(X: np.ndarray, n_microstates: int, seed: int = 0,
                        batch_size: int = 2048, n_init: int = 3) -> MiniBatchKMeans:
    k = min(n_microstates, len(X))
    return MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=batch_size, n_init=n_init).fit(X)


def proof_codes(sorted_steps: pd.DataFrame) -> np.ndarray:
    return pd.factorize(sorted_steps["proof_id"], sort=False)[0]


def split_trajectories(sorted_steps: pd.DataFrame, labels: np.ndarray) -> list[np.ndarray]:
    """One discrete trajectory per proof.  Rows must already be sorted by proof then time."""
    bounds = np.flatnonzero(np.diff(proof_codes(sorted_steps))) + 1
    return [np.asarray(t, dtype=np.int64) for t in np.split(np.asarray(labels), bounds)]


def count_model(dtrajs: list[np.ndarray], lag: int, n_states: int):
    """Sliding-window counts at ``lag``, summed over proofs; nothing crosses a proof boundary."""
    usable = [d for d in dtrajs if len(d) > lag]
    if not usable:
        raise ValueError(f"no proof is longer than lag {lag}")
    return TransitionCountEstimator(lagtime=lag, count_mode="sliding", n_states=n_states).fit(usable).fetch_model()


def estimate_msm(dtrajs: list[np.ndarray], lag: int, n_states: int, reversible: bool = True):
    """MLE MSM on the largest strongly connected set.  Returns (msm, full count model)."""
    counts = count_model(dtrajs, lag, n_states)
    largest = counts.submodel_largest(directed=True)
    msm = MaximumLikelihoodMSM(reversible=reversible).fit(largest).fetch_model()
    return msm, counts


def pick_n_macrostates(msm: MarkovStateModel, max_k: int = 12) -> int:
    """Number of metastable sets at the largest spectral gap.

    With eigenvalues 1 = l1 >= l2 >= ..., k sets need k-1 slow processes, so
    choose k in 2..max_k maximizing l_k - l_{k+1}.
    """
    k_max = min(max_k, msm.n_states - 1)
    if k_max < 2:
        return max(1, k_max)
    ev = np.sort(np.abs(np.real(msm.eigenvalues(k_max + 1))))[::-1]
    gaps = ev[1:k_max] - ev[2:k_max + 1]
    return int(np.argmax(gaps)) + 2


def select_n_microstates(X: np.ndarray, codes: np.ndarray, candidates: list[int], lag: int, dim: int,
                         folds: int, seed: int, reversible: bool = True, **km) -> tuple[int, pd.DataFrame]:
    """Cross-validated VAMP-2 over held-out proofs; returns (best k, per-fold scores).

    Proofs, not steps, are split into folds, so a test proof is never seen in
    training.  ``dim`` caps the number of singular values scored, which keeps
    larger k from winning just by having more of them.
    """
    rng = np.random.default_rng(seed)
    n_proofs = int(codes.max()) + 1
    fold_of = rng.permutation(n_proofs) % folds
    row_fold = fold_of[codes]
    rows = []
    for k in candidates:
        if k >= len(X):
            continue
        for f in range(folds):
            tr, te = row_fold != f, row_fold == f
            model = cluster_microstates(X[tr], k, seed, **km)
            d_tr = np.split(model.labels_.astype(np.int64), np.flatnonzero(np.diff(codes[tr])) + 1)
            d_te = np.split(model.predict(X[te]).astype(np.int64), np.flatnonzero(np.diff(codes[te])) + 1)
            try:
                msm, _ = estimate_msm(d_tr, lag, model.n_clusters, reversible)
                s = float(msm.score([d for d in d_te if len(d) > lag], r=2, dim=min(dim, msm.n_states)))
            except (ValueError, np.linalg.LinAlgError):
                s = float("nan")
            rows.append({"n_microstates": k, "fold": f, "vamp2": s})
    cv = pd.DataFrame(rows, columns=["n_microstates", "fold", "vamp2"])
    means = cv.groupby("n_microstates")["vamp2"].mean().dropna()
    if means.empty:
        raise ValueError("VAMP-2 cross-validation failed for every candidate")
    return int(means.idxmax()), cv


def implied_timescales_table(dtrajs, lags, n_states, n_its=10, reversible=True) -> tuple[pd.DataFrame, dict]:
    rows, models = [], {}
    for lag in lags:
        try:
            m, _ = estimate_msm(dtrajs, lag, n_states, reversible)
        except ValueError:
            continue
        models[lag] = m
        n = min(n_its, m.n_states - 1)
        for i, t in enumerate(m.timescales(n) if n >= 1 else []):
            rows.append({"lag": lag, "index": i + 1, "timescale": float(t)})
    return pd.DataFrame(rows, columns=["lag", "index", "timescale"]), models


def its_relative_change(its: pd.DataFrame, n_slow: int) -> pd.Series:
    """Max relative change of the ``n_slow`` slowest timescales from each lag to the next."""
    piv = its[its["index"] <= n_slow].pivot(index="lag", columns="index", values="timescale").sort_index()
    rel = (piv.shift(-1) - piv).abs() / piv
    return rel.max(axis=1).iloc[:-1]


def choose_lag(its: pd.DataFrame, n_slow: int, tol: float, max_lag: int) -> tuple[int, bool]:
    """Smallest lag (<= max_lag) after which the slow timescales change by < tol.

    Returns (lag, converged).  If none converges, the lag with the smallest
    change is returned and ``converged`` is False.
    """
    rel = its_relative_change(its, n_slow)
    rel = rel[rel.index <= max_lag]
    if rel.empty:
        return int(its["lag"].min()), False
    ok = rel[rel < tol]
    if len(ok):
        return int(ok.index[0]), True
    return int(rel.idxmin()), False


def ck_test_table(msm: MarkovStateModel, dtrajs, lag, mlags, n_sets, n_states, reversible=True) -> pd.DataFrame:
    """Chapman-Kolmogorov test: T(lag)^k against MSMs estimated at k*lag, on PCCA+ sets."""
    cols = ["lag", "from_set", "to_set", "predicted", "estimated"]
    models = []
    for k in range(1, mlags + 1):
        try:
            models.append(estimate_msm(dtrajs, k * lag, n_states, reversible)[0])
        except ValueError:
            break
    if len(models) < 2:
        return pd.DataFrame(columns=cols)
    ck = msm.ck_test(models, n_sets, include_lag0=False)
    pred, est = np.real(np.asarray(ck.predictions)), np.real(np.asarray(ck.estimates))
    rows = [{"lag": int(L), "from_set": i, "to_set": j, "predicted": float(pred[li, i, j]),
             "estimated": float(est[li, i, j])}
            for li, L in enumerate(ck.lagtimes) for i in range(n_sets) for j in range(n_sets)]
    return pd.DataFrame(rows, columns=cols)


def bootstrap_timescales(dtrajs, lag, n_states, k, n_samples, seed, reversible=True) -> dict:
    """Mean and 95% interval of the k slowest timescales, bootstrapping whole proofs.

    Each resample draws proofs with replacement (a seeded numpy Generator) and
    refits the maximum-likelihood MSM, so the same inputs always give the same
    interval.  Resampling proofs rather than steps keeps within-proof
    correlation intact.
    """
    usable = [d for d in dtrajs if len(d) > lag]
    rng = np.random.default_rng(seed)
    draws = np.full((n_samples, k), np.nan)
    for b in range(n_samples):
        idx = rng.integers(len(usable), size=len(usable))
        try:
            m, _ = estimate_msm([usable[i] for i in idx], lag, n_states, reversible)
        except ValueError:
            continue
        t = m.timescales(min(k, m.n_states - 1))
        draws[b, :len(t)] = t
    return {"method": "bootstrap", "mean": np.nanmean(draws, 0).tolist(),
            "lower": np.nanpercentile(draws, 2.5, 0).tolist(),
            "upper": np.nanpercentile(draws, 97.5, 0).tolist(), "n_samples": n_samples}


def bayesian_timescales(dtrajs, lag, n_states, k, n_samples, reversible=True) -> dict:
    """Mean and 95% interval of the k slowest timescales from deeptime's Bayesian MSM.

    deeptime 0.5.0 samples transition matrices in C++ without a seed, so this
    interval changes from run to run; use ``bootstrap_timescales`` when the
    result must be reproducible.
    """
    # effective counts (deeptime's statistically uncorrelated count estimate) so the
    # posterior is not overconfident from overlapping sliding windows
    usable = [d for d in dtrajs if len(d) > lag]
    counts = (TransitionCountEstimator(lagtime=lag, count_mode="effective", n_states=n_states)
              .fit(usable).fetch_model().submodel_largest(directed=True))
    post = BayesianMSM(n_samples=n_samples, reversible=reversible).fit(counts).fetch_model()
    stats = post.gather_stats("timescales", k=k, confidence=0.95)
    return {"method": "bayesian", "mean": np.asarray(stats.mean).tolist(), "lower": np.asarray(stats.L).tolist(),
            "upper": np.asarray(stats.R).tolist(), "n_samples": n_samples}


def map_microstates(n_micro: int, connected: np.ndarray, assignments: np.ndarray, memberships: np.ndarray,
                    centers: np.ndarray, disconnected: str = "nearest") -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Macrostate and membership for every microstate, including ones outside the connected set.

    ``assignments`` and ``memberships`` are PCCA+ outputs indexed like
    ``connected``.  Dropped microstates get -1 and NaN with ``"drop"``, or the
    values of the kept microstate with the nearest k-means center with
    ``"nearest"``.  Returns (macrostate, membership, reassigned microstate ids).
    """
    if disconnected not in ("nearest", "drop"):
        raise ValueError("disconnected must be 'nearest' or 'drop'")
    connected = np.asarray(connected, dtype=np.int64)
    dropped = np.setdiff1d(np.arange(n_micro), connected)
    macro = np.full(n_micro, -1, dtype=np.int64)
    member = np.full(n_micro, np.nan)
    macro[connected] = assignments
    member[connected] = np.asarray(memberships)[np.arange(len(connected)), assignments]
    if disconnected == "drop" or not len(dropped):
        return macro, member, []
    d2 = ((centers[dropped, None, :] - centers[None, connected, :]) ** 2).sum(-1)
    nearest = connected[np.argmin(d2, axis=1)]
    macro[dropped] = macro[nearest]
    member[dropped] = member[nearest]
    return macro, member, dropped.tolist()


def _check(passed, value, threshold, detail) -> dict:
    return {"passed": bool(passed), "value": value, "threshold": threshold, "detail": detail}


# ---------------------------------------------------------------- pipeline

def run(steps: pd.DataFrame, config: MSMConfig | None = None) -> MSMResult:
    """Full automatic pipeline on a ``steps`` table: assignments plus every check."""
    cfg = config or MSMConfig()
    t0 = time.perf_counter()
    df = sort_steps(steps, cfg.time_col)
    codes = proof_codes(df)
    lens = np.bincount(codes)
    median_len = float(np.median(lens))
    texts = df["state_before"].fillna("").astype(str).tolist()

    # 1. features
    fkw = dict(cfg.featurizer_kwargs)
    if cfg.featurizer == "tfidf_svd":
        fkw.setdefault("seed", cfg.seed)
    X = make_featurizer(cfg.featurizer, **fkw).fit_transform(texts)
    t_feat = time.perf_counter()

    # 2-3. microstates
    km = {"batch_size": cfg.kmeans_batch_size, "n_init": cfg.kmeans_n_init}
    if isinstance(cfg.n_microstates, int):
        n_micro_req, cv = cfg.n_microstates, pd.DataFrame(columns=["n_microstates", "fold", "vamp2"])
        micro_by = "config"
    else:
        n_micro_req, cv = select_n_microstates(
            X, codes, list(cfg.n_microstates), cfg.cv_lag, cfg.max_macrostates + 1,
            cfg.cv_folds, cfg.seed, cfg.reversible, **km)
        micro_by = "vamp2_cv"
    kmeans = cluster_microstates(X, n_micro_req, cfg.seed, **km)
    micro = kmeans.labels_.astype(np.int64)
    centers = kmeans.cluster_centers_
    n_micro = centers.shape[0]
    dtrajs = split_trajectories(df, micro)
    t_clust = time.perf_counter()

    # 4-6. implied timescales over lags, then the lag
    lags = sorted(set(cfg.its_lags) | ({cfg.lag} if isinstance(cfg.lag, int) else set()))
    its, models = implied_timescales_table(dtrajs, lags, n_micro, cfg.n_its, cfg.reversible)
    if not models:
        raise ValueError("no lag could be estimated; proofs are too short")
    first = models[min(models)]
    n_macro_prelim = cfg.n_macrostates or pick_n_macrostates(first, cfg.max_macrostates)
    max_lag = max(1, int(cfg.max_lag_fraction * median_len // max(cfg.ck_mlags, 1)))
    if cfg.lag == "auto":
        lag, its_converged = choose_lag(its, max(n_macro_prelim - 1, 1), cfg.its_tol, max_lag)
        lag_by = "its_convergence"
    else:
        lag = int(cfg.lag)
        rel = its_relative_change(its, max(n_macro_prelim - 1, 1))
        its_converged = bool(lag in rel.index and rel.loc[lag] < cfg.its_tol)
        lag_by = "config"
    msm = models.get(lag) or estimate_msm(dtrajs, lag, n_micro, cfg.reversible)[0]
    counts = count_model(dtrajs, lag, n_micro)

    connected = np.asarray(msm.count_model.state_symbols, dtype=np.int64)
    dropped = np.setdiff1d(np.arange(n_micro), connected)

    # 7. macrostates
    n_macro = cfg.n_macrostates or pick_n_macrostates(msm, cfg.max_macrostates)
    n_macro = int(min(n_macro, msm.n_states))
    pcca = msm.pcca(n_macro)
    micro_to_macro, micro_membership, reassigned = map_microstates(
        n_micro, connected, pcca.assignments, pcca.memberships, centers, cfg.disconnected)
    assignments = pd.DataFrame({
        "proof_id": df["proof_id"].astype(str).values,
        "step": df["step"].astype("int64").values,
        "microstate": micro,
        "macrostate": micro_to_macro[micro],
        "membership": micro_membership[micro],
    })[ASSIGNMENT_COLUMNS]
    t_msm = time.perf_counter()

    # 8. validation
    ck = ck_test_table(msm, dtrajs, lag, cfg.ck_mlags, n_macro, n_micro, cfg.reversible)
    ck_err = float((ck["predicted"] - ck["estimated"]).abs().max()) if len(ck) else float("nan")
    n_slow = max(n_macro - 1, 1)
    ts = msm.timescales(min(cfg.n_its, msm.n_states - 1))
    n_err = cfg.error_samples if cfg.bayesian_samples is None else cfg.bayesian_samples
    bayes = None
    if n_err > 0:
        if cfg.error_method == "bootstrap":
            bayes = bootstrap_timescales(dtrajs, lag, n_micro, n_slow, n_err, cfg.seed, cfg.reversible)
        elif cfg.error_method == "bayesian":
            bayes = bayesian_timescales(dtrajs, lag, n_micro, n_slow, n_err, cfg.reversible)
        else:
            raise ValueError("error_method must be 'bootstrap' or 'bayesian'")
    t_val = time.perf_counter()

    visits = np.bincount(micro, minlength=n_micro)[connected]
    sparse_frac = float((visits < cfg.min_visits).mean())
    kept_frac = float(msm.count_model.count_matrix.sum() / max(counts.count_matrix.sum(), 1))
    proofs_long = float((lens > lag * cfg.ck_mlags).mean())
    gap_ratio = float(ts[n_slow - 1] / ts[n_slow]) if len(ts) > n_slow else float("nan")
    mean_membership = float(np.nanmean(assignments["membership"]))
    rel = its_relative_change(its, n_slow)
    rel_at_lag = float(rel.loc[lag]) if lag in rel.index else float("nan")
    checks = {
        "connectivity": _check(kept_frac >= cfg.min_counts_kept, kept_frac, cfg.min_counts_kept,
                               f"fraction of transition counts inside the largest connected set; "
                               f"{len(dropped)} of {n_micro} microstates dropped"),
        "sampling": _check(sparse_frac <= cfg.max_sparse_fraction, sparse_frac, cfg.max_sparse_fraction,
                           f"fraction of kept microstates visited fewer than {cfg.min_visits} times"),
        "lag_vs_proof_length": _check(proofs_long >= 0.5, proofs_long, 0.5,
                                      f"fraction of proofs longer than {cfg.ck_mlags} x lag = "
                                      f"{lag * cfg.ck_mlags} steps (median length {median_len:g})"),
        "its_converged": _check(its_converged, rel_at_lag, cfg.its_tol,
                                f"max relative change of the {n_slow} slowest timescales from lag {lag} to the next lag"),
        "timescales_resolved": _check(bool(len(ts) >= n_slow and ts[n_slow - 1] > lag),
                                      float(ts[n_slow - 1]) if len(ts) >= n_slow else float("nan"), lag,
                                      f"timescale {n_slow} (slowest one PCCA+ uses) must exceed the lag, in proof steps"),
        "spectral_gap": _check(gap_ratio >= cfg.min_gap_ratio, gap_ratio, cfg.min_gap_ratio,
                               f"ratio of timescale {n_slow} to timescale {n_slow + 1}"),
        "pcca_crisp": _check(mean_membership >= cfg.min_membership, mean_membership, cfg.min_membership,
                             "mean PCCA+ membership of each step's assigned macrostate"),
        "chapman_kolmogorov": _check(bool(np.isfinite(ck_err) and ck_err <= cfg.ck_tol), ck_err, cfg.ck_tol,
                                     f"max |predicted - estimated| set-to-set probability up to {cfg.ck_mlags} x lag"),
    }
    if bayes is not None:
        lo, hi = np.asarray(bayes["lower"]), np.asarray(bayes["upper"])
        nxt = its[(its["lag"] > lag)].sort_values("lag")
        nxt = nxt[nxt["lag"] == nxt["lag"].min()] if len(nxt) else nxt
        inside = None
        if len(nxt):
            t_next = nxt.sort_values("index")["timescale"].to_numpy()[:n_slow]
            inside = float(np.mean((t_next >= lo[:len(t_next)]) & (t_next <= hi[:len(t_next)])))
        checks["its_within_errors"] = _check(
            inside is None or inside >= 0.5, inside, 0.5,
            f"share of the slow timescales at the next lag inside the 95% {bayes['method']} interval at the chosen lag")

    report = {
        "time_unit": TIME_UNIT,
        "time_col": cfg.time_col,
        "config": asdict(cfg),
        "n_steps": int(len(df)),
        "n_proofs": int(len(lens)),
        "proof_length": {"min": int(lens.min()), "median": median_len, "max": int(lens.max())},
        "feature_dim": int(X.shape[1]),
        "n_microstates": int(n_micro),
        "n_microstates_chosen_by": micro_by,
        "lag": int(lag),
        "lag_chosen_by": lag_by,
        "max_lag_considered": int(max_lag),
        "n_macrostates": n_macro,
        "n_macrostates_chosen_by": "config" if cfg.n_macrostates else "spectral_gap",
        "n_macrostates_at_first_lag": int(n_macro_prelim),
        "n_transitions_counted": float(counts.count_matrix.sum()),
        "connected_set": {
            "n_kept": int(len(connected)),
            "n_dropped": int(len(dropped)),
            "dropped_microstates": dropped.tolist(),
            "fraction_steps_kept": float(np.isin(micro, connected).mean()),
            "fraction_counts_kept": kept_frac,
            "dropped_handling": cfg.disconnected,
            "reassigned_microstates": reassigned,
        },
        "timescales_steps": [float(t) for t in ts],
        "timescale_errors_steps": bayes,
        "coarse_stationary": np.asarray(pcca.coarse_grained_stationary_probability).tolist(),
        "checks": checks,
        "all_checks_passed": bool(all(c["passed"] for c in checks.values())),
        "seconds": {
            "featurize": round(t_feat - t0, 3),
            "microstates": round(t_clust - t_feat, 3),
            "its_msm_pcca": round(t_msm - t_clust, 3),
            "validation": round(t_val - t_msm, 3),
        },
    }
    return MSMResult(
        assignments=assignments, msm=msm, microstates=micro, centers=centers, connected=connected,
        micro_to_macro=micro_to_macro, coarse_T=np.real(np.asarray(pcca.coarse_grained_transition_matrix)),
        timescales=its, ck=ck, microstate_cv=cv, report=report,
    )
