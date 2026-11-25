"""Within-proof area switching: do single proofs move between areas, and is that more than chance?

``markov-lean switches --run results/pilot3`` reads a finished pilot run and
writes ``<run>/switches/``.  It makes no API call and refits nothing.

Gate 1 (``markov-lean significance``) shows that macrostates line up with the
judges' MSC labels.  It cannot show that one proof moves from one area to
another: a partition can agree with the labels while every proof sits in one
area.  This module tests movement directly, inside each proof.

**Area sequences.**  Each proof gives one sequence of areas, in step order, from
each source (analysed on its own):

* ``judge``: step-level MSC labels from ``<run>/judge/labels.parquet``, chosen
  as ``markov-lean bridges`` chooses them (the ``consensus`` judge, else the two
  judges' agreed steps, else the one judge; ``--judge`` picks one).  The judges
  see an even sample of at most 48 steps of a long proof, so the sequence is the
  labelled steps in order, not necessarily consecutive tactics.
* ``macrostate``: PCCA+ macrostates from ``<run>/msm/assignments.parquet``.
* ``true_area``: synthetic runs only, used in place of ``judge`` when there are no
  labels and the steps carry the generating area.

Proofs with fewer than two area-labelled steps cannot switch and are dropped.

**Statistics** (one value per proof, averaged with every proof weighted equally):

* ``has_switch``: share of proofs with at least one switch (two consecutive
  labelled steps in different areas).
* ``switch_rate``: switches divided by the proof's moves (length minus one), so
  long proofs do not dominate.
* ``mean_dwell``: the proof's length divided by its number of runs, the average
  number of steps spent in an area before switching.  Runs touching either end
  of a proof are cut short by the proof itself (censored); every null below is
  cut the same way, so comparisons stay fair.
* ``off_mode``: share of the proof's steps outside its most common area, the
  amount of the proof spent away from its main area.

**Nulls.**  Each keeps every proof's length.  ``n_perm`` draws per null give
permutation p-values (``(1 + #{null as or more extreme}) / (n_perm + 1)``,
two-sided as twice the smaller one-sided value), and per-proof null means give
paired differences ``observed - null`` whose 95% intervals are percentile
bootstraps over proofs (``n_boot`` resamples).  The resampling unit is always a
proof, as in ``markov-lean significance``.

* ``shuffle``: each proof's own areas in a random order.  It keeps each proof's
  area composition exactly, so ``off_mode`` cannot move; it tests ordering.  A
  switch rate below this null (dwell above it) means a proof's steps in one area
  come in runs: segments of a proof, not step-to-step flicker.
* ``markov``: a first-order chain over areas fitted per Mathlib folder (one
  chain when folders are unknown; ``--markov-group first`` groups by the proof's
  first area), started at the proof's observed first area.  The chain matches
  each area's switch rates, so it tests how switches spread over proofs: fewer
  proofs with a switch than the chain predicts means switching is concentrated
  in a minority of proofs that move, rather than a constant hazard; a longer
  dwell than the chain means memory beyond one step.  Grouping by the proof's
  most common area (``--markov-group mode``) is offered but conditions on the
  outcome: it makes each group's chain stickier than the truth and rejects a
  genuine Markov chain, so it is not the default.
* ``pooled``: all labelled steps of proofs in the same Mathlib folder (or all
  proofs, ``--pooled-within all``) dealt out at random to those proofs, which
  keeps the folder's area composition but not the proof's.  Switching far below
  this null means proofs are coherent about their area, beyond their folder.
* ``noise``: every step in the proof's most common area, then each step moved to
  another area with probability ``epsilon`` (drawn from the overall area mix).
  This is a proof that never moves, seen through a noisy labeller.  ``off_mode``
  and ``has_switch`` above this null mean movement beyond label noise.  For the
  judge source ``epsilon`` is estimated from the two judges' raw step-level
  agreement ``a`` as ``1 - sqrt(a)`` (independent errors that rarely coincide);
  ``--noise-rate`` sets it for every source.  Without either it is skipped.

**Strata.**  Every statistic and null is repeated by Mathlib folder (folders with
at least ``--min-proofs`` proofs; the rest pooled as ``other``) and by proof
length (labelled steps: 2-3, 4-7, 8-15, 16-31, 32+), with Holm and
Benjamini-Hochberg corrections across the strata of each kind.

**Outputs** in ``--out`` (default ``<run>/switches``):

| file | what it holds |
|---|---|
| ``switches.json`` | settings, inputs, every statistic with CI, null means, paired differences, p-values, strata |
| ``switches.csv`` | the same as one long table (source x stratum x null x statistic) |
| ``proofs.csv`` | per proof and source: length, folder, switches, statistics and per-proof null means |
| ``dwell.csv`` | dwell (run length) distribution, observed and under each null |
| ``REPORT.md`` | a readable summary |
| ``table_switches.tex`` | booktabs tabular for the paper: observed [95% CI] and each null's mean with its p |
| ``switches.pdf`` / ``.png`` | statistics against nulls, dwell distribution, switch rate by length (``figures`` extra) |

Hooking into ``markov-lean paper``: add a ``switches`` source the way ``bridges``
is added (``--switches`` folder, or ``<results>/**/switches`` when found), copy
``table_switches.tex`` and ``switches.pdf``, and read headline numbers from
``switches.json["headline"]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from . import evaluate as ev
from .significance import _weights, benjamini_hochberg, holm, interval

KEY = ["proof_id", "step"]
STATS = ("has_switch", "switch_rate", "mean_dwell", "off_mode")
NULLS = ("shuffle", "markov", "pooled", "noise")
STAT_LABELS = {
    "has_switch": "Proofs with a switch",
    "switch_rate": "Switches per move",
    "mean_dwell": "Mean dwell (steps)",
    "off_mode": "Share off main area",
}
NULL_LABELS = {"shuffle": "Shuffle", "markov": "Markov", "pooled": "Folder pool", "noise": "Noise floor"}
SOURCE_LABELS = {"judge": "Judge MSC", "true_area": "True area", "macrostate": "Macrostate",
                 "premise": "Premise MSC", "premise_carry": "Premise (carried)", "premise_strict": "Premise (strict)",
                 "premise_glue": "Premise (with glue)", "premise_folder": "Premise folder"}
LENGTH_EDGES = (2, 4, 8, 16, 32)


# --------------------------------------------------------------------------- sequences


class Sequences:
    """Area sequences of many proofs, concatenated in (proof, step) order.

    ``y`` holds integer area codes, ``pidx`` the proof index of each step,
    ``start`` the offset of each proof's first step and ``length`` its number of
    labelled steps.  Proofs are sorted by ``proof_id``.
    """

    def __init__(self, areas: pd.DataFrame, min_length: int = 2):
        a = areas[KEY + ["area"]].copy()
        a["proof_id"] = a["proof_id"].astype(str)
        a["step"] = pd.to_numeric(a["step"]).astype("int64")
        a["area"] = a["area"].astype(str)
        a = a[a["area"].str.len() > 0].drop_duplicates(KEY).sort_values(KEY, kind="stable")
        n = a.groupby("proof_id", sort=True)["step"].transform("size")
        self.n_dropped = int(a.loc[n < min_length, "proof_id"].nunique())
        a = a[n >= min_length].reset_index(drop=True)
        codes, names = pd.factorize(a["area"], sort=True)
        pcodes, pnames = pd.factorize(a["proof_id"], sort=True)
        self.areas: list[str] = list(names)
        self.proofs: list[str] = list(pnames)
        self.y = codes.astype(np.int64)
        self.pidx = pcodes.astype(np.int64)
        self.steps = a["step"].to_numpy()
        self.P = len(self.proofs)
        self.N = len(self.y)
        self.K = max(len(self.areas), 1)
        self.length = np.bincount(self.pidx, minlength=self.P).astype(np.int64)
        self.start = np.concatenate([[0], np.cumsum(self.length)[:-1]]).astype(np.int64) if self.P else np.zeros(0, int)
        first = np.zeros(self.N, dtype=bool)
        first[self.start] = True
        self.first = first
        # moves: position i -> i + 1 inside one proof
        self.move_ok = ~first[1:] if self.N else np.zeros(0, bool)
        self.move_proof = self.pidx[1:]
        self.M_move = sparse.csr_matrix(
            (self.move_ok.astype(np.float32), (np.arange(self.N - 1), self.move_proof)),
            shape=(max(self.N - 1, 0), self.P))

    def counts(self, y: np.ndarray | None = None) -> np.ndarray:
        """Per-proof area counts, shape ``(P, K)``."""
        y = self.y if y is None else y
        return np.bincount(self.pidx * self.K + y, minlength=self.P * self.K).reshape(self.P, self.K)

    def mode(self) -> np.ndarray:
        """Each proof's most common area (ties: the lowest code)."""
        return self.counts().argmax(1)


def per_proof_stats(seq: Sequences, Y: np.ndarray) -> dict[str, np.ndarray]:
    """Statistics of each proof for a stack of sequences ``Y`` of shape ``(R, N)``; each value ``(R, P)``."""
    Y = np.atleast_2d(Y)
    R = Y.shape[0]
    L = seq.length.astype(float)
    sw = (Y[:, 1:] != Y[:, :-1]).astype(np.float32)
    n_sw = np.asarray((seq.M_move.T @ sw.T).T, dtype=float).reshape(R, seq.P)
    maxc = np.empty((R, seq.P))
    for r in range(R):
        maxc[r] = seq.counts(Y[r]).max(1)
    return {
        "n_switches": n_sw,
        "has_switch": (n_sw > 0).astype(float),
        "switch_rate": n_sw / (L - 1.0),
        "mean_dwell": L / (n_sw + 1.0),
        "off_mode": 1.0 - maxc / L,
    }


def expected_shuffle_rate(seq: Sequences) -> np.ndarray:
    """Exact expected switch rate of each proof under a within-proof shuffle.

    For a sequence of length ``L`` with area counts ``n_i``, two neighbours share
    an area with probability ``sum n_i (n_i - 1) / (L (L - 1))``.
    """
    C = seq.counts().astype(float)
    L = seq.length.astype(float)
    return 1.0 - (C * (C - 1)).sum(1) / (L * (L - 1))


def run_lengths(seq: Sequences, y: np.ndarray) -> np.ndarray:
    """Lengths of every run (maximal stretch of one area inside one proof)."""
    if seq.N == 0:
        return np.zeros(0, int)
    boundary = seq.first.copy()
    boundary[1:] |= y[1:] != y[:-1]
    rid = np.cumsum(boundary) - 1
    return np.bincount(rid)


# --------------------------------------------------------------------------- nulls


def _row_normalize(C: np.ndarray) -> np.ndarray:
    s = C.sum(-1, keepdims=True)
    return np.divide(C, s, out=np.full_like(C, 1.0 / C.shape[-1], dtype=float), where=s > 0)


def _off_diagonal_cdf(T: np.ndarray) -> np.ndarray:
    """For each row of a ``(..., K, K)`` stochastic matrix, the CDF of the next state given it differs."""
    K = T.shape[-1]
    off = T * (1 - np.eye(K))
    off = _row_normalize(off)
    # rows with no off-diagonal mass: move uniformly to another state
    empty = off.sum(-1) == 0
    if empty.any():
        uni = (1 - np.eye(K)) / max(K - 1, 1)
        off = np.where(empty[..., None], np.broadcast_to(uni, off.shape), off)
    cdf = np.cumsum(off, -1)
    cdf[..., -1] = 1.0
    return cdf


def _draw_other(cdf_rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(len(cdf_rows))
    return np.minimum((cdf_rows < u[:, None]).sum(1), cdf_rows.shape[1] - 1)


def fit_markov(seq: Sequences, group: np.ndarray, pseudo: float = 1.0) -> np.ndarray:
    """Transition matrix per group, shape ``(G, K, K)``, shrunk toward the pooled matrix with weight ``pseudo``."""
    K = seq.K
    G = int(group.max()) + 1 if len(group) else 1
    src = seq.y[:-1][seq.move_ok]
    dst = seq.y[1:][seq.move_ok]
    g = group[seq.move_proof[seq.move_ok]]
    C = np.bincount((g * K + src) * K + dst, minlength=G * K * K).reshape(G, K, K).astype(float)
    T0 = _row_normalize(C.sum(0) + 1e-3)
    return _row_normalize(C + pseudo * T0[None])


def simulate_markov(seq: Sequences, T: np.ndarray, group: np.ndarray, R: int,
                    rng: np.random.Generator) -> np.ndarray:
    """``R`` sequences per proof from the group's chain, started at the proof's observed first area."""
    K = seq.K
    stay = np.diagonal(T, axis1=1, axis2=2).reshape(-1)          # (G*K,)
    cdf = _off_diagonal_cdf(T).reshape(-1, K)                     # (G*K, K)
    Y = np.empty((R, seq.N), dtype=np.int64)
    Y[:, seq.start] = seq.y[seq.start][None]
    for t in range(1, int(seq.length.max()) if seq.P else 0):
        act = np.flatnonzero(seq.length > t)
        pos = seq.start[act] + t
        cur = Y[:, pos - 1]
        rows = (group[act][None] * K + cur).reshape(-1)
        nxt = cur.reshape(-1).copy()
        leave = rng.random(len(rows)) >= stay[rows]
        if leave.any():
            nxt[leave] = _draw_other(cdf[rows[leave]], rng)
        Y[:, pos] = nxt.reshape(R, len(act))
    return Y


def simulate_shuffle(seq: Sequences, R: int, rng: np.random.Generator) -> np.ndarray:
    order = np.argsort(seq.pidx[None] + rng.random((R, seq.N)), axis=1, kind="stable")
    return seq.y[order]


def simulate_pooled(seq: Sequences, block: np.ndarray, R: int, rng: np.random.Generator) -> np.ndarray:
    """Labels permuted among all steps of proofs in the same ``block`` (per-proof block index)."""
    sb = block[seq.pidx]
    srt = np.argsort(sb, kind="stable")                   # positions grouped by block
    Y = np.empty((R, seq.N), dtype=np.int64)
    for r in range(R):
        perm = srt[np.argsort(sb[srt] + rng.random(seq.N), kind="stable")]
        Y[r, srt] = seq.y[perm]
    return Y


def simulate_noise(seq: Sequences, epsilon: float, R: int, rng: np.random.Generator) -> np.ndarray:
    """Every step in the proof's main area, each moved to another area with probability ``epsilon``."""
    K = seq.K
    mode = seq.mode()[seq.pidx]
    freq = np.bincount(seq.y, minlength=K).astype(float)
    T = np.broadcast_to(freq, (K, K)).copy()
    cdf = _off_diagonal_cdf(_row_normalize(T))
    Y = np.broadcast_to(mode, (R, seq.N)).copy()
    flip = rng.random((R, seq.N)) < epsilon
    if flip.any():
        rows = np.broadcast_to(mode, (R, seq.N))[flip]
        Y[flip] = _draw_other(cdf[rows], rng) if K > 1 else rows
    return Y


# --------------------------------------------------------------------------- strata


def length_bin(L: np.ndarray, edges=LENGTH_EDGES) -> np.ndarray:
    names = [f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:])] + [f"{edges[-1]}+"]
    idx = np.searchsorted(np.asarray(edges), L, side="right") - 1
    return np.asarray(names, dtype=object)[np.clip(idx, 0, len(names) - 1)]


def strata(proof_info: pd.DataFrame, min_proofs: int) -> list[tuple[str, str, np.ndarray]]:
    """``(kind, name, mask)`` for all proofs, each large folder and each length bin."""
    out = [("all", "all", np.ones(len(proof_info), bool))]
    if proof_info["folder"].notna().any():
        f = proof_info["folder"].fillna("unknown")
        counts = f.value_counts()
        big = [x for x in counts.index if counts[x] >= min_proofs and x != "unknown"]
        for name in sorted(big):
            out.append(("folder", name, (f == name).to_numpy()))
        rest = ~f.isin(big).to_numpy()
        if rest.sum() >= max(min_proofs, 1) and len(big):
            out.append(("folder", "other", rest))
    lb = proof_info["length_bin"]
    order = sorted(lb.unique(), key=lambda s: int(str(s).split("-")[0].rstrip("+")))
    for name in order:
        m = (lb == name).to_numpy()
        if m.sum() >= 1:
            out.append(("length", str(name), m))
    return out


# --------------------------------------------------------------------------- the test for one source


def _pvals(obs: float, draws: np.ndarray) -> dict:
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    if not np.isfinite(obs) or len(draws) == 0:
        nan = float("nan")
        return {"p_less": nan, "p_greater": nan, "p_two_sided": nan}
    tol = 1e-12 * max(1.0, abs(obs))
    R = len(draws)
    pl = (1 + np.sum(draws <= obs + tol)) / (R + 1)
    pg = (1 + np.sum(draws >= obs - tol)) / (R + 1)
    return {"p_less": float(pl), "p_greater": float(pg), "p_two_sided": float(min(1.0, 2 * min(pl, pg)))}


def analyse_source(seq: Sequences, folder: np.ndarray, nulls: list[str], *, n_perm: int = 1000, n_boot: int = 2000,
                seed: int = 0, noise_rate: float | None = None, markov_group: str = "folder",
                pooled_within: str = "folder", min_proofs: int = 30, chunk: int | None = None) -> dict:
    """Every statistic, null, stratum, interval and p-value for one source's sequences."""
    P = seq.P
    obs_pp = {k: v[0] for k, v in per_proof_stats(seq, seq.y[None]).items()}
    info = pd.DataFrame({"proof_id": seq.proofs, "length": seq.length, "folder": folder,
                         "length_bin": length_bin(seq.length)})
    st = strata(info, min_proofs)
    Wst = np.stack([m / max(m.sum(), 1) for _, _, m in st], axis=1)   # (P, S)

    if "noise" in nulls and noise_rate is None:
        nulls = [n for n in nulls if n != "noise"]
    mode = seq.mode()
    fcodes = pd.factorize(pd.Series(folder).fillna("unknown"), sort=True)[0].astype(np.int64)
    has_folder = pd.Series(folder).notna().any()
    if markov_group == "folder" and not has_folder:
        markov_group = "none"
    group = {"mode": mode, "folder": fcodes, "first": seq.y[seq.start],
             "none": np.zeros(P, np.int64)}[markov_group]
    T = fit_markov(seq, group) if "markov" in nulls else None
    block = fcodes if pooled_within == "folder" else np.zeros(P, np.int64)

    if chunk is None:
        chunk = int(max(1, min(50, 2e7 // max(seq.N, 1))))
    draws: dict[str, dict[str, list]] = {n: {s: [] for s in STATS} for n in nulls}
    pp_sum: dict[str, dict[str, np.ndarray]] = {n: {s: np.zeros(P) for s in STATS} for n in nulls}
    dwell: dict[str, np.ndarray] = {"observed": np.bincount(run_lengths(seq, seq.y))}
    for i, n in enumerate(nulls):
        rng = np.random.default_rng([seed, i])
        done = 0
        dw_draws = 0
        dw = np.zeros(1)
        while done < n_perm:
            R = min(chunk, n_perm - done)
            if n == "shuffle":
                Y = simulate_shuffle(seq, R, rng)
            elif n == "markov":
                Y = simulate_markov(seq, T, group, R, rng)
            elif n == "pooled":
                Y = simulate_pooled(seq, block, R, rng)
            else:
                Y = simulate_noise(seq, float(noise_rate), R, rng)
            pp = per_proof_stats(seq, Y)
            for s in STATS:
                draws[n][s].append(pp[s] @ Wst)
                pp_sum[n][s] += pp[s].sum(0)
            for r in range(min(R, max(0, 100 - dw_draws))):   # dwell histogram from the first 100 draws
                h = np.bincount(run_lengths(seq, Y[r]))
                if len(h) > len(dw):
                    dw = np.pad(dw, (0, len(h) - len(dw)))
                dw[:len(h)] += h
                dw_draws += 1
            done += R
        dwell[n] = dw / max(dw_draws, 1)
    draws = {n: {s: np.concatenate(v) for s, v in d.items()} for n, d in draws.items()}
    null_pp = {n: {s: pp_sum[n][s] / n_perm for s in STATS} for n in nulls}

    # paired bootstrap over proofs, per stratum: observed stats and observed - null per proof
    cols = [("obs", s) for s in STATS] + [(n, s) for n in nulls for s in STATS]
    V = np.stack([obs_pp[s] if src == "obs" else obs_pp[s] - null_pp[src][s] for src, s in cols], axis=1)
    rng = np.random.default_rng([seed, 99])
    rows = []
    strata_out = []
    for j, (kind, name, mask) in enumerate(st):
        idx = np.flatnonzero(mask)
        Vs = V[idx]
        boots = np.concatenate([W @ Vs / len(idx) for W in _weights(rng, len(idx), n_boot, 250)]) \
            if len(idx) else np.zeros((0, len(cols)))
        entry = {"kind": kind, "stratum": name, "n_proofs": int(len(idx)), "n_steps": int(seq.length[idx].sum()),
                 "statistics": {}}
        for s in STATS:
            c = cols.index(("obs", s))
            ov = float(Vs[:, c].mean()) if len(idx) else float("nan")
            e = {"observed": ov, "ci95": interval(boots[:, c]), "nulls": {}}
            for n in nulls:
                c = cols.index((n, s))
                dr = draws[n][s][:, j]
                sd = float(dr.std(ddof=1)) if len(dr) > 1 else float("nan")
                fixed = bool(np.isfinite(sd) and sd < 1e-12 and abs(dr.mean() - ov) < 1e-9)
                e["nulls"][n] = {"null_mean": float(dr.mean()), "null_sd": sd,
                                 "null_ci95": interval(dr), "difference": float(Vs[:, c].mean()),
                                 "difference_ci95": interval(boots[:, c]), "fixed_by_null": fixed,
                                 **_pvals(ov, dr)}
                if fixed:
                    e["nulls"][n].update({"p_less": 1.0, "p_greater": 1.0, "p_two_sided": 1.0})
            entry["statistics"][s] = e
        strata_out.append(entry)

    # corrections across strata of one kind, for each null and statistic
    for kind in ("folder", "length"):
        group_entries = [e for e in strata_out if e["kind"] == kind]
        if not group_entries:
            continue
        for s in STATS:
            for n in nulls:
                p = [e["statistics"][s]["nulls"][n]["p_two_sided"] for e in group_entries]
                for e, h, b in zip(group_entries, holm(p), benjamini_hochberg(p)):
                    e["statistics"][s]["nulls"][n].update({"holm": h, "bh": b})

    proofs = info.assign(n_switches=obs_pp["n_switches"].astype(int), n_areas=(seq.counts() > 0).sum(1),
                         main_area=[seq.areas[m] for m in mode], **{s: obs_pp[s] for s in STATS})
    for n in nulls:
        for s in STATS:
            proofs[f"{n}_{s}"] = null_pp[n][s]
    return {
        "n_proofs": int(P), "n_steps": int(seq.N), "n_areas": len(seq.areas), "areas": seq.areas,
        "n_dropped_short": seq.n_dropped, "nulls": nulls, "noise_rate": noise_rate,
        "markov_group": markov_group, "pooled_within": pooled_within,
        "strata": strata_out, "_proofs": proofs, "_dwell": dwell,
    }


# --------------------------------------------------------------------------- loading


def _read(path: Path) -> pd.DataFrame:
    from .bridges import read_table

    return read_table(path)


def estimate_noise(raw: pd.DataFrame | None) -> tuple[float | None, str]:
    """``1 - sqrt(agreement)`` from two judges' raw step-level labels, if there are two judges."""
    if raw is None or raw.empty:
        return None, "no raw labels"
    from .judge import paired

    raw = raw.assign(proof_id=raw["proof_id"].astype(str), step=pd.to_numeric(raw["step"]).astype("int64"))
    judges = [j for j in dict.fromkeys(raw["judge"].astype(str)) if j != "consensus"]
    if len(judges) < 2:
        return None, "fewer than two judges"
    p = paired(raw[raw["step"] >= 0], judges[0], judges[1])
    if p.empty:
        return None, "no step-level pairs"
    a = float((p["msc_a"].astype(str) == p["msc_b"].astype(str)).mean())
    return float(1 - np.sqrt(a)), f"1 - sqrt(step agreement {a:.3f}) of {judges[0]} vs {judges[1]}"


def load_sources(run: Path, steps_path=None, labels_path=None, assignments_path=None, config_path=None,
                 judge: str | None = None, sources: list[str] | None = None) -> dict:
    """Area tables per source, the steps table (for folders) and raw labels, from a run folder."""
    from .bridges import judge_areas, macrostate_areas

    run = Path(run)
    want = sources or ["judge", "macrostate"]
    steps = None
    if steps_path is None and (run / "judge" / "sample_steps.parquet").exists():
        steps_path = run / "judge" / "sample_steps.parquet"
    if steps_path is not None:
        steps = _read(Path(steps_path))
    elif config_path is not None:
        import yaml

        from . import pilot

        data = (yaml.safe_load(Path(config_path).read_text()) or {}).get("data") or {}
        steps = pilot.sample_proofs(pilot.load_steps(data), data.get("max_proofs"), int(data.get("min_steps", 2)),
                                    int(data.get("seed", 0)))
    if steps is not None:
        steps = steps.assign(proof_id=steps["proof_id"].astype(str))

    out: dict[str, Any] = {"steps": steps, "areas": {}, "notes": {}, "raw": None,
                           "inputs": {"steps": str(steps_path) if steps_path else None}}
    if "judge" in want:
        lp = Path(labels_path) if labels_path else run / "judge" / "labels.parquet"
        if lp.exists():
            raw = _read(lp)
            out["raw"] = raw
            a, used = judge_areas(raw, judge)
            if len(a):
                out["areas"]["judge"] = a
            out["notes"]["judge"] = used
            out["inputs"]["labels"] = str(lp)
        elif steps is not None and "true_area" in steps.columns and steps["true_area"].notna().any():
            t = steps[steps["true_area"].notna()]
            out["areas"]["true_area"] = pd.DataFrame({"proof_id": t["proof_id"], "step": t["step"],
                                                      "area": t["true_area"].astype(str)})
            out["notes"]["true_area"] = "synthetic true_area (no judge labels)"
    if "macrostate" in want:
        ap = Path(assignments_path) if assignments_path else run / "msm" / "assignments.parquet"
        if ap.exists():
            out["areas"]["macrostate"] = macrostate_areas(_read(ap))
            out["inputs"]["assignments"] = str(ap)
    return out


def _folders(seq: Sequences, steps: pd.DataFrame | None) -> np.ndarray:
    if steps is None or "file_path" not in steps.columns:
        return np.full(seq.P, None, dtype=object)
    fp = steps.drop_duplicates("proof_id").set_index("proof_id")["file_path"]
    return np.asarray([ev.folder_label(fp.get(p)) for p in seq.proofs], dtype=object)


# --------------------------------------------------------------------------- outputs


def long_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for src, r in results.items():
        for e in r["strata"]:
            for s, v in e["statistics"].items():
                base = {"source": src, "kind": e["kind"], "stratum": e["stratum"], "n_proofs": e["n_proofs"],
                        "statistic": s, "observed": v["observed"], "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1]}
                if not v["nulls"]:
                    rows.append(base)
                for n, d in v["nulls"].items():
                    rows.append({**base, "null": n, "null_mean": d["null_mean"], "null_sd": d["null_sd"],
                                 "difference": d["difference"], "diff_lo": d["difference_ci95"][0],
                                 "diff_hi": d["difference_ci95"][1], "p_less": d["p_less"],
                                 "p_greater": d["p_greater"], "p_two_sided": d["p_two_sided"],
                                 "holm": d.get("holm"), "bh": d.get("bh"), "fixed_by_null": d["fixed_by_null"]})
    return pd.DataFrame(rows)


def dwell_table(results: dict[str, dict], max_len: int = 64) -> pd.DataFrame:
    rows = []
    for src, r in results.items():
        for series, h in r["_dwell"].items():
            h = np.asarray(h, float)
            tot = h.sum()
            for d in range(1, min(len(h), max_len + 1)):
                rows.append({"source": src, "series": series, "dwell": d, "runs": h[d],
                             "share": h[d] / tot if tot else np.nan})
            if len(h) > max_len + 1:
                rows.append({"source": src, "series": series, "dwell": max_len + 1, "runs": h[max_len + 1:].sum(),
                             "share": h[max_len + 1:].sum() / tot})
    return pd.DataFrame(rows)


def _f(x, d=3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _p(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else (f"{x:.4f}" if x >= 1e-4 else f"{x:.1e}")


def _overall(r: dict) -> dict:
    return next(e for e in r["strata"] if e["kind"] == "all")


def headline(results: dict[str, dict]) -> dict:
    """Numbers the paper text can quote, per source."""
    out = {}
    for src, r in results.items():
        a = _overall(r)["statistics"]
        h = {s: a[s]["observed"] for s in STATS}
        h.update({f"{s}_ci95": a[s]["ci95"] for s in STATS})
        for n in r["nulls"]:
            h[f"switch_rate_{n}"] = a["switch_rate"]["nulls"][n]["null_mean"]
            h[f"p_switch_rate_{n}"] = a["switch_rate"]["nulls"][n]["p_two_sided"]
            h[f"has_switch_{n}"] = a["has_switch"]["nulls"][n]["null_mean"]
            h[f"p_has_switch_{n}"] = a["has_switch"]["nulls"][n]["p_two_sided"]
            h[f"off_mode_{n}"] = a["off_mode"]["nulls"][n]["null_mean"]
            h[f"p_off_mode_{n}"] = a["off_mode"]["nulls"][n]["p_two_sided"]
        h["n_proofs"] = r["n_proofs"]
        out[src] = h
    return out


def _tex_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    return r"$<\!10^{-3}$" if p < 1e-3 else f"{p:.3f}"


def table_tex(results: dict[str, dict]) -> str:
    """Booktabs tabular: per source and statistic, observed [95% CI] and each null's mean (two-sided p)."""
    nulls = [n for n in NULLS if any(n in r["nulls"] for r in results.values())]
    lines = [
        "% Within-proof area switching: observed [95% bootstrap CI over proofs] and each null's mean",
        "% (two-sided permutation p in parentheses). Written by `markov-lean switches`.",
        r"\begin{tabular}{ll" + "c" * (1 + len(nulls)) + "}",
        r"  \toprule",
        "  Areas & Statistic & Observed & " + " & ".join(NULL_LABELS[n] for n in nulls) + r" \\",
        r"  \midrule",
    ]
    for i, (src, r) in enumerate(results.items()):
        if i:
            lines.append(r"  \midrule")
        lines += table_rows(src, r, nulls)
    lines += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_rows(src: str, r: dict, nulls=NULLS) -> list[str]:
    """One source's rows of ``table_tex``, one per statistic, with a column per null in ``nulls``."""
    a = _overall(r)["statistics"]
    out = []
    for j, s in enumerate(STATS):
        v = a[s]
        cells = [SOURCE_LABELS.get(src, src) if j == 0 else "", STAT_LABELS[s],
                 f"{_f(v['observed'])} [{_f(v['ci95'][0])}, {_f(v['ci95'][1])}]"]
        for n in nulls:
            d = v["nulls"].get(n)
            if d is None:
                cells.append("--")
            elif d["fixed_by_null"]:
                cells.append("(fixed)")
            else:
                cells.append(f"{_f(d['null_mean'])} ({_tex_p(d['p_two_sided'])})")
        out.append("  " + " & ".join(cells) + r" \\")
    return out


def markdown(rep: dict, results: dict[str, dict]) -> str:
    s = rep["settings"]
    lines = ["# Within-proof area switching", "",
             f"Run `{rep['run']}`. {s['n_perm']} draws per null, {s['n_boot']} bootstrap resamples over proofs, "
             f"seed {s['seed']}. Two-sided permutation p-values; intervals are 95% percentile bootstraps.", ""]
    for src, r in results.items():
        a = _overall(r)
        lines += [f"## {SOURCE_LABELS.get(src, src)}", "",
                  f"{r['n_proofs']} proofs, {r['n_steps']} area-labelled steps, {r['n_areas']} areas"
                  f" ({r['n_dropped_short']} proofs with under two labelled steps dropped). "
                  f"Source: {rep['notes'].get(src, src)}."]
        if "noise" not in r["nulls"]:
            lines.append("No noise rate for this source, so the noise-floor null was skipped.")
        else:
            lines.append(f"Noise floor epsilon = {_f(r['noise_rate'])} ({rep['noise_rate_source'].get(src, '')}).")
        lines += ["", "| statistic | observed [95% CI] | " + " | ".join(
            f"{NULL_LABELS[n]} mean (p)" for n in r["nulls"]) + " |", "|---|---|" + "---|" * len(r["nulls"])]
        for st in STATS:
            v = a["statistics"][st]
            row = [STAT_LABELS[st], f"{_f(v['observed'])} [{_f(v['ci95'][0])}, {_f(v['ci95'][1])}]"]
            for n in r["nulls"]:
                d = v["nulls"][n]
                row.append("fixed by the null" if d["fixed_by_null"] else f"{_f(d['null_mean'])} ({_p(d['p_two_sided'])})")
            lines.append("| " + " | ".join(row) + " |")
        for kind in ("folder", "length"):
            ents = [e for e in r["strata"] if e["kind"] == kind]
            if not ents or "shuffle" not in r["nulls"]:
                continue
            lines += ["", f"Switch rate by {kind} (shuffle null, Holm across {kind}s):", "",
                      f"| {kind} | proofs | observed [95% CI] | shuffle mean | difference [95% CI] | Holm p |",
                      "|---|---|---|---|---|---|"]
            for e in ents:
                v = e["statistics"]["switch_rate"]
                d = v["nulls"]["shuffle"]
                lines.append(f"| {e['stratum']} | {e['n_proofs']} | {_f(v['observed'])} [{_f(v['ci95'][0])}, "
                             f"{_f(v['ci95'][1])}] | {_f(d['null_mean'])} | {_f(d['difference'])} "
                             f"[{_f(d['difference_ci95'][0])}, {_f(d['difference_ci95'][1])}] | {_p(d.get('holm'))} |")
        lines.append("")
    lines += ["## What each null tests", "",
              "- **Shuffle**: each proof's own areas in random order. Keeps composition, so share off main area is "
              "fixed; a switch rate below it means areas come in runs (segments), not flicker.",
              "- **Markov**: first-order chain per Mathlib folder, started at the observed first area. A share of proofs "
              "with a switch below it means switching is concentrated in a minority of proofs.",
              "- **Folder pool**: steps dealt out at random across proofs of one Mathlib folder. Switching far below "
              "it means proofs keep to an area beyond what their folder explains.",
              "- **Noise floor**: a proof that never moves, seen through a labeller that errs with probability "
              "epsilon. Share off main area above it means movement beyond label noise.", ""]
    return "\n".join(lines)


def draw(results: dict[str, dict], dwell: pd.DataFrame):
    """The figure as a matplotlib Figure: statistics against nulls (top), dwell and switch rate by length (bottom)."""
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .figures import paper_rc

    with matplotlib.rc_context(paper_rc()):
        fig = plt.figure(figsize=(7.0, 4.4), layout="constrained")
        top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.05])
        axes = top.subplots(1, len(STATS))
        srcs = list(results)
        pal = dict(zip(["observed", *NULLS], sns.color_palette(n_colors=1 + len(NULLS))))
        seen: set[str] = set()  # label each series once, at its first appearance (macrostates have no noise null)
        for ax, st in zip(axes, STATS):
            for i, src in enumerate(srcs):
                v = _overall(results[src])["statistics"][st]
                ax.errorbar([i], [v["observed"]], yerr=[[v["observed"] - v["ci95"][0]], [v["ci95"][1] - v["observed"]]],
                            fmt="o", color=pal["observed"], ms=5, capsize=2, zorder=3,
                            label="Observed" if i == 0 else None)
                for k, n in enumerate(results[src]["nulls"]):
                    d = v["nulls"][n]
                    ax.plot([i + 0.12 * (k + 1)], [d["null_mean"]], marker="_", ms=10, mew=2, color=pal[n],
                            ls="none", label=NULL_LABELS[n] if n not in seen else None)
                    seen.add(n)
            # one word per line, so two or three sources fit under a quarter-width panel
            ax.set_xticks(range(len(srcs)), [SOURCE_LABELS.get(s, s).replace(" ", "\n") for s in srcs], rotation=0,
                          fontsize="small")
            ax.set_xlim(-0.4, len(srcs) - 0.2)
            ax.set_title(STAT_LABELS[st], fontsize="medium")
            sns.despine(ax=ax)
        handles, labels = [], []
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        top.legend(handles, labels, loc="outside lower center", ncol=len(labels))

        bax = bottom.subplots(1, len(srcs) + 1)
        bax = np.atleast_1d(bax)
        for ax, src in zip(bax, srcs):
            d = dwell[(dwell["source"] == src) & (dwell["dwell"] <= 30) & (dwell["share"] > 0)]
            sns.lineplot(data=d, x="dwell", y="share", hue="series", ax=ax, palette=pal, marker="o", ms=3,
                         hue_order=[h for h in ["observed", *NULLS] if h in set(d["series"])], legend=False)
            ax.set_yscale("log")
            ax.set_xlabel("Dwell (steps)")
            ax.set_ylabel("Share of runs")
            ax.set_title(f"{SOURCE_LABELS.get(src, src)}: dwell", fontsize="medium")
            sns.despine(ax=ax)
        ax = bax[-1]
        rows = []
        for src in srcs:
            for e in results[src]["strata"]:
                if e["kind"] != "length":
                    continue
                v = e["statistics"]["switch_rate"]
                rows.append({"source": SOURCE_LABELS.get(src, src), "length": e["stratum"], "series": "Observed",
                             "rate": v["observed"]})
                if "shuffle" in v["nulls"]:
                    rows.append({"source": SOURCE_LABELS.get(src, src), "length": e["stratum"], "series": "Shuffle",
                                 "rate": v["nulls"]["shuffle"]["null_mean"]})
        if rows:
            df = pd.DataFrame(rows)
            order = list(length_bin(np.asarray(LENGTH_EDGES)))  # 2-3, 4-7, ... whichever source comes first
            df["length"] = pd.Categorical(df["length"], categories=[o for o in order if o in set(df["length"])],
                                          ordered=True)
            df = df.sort_values("length", kind="stable")
            sns.lineplot(data=df, x="length", y="rate", hue="source", style="series", marker="o", ms=3, ax=ax,
                         sort=False)
            ax.legend(fontsize="x-small", frameon=False)
        ax.set_xlabel("Proof length (labelled steps)")
        ax.set_ylabel("Switches per move")
        ax.set_title("Switch rate by length", fontsize="medium")
        sns.despine(ax=ax)
    return fig


def figure(results: dict[str, dict], dwell: pd.DataFrame, path_stem: Path) -> list[Path]:
    """Write ``draw``'s figure to ``<path_stem>.pdf`` and ``.png``."""
    import matplotlib
    import matplotlib.pyplot as plt

    from .figures import paper_rc

    fig = draw(results, dwell)
    written = []
    with matplotlib.rc_context(paper_rc()):  # font families resolve when the figure is drawn
        for suffix in (".pdf", ".png"):
            p = path_stem.with_suffix(suffix)
            fig.savefig(p)
            written.append(p)
    plt.close(fig)
    return written


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# --------------------------------------------------------------------------- run


def run(run_dir: str | Path, out_dir: str | Path | None = None, *, steps_path=None, labels_path=None,
        assignments_path=None, config_path=None, judge: str | None = None, sources: list[str] | None = None,
        nulls: list[str] | None = None, n_perm: int = 1000, n_boot: int = 2000, seed: int = 0,
        noise_rate: float | None = None, markov_group: str = "folder", pooled_within: str = "folder",
        min_proofs: int = 30, same_proofs: bool = False, figures: bool = True, verbose: bool = True) -> dict:
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "switches"
    say = print if verbose else (lambda *a, **k: None)
    nulls = list(nulls or NULLS)
    inp = load_sources(run_dir, steps_path, labels_path, assignments_path, config_path, judge, sources)
    if not inp["areas"]:
        raise FileNotFoundError(f"no area sequences in {run_dir}: need judge/labels.parquet (step labels), "
                                "msm/assignments.parquet, or a synthetic steps table with true_area")
    areas = inp["areas"]
    if same_proofs and len(areas) > 1:
        common = set.intersection(*(set(a["proof_id"].astype(str)) for a in areas.values()))
        areas = {k: a[a["proof_id"].astype(str).isin(common)] for k, a in areas.items()}

    est, est_note = estimate_noise(inp["raw"])
    results, noise_src = {}, {}
    for i, (src, a) in enumerate(areas.items()):
        seq = Sequences(a)
        if seq.P == 0:
            say(f"switches: {src}: no proof has two area-labelled steps; skipped")
            continue
        eps = noise_rate if noise_rate is not None else (est if src == "judge" else None)
        noise_src[src] = "--noise-rate" if noise_rate is not None else (est_note if src == "judge" else "none")
        say(f"switches: {src}: {seq.P} proofs, {seq.N} steps, {seq.K} areas, nulls {nulls}, {n_perm} draws")
        results[src] = analyse_source(seq, _folders(seq, inp["steps"]), nulls, n_perm=n_perm, n_boot=n_boot,
                                   seed=seed + 1000 * i, noise_rate=eps, markov_group=markov_group,
                                   pooled_within=pooled_within, min_proofs=min_proofs)

    out.mkdir(parents=True, exist_ok=True)
    report = {
        "run": str(run_dir),
        "inputs": inp["inputs"],
        "notes": inp["notes"],
        "noise_rate_source": noise_src,
        "settings": {"n_perm": n_perm, "n_boot": n_boot, "seed": seed, "nulls": nulls, "markov_group": markov_group,
                     "pooled_within": pooled_within, "min_proofs": min_proofs, "same_proofs": same_proofs,
                     "length_edges": list(LENGTH_EDGES), "resampling_unit": "proof"},
        "headline": headline(results),
        "sources": {k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for k, r in results.items()},
    }
    (out / "switches.json").write_text(json.dumps(report, indent=2, default=_jsonable))
    long_table(results).to_csv(out / "switches.csv", index=False)
    pd.concat([r["_proofs"].assign(source=k) for k, r in results.items()]).to_csv(out / "proofs.csv", index=False)
    dw = dwell_table(results)
    dw.to_csv(out / "dwell.csv", index=False)
    (out / "table_switches.tex").write_text(table_tex(results))
    (out / "REPORT.md").write_text(markdown(report, results))
    if figures:
        try:
            figure(results, dw, out / "switches")
        except ImportError as e:  # pragma: no cover - figures extra missing
            say(f"switches: no figure ({e}); install the figures extra")
    for src, h in report["headline"].items():
        msg = f"switches: {src}: {h['has_switch']:.1%} of proofs switch, {h['switch_rate']:.3f} switches per move"
        if "switch_rate_shuffle" in h:
            msg += f" (shuffle {h['switch_rate_shuffle']:.3f}, p = {_p(h['p_switch_rate_shuffle'])})"
        say(msg)
    say(f"wrote {out}")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="markov-lean switches",
                                description="Within-proof area switching against length- and composition-"
                                            "preserving nulls, by judge labels and by macrostates.")
    p.add_argument("--run", required=True, help="pilot output directory (holds msm/ and judge/labels.parquet)")
    p.add_argument("--out", default=None, help="output directory (default <run>/switches)")
    p.add_argument("--steps", default=None, help="steps table, for Mathlib folders "
                                                 "(default <run>/judge/sample_steps.parquet)")
    p.add_argument("--config", default=None, help="pilot config, to rebuild the steps table when it is missing")
    p.add_argument("--labels", default=None, help="judge labels (default <run>/judge/labels.parquet)")
    p.add_argument("--assignments", default=None, help="macrostate assignments (default <run>/msm/assignments.parquet)")
    p.add_argument("--judge", default=None, help="use this judge's labels instead of the consensus")
    p.add_argument("--sources", nargs="+", choices=["judge", "macrostate"], default=None)
    p.add_argument("--nulls", nargs="+", choices=list(NULLS), default=None)
    p.add_argument("--n-perm", type=int, default=1000, help="draws per null")
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples over proofs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-rate", type=float, default=None,
                   help="label error rate for the noise-floor null (default: from two judges' agreement)")
    p.add_argument("--markov-group", choices=["folder", "first", "none", "mode"], default="folder",
                   help="fit the Markov null per Mathlib folder (default; one chain when there are no folders), "
                        "per first area, as one chain, or per main area (biased: see the module notes)")
    p.add_argument("--pooled-within", choices=["folder", "all"], default="folder")
    p.add_argument("--min-proofs", type=int, default=30, help="smallest folder reported as its own stratum")
    p.add_argument("--same-proofs", action="store_true", help="restrict every source to proofs present in all")
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    run(a.run, a.out, steps_path=a.steps, labels_path=a.labels, assignments_path=a.assignments,
        config_path=a.config, judge=a.judge, sources=a.sources, nulls=a.nulls, n_perm=a.n_perm, n_boot=a.n_boot,
        seed=a.seed, noise_rate=a.noise_rate, markov_group=a.markov_group, pooled_within=a.pooled_within,
        min_proofs=a.min_proofs, same_proofs=a.same_proofs, figures=not a.no_figures)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
