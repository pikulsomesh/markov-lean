"""Reversible against non-reversible Markov state models of the same proofs.

A Lean proof runs one way: it starts at a theorem statement and ends at
``no goals``.  The main pipeline (``msm.py``) fits a reversible
(detailed-balance) maximum-likelihood MSM and coarse-grains it with PCCA+,
which assumes reversibility.  This module fits, on the *same* microstate
trajectories and lag, three models side by side:

``reversible``
    ``MaximumLikelihoodMSM(reversible=True)`` on the largest strongly
    connected set, coarse-grained with PCCA+ (as ``msm.run``).
``nonreversible``
    ``MaximumLikelihoodMSM(reversible=False)`` on the same set (row-normalized
    counts), coarse-grained with GPCCA (below), which does not need detailed
    balance.  Proof ends are censoring: nothing is counted from the end of one
    proof to the start of the next.
``absorbing``
    The same counts plus an explicit absorbing ``done`` state: every window
    that runs past a proof's last tactic step lands in ``done`` (the proof has
    reached ``no goals``) and ``done`` maps to itself.  No connectivity pruning
    is needed, because every visited microstate reaches ``done``, so the
    theorem-statement states that the strongly connected set drops are kept.
    GPCCA with one extra set (``n_macrostates + 1``) leaves room for the end
    state.

We do not restart trajectories (joining each proof's end to another proof's
start): that invents end-to-start transitions whose statistics depend on how
proofs are ordered, and makes the chain ergodic by construction.

**GPCCA.**  Generalized PCCA+ (Reuter et al. 2018, J. Chem. Phys. 149,
174105; Reuter, Fackeldey & Weber 2019, J. Chem. Phys. 150, 174103) replaces
PCCA+'s eigenvectors by a real Schur basis of the dominant invariant
subspace, which exists for any stochastic matrix.  With a positive weighting
distribution ``eta`` (here the share of steps in each state by default):

1. ``P_eta = D^{1/2} P D^{-1/2}`` with ``D = diag(eta)``;
2. an ordered real Schur form puts the ``m`` eigenvalues of largest modulus
   first (a complex-conjugate pair is never split; ``m`` moves up one when it
   would be);
3. the leading ``m`` Schur vectors, re-based so the first is ``sqrt(eta)``,
   give ``X = D^{-1/2} Z`` with ``X[:, 0] = 1`` and ``X^T D X = I``;
4. the PCCA+ inner simplex algorithm and Roeblitz-Weber optimization turn
   ``X`` into memberships ``chi = X A`` (non-negative, rows sum to 1);
5. ``P_c = (chi^T D chi)^{-1} chi^T D P chi``.

On a reversible matrix with ``eta`` its stationary distribution the Schur
vectors span the same space as the eigenvectors, so GPCCA reproduces PCCA+
(tested).  pyGPCCA is the reference implementation; it is not a dependency of
this repository, so this module carries its own (numpy/scipy only).

**What is compared.**  Implied timescales over lags (``-lag / ln|lambda|``,
moduli for complex eigenvalues), a Chapman-Kolmogorov test computed the same
way for all three fits, the spectral gap and membership crispness, macrostate
occupancy (the share of steps in the largest macrostate), step-level NMI/ARI
between the fits and of each against labels when given, and how
irreversible the data are: a likelihood-ratio test of detailed balance,
held-out log-likelihood per transition (folds over proofs), the share of
counts on one-way edges, entropy production, and the net flux between
macrostates.  For the absorbing view at lag 1: the probability of reaching
each other macrostate before ``no goals``, of finishing without leaving the
current macrostate, and of a proof switching at least once, each next to its
empirical value (a Markov check on the forward question), plus the expected
number of steps left.

Time is counted in proof steps, as in ``msm.py``.  No step makes an API call.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.optimize import fmin
from scipy.stats import chi2

from . import msm as M

FITS = ("reversible", "nonreversible", "absorbing")
FIT_LABEL = {"reversible": "Reversible + PCCA+", "nonreversible": "Non-reversible + GPCCA",
             "absorbing": "Absorbing + GPCCA"}


# ============================================================================ GPCCA


class SplitPairError(ValueError):
    """The requested number of sets would split a complex-conjugate (or equal-modulus) pair."""


def _sorted_eigs(P: np.ndarray) -> np.ndarray:
    ev = np.linalg.eigvals(P)
    return ev[np.argsort(-np.abs(ev), kind="stable")]


def splits_pair(ev_sorted: np.ndarray, m: int, rtol: float = 1e-8) -> bool:
    """True if cutting the modulus-sorted spectrum after ``m`` values separates equal moduli."""
    if m <= 0 or m >= len(ev_sorted):
        return False
    a, b = abs(ev_sorted[m - 1]), abs(ev_sorted[m])
    return bool(np.isclose(a, b, rtol=rtol, atol=1e-12))


def resolve_n_sets(P: np.ndarray, m: int) -> int:
    """``m``, or the next count up that does not split a conjugate pair."""
    ev = _sorted_eigs(P)
    while m < len(ev) and splits_pair(ev, m):
        m += 1
    return m


def schur_basis(P: np.ndarray, m: int, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """eta-orthonormal basis ``X`` (n, m) of the dominant invariant subspace, ``X[:, 0] == 1``.

    Returns ``(X, eigenvalues of the selected block)``.  Raises
    :class:`SplitPairError` when the cut would split equal moduli.
    """
    n = P.shape[0]
    eta = np.asarray(eta, dtype=float)
    if np.any(eta <= 0) or len(eta) != n:
        raise ValueError("eta must be positive with one entry per state")
    eta = eta / eta.sum()
    s = np.sqrt(eta)
    Pt = (s[:, None] * P) / s[None, :]
    ev = _sorted_eigs(Pt)
    if m >= n:
        m, cut = n, -1.0
    else:
        if splits_pair(ev, m):
            raise SplitPairError(f"{m} sets would split eigenvalues of equal modulus "
                                 f"({ev[m - 1]:.4g}, {ev[m]:.4g}); use {resolve_n_sets(P, m)}")
        cut = 0.5 * (abs(ev[m - 1]) + abs(ev[m]))
    _, Q, sdim = scipy.linalg.schur(Pt, output="real", sort=lambda x, y: math.hypot(x, y) > cut)
    if sdim != m:
        raise ValueError(f"ordered Schur form selected {sdim} eigenvalues, expected {m}")
    Qm = Q[:, :m]
    u = s  # unit norm since eta sums to 1
    if np.linalg.norm(u - Qm @ (Qm.T @ u)) > 1e-6:
        raise ValueError("the constant vector is not in the dominant subspace (is P stochastic?)")
    B = Qm - np.outer(u, u @ Qm)
    U = np.linalg.svd(B, full_matrices=False)[0][:, : m - 1]
    Z = np.column_stack([u, U])
    return Z / s[:, None], ev[:m]


def _inner_simplex(X: np.ndarray) -> np.ndarray:
    """PCCA+ inner simplex algorithm (Deuflhard & Weber 2005): the initial ``A = X[ind]^{-1}``."""
    n, m = X.shape
    ind = np.zeros(m, dtype=np.int64)
    ind[0] = int(np.argmax(np.linalg.norm(X, axis=1)))
    ortho = X - X[ind[0]]
    for j in range(1, m):
        t = ortho[ind[j - 1]].copy()
        ortho -= np.outer(ortho @ t, t)
        d = np.linalg.norm(ortho, axis=1)
        d[ind[:j]] = -1.0
        ind[j] = int(np.argmax(d))
        ortho /= np.linalg.norm(ortho[ind[j]])
    return np.linalg.inv(X[ind])


def _fill(crop: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Feasible ``A`` from its lower-right block: rows 2..m sum to 0, row 1 makes chi >= 0, rows of chi sum to 1."""
    A = np.zeros((crop.shape[0] + 1, crop.shape[1] + 1))
    A[1:, 1:] = crop
    A[1:, 0] = -crop.sum(axis=1)
    A[0] = np.max(-(X[:, 1:] @ A[1:]), axis=0)
    return A / A[0].sum()


def _optimize(X: np.ndarray, A0: np.ndarray, maxiter: int | None = None) -> np.ndarray:
    """Roeblitz-Weber crispness objective, maximized by Nelder-Mead over the free block of ``A``."""
    m = X.shape[1]
    if m == 1:
        return np.ones((1, 1))
    shape = (m - 1, m - 1)

    def neg_crisp(v):
        A = _fill(v.reshape(shape), X)
        return -float(np.sum(A ** 2 / A[0][None, :]))

    v = fmin(neg_crisp, A0[1:, 1:].ravel(), disp=False, maxiter=maxiter,
             maxfun=None if maxiter is None else 2 * maxiter)
    return _fill(v.reshape(shape), X)


@dataclass
class GPCCA:
    memberships: np.ndarray        # (n, m), rows sum to 1
    assignments: np.ndarray        # (n,) argmax of memberships
    coarse_T: np.ndarray           # (m, m)
    coarse_eta: np.ndarray         # (m,)
    eigenvalues: np.ndarray        # the m selected eigenvalues (complex)
    n_sets: int
    n_sets_requested: int


def gpcca(P: np.ndarray, m: int, eta: np.ndarray | None = None, adjust: bool = True) -> GPCCA:
    """Generalized PCCA+ of a (possibly non-reversible) stochastic matrix into ``m`` sets.

    ``eta`` defaults to uniform.  With ``adjust`` a count that would split a
    conjugate pair moves up to the next valid one (recorded in ``n_sets``).
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    eta = np.full(n, 1.0 / n) if eta is None else np.asarray(eta, float) / np.sum(eta)
    req = int(m)
    m = resolve_n_sets(P, req) if adjust else req
    m = min(m, n)
    if m == 1:
        chi = np.ones((n, 1))
        ev = _sorted_eigs(P)[:1]
    else:
        X, ev = schur_basis(P, m, eta)
        A = _optimize(X, _inner_simplex(X))
        chi = np.clip(X @ A, 0.0, None)
        chi /= chi.sum(axis=1, keepdims=True)
    D = eta[:, None]
    W = chi.T @ (D * chi)
    Pc = np.linalg.solve(W, chi.T @ (D * (P @ chi)))
    return GPCCA(memberships=chi, assignments=np.argmax(chi, axis=1), coarse_T=Pc,
                 coarse_eta=chi.T @ eta, eigenvalues=ev, n_sets=m, n_sets_requested=req)


def choose_n_sets(P: np.ndarray, max_sets: int = 12, min_sets: int = 2) -> int:
    """Number of sets at the largest gap in eigenvalue modulus that splits no pair."""
    evc = _sorted_eigs(P)
    ev = np.abs(evc)
    best, arg = -1.0, min_sets
    for m in range(min_sets, min(max_sets, len(ev) - 1) + 1):
        if splits_pair(evc, m):
            continue
        gap = ev[m - 1] - ev[m]
        if gap > best:
            best, arg = gap, m
    return arg


# ============================================================================ counts and fits


def dense_counts(dtrajs: list[np.ndarray], lag: int, n: int) -> np.ndarray:
    """Sliding-window counts at ``lag`` within each trajectory."""
    C = np.zeros((n, n))
    for d in dtrajs:
        if len(d) > lag:
            np.add.at(C, (d[:-lag], d[lag:]), 1.0)
    return C


def absorbing_counts(dtrajs: list[np.ndarray], lag: int, n_micro: int) -> np.ndarray:
    """Counts over ``n_micro + 1`` states; index ``n_micro`` is ``done``.

    A window from step ``t`` lands in ``done`` when ``t + lag`` is past the
    last step, so every step contributes exactly one count; ``done -> done``
    gets one count per proof.
    """
    done = n_micro
    C = np.zeros((n_micro + 1, n_micro + 1))
    for d in dtrajs:
        L = len(d)
        if L == 0:
            continue
        if L > lag:
            np.add.at(C, (d[:-lag], d[lag:]), 1.0)
        np.add.at(C[:, done], d[max(L - lag, 0):], 1.0)
        C[done, done] += 1.0
    return C


def visits(dtrajs: list[np.ndarray], n: int) -> np.ndarray:
    return np.bincount(np.concatenate(dtrajs), minlength=n).astype(float) if dtrajs else np.zeros(n)


@dataclass
class Fit:
    name: str
    lag: int
    T: np.ndarray                   # transition matrix on the active states
    states: np.ndarray              # full-space ids of the active states (absorbing: last is ``done``)
    counts: np.ndarray              # counts on the active states
    chi: np.ndarray                 # memberships on the active states
    coarse_T: np.ndarray
    eigenvalues: np.ndarray         # leading eigenvalues by modulus (complex)
    method: str
    n_sets_requested: int
    eta: np.ndarray                 # weights on the active states


def _fit_msm(dtrajs, lag, n_micro, reversible):
    m, _ = M.estimate_msm(dtrajs, lag, n_micro, reversible)
    states = np.asarray(m.count_model.state_symbols, dtype=np.int64)
    C = np.asarray(m.count_model.count_matrix, dtype=float)
    if hasattr(C, "toarray"):
        C = C.toarray()
    return np.asarray(m.transition_matrix, dtype=float), states, C, m


def _fit_absorbing(dtrajs, lag, n_micro):
    C = absorbing_counts(dtrajs, lag, n_micro)
    rows = C.sum(axis=1)
    states = np.flatnonzero(rows > 0)
    Cs = C[np.ix_(states, states)]
    return Cs / Cs.sum(axis=1, keepdims=True), states, Cs


def transition_builder(name: str, dtrajs, n_micro):
    """``lag -> (T, active states)`` for one model family."""
    if name == "absorbing":
        return lambda lag: _fit_absorbing(dtrajs, lag, n_micro)[:2]
    rev = name == "reversible"
    return lambda lag: _fit_msm(dtrajs, lag, n_micro, rev)[:2]


def _eta_for(name, states, dtrajs, n_micro, mode):
    if mode == "uniform":
        return np.full(len(states), 1.0 / len(states))
    v = np.append(visits(dtrajs, n_micro), float(len(dtrajs)))  # done: one per proof
    e = v[states].astype(float)
    if np.any(e <= 0):
        e = e + 1e-12
    return e / e.sum()


def fit_all(dtrajs, n_micro, lag, n_macro, eta_mode="empirical", n_eigs=13,
            fits: tuple[str, ...] = FITS) -> dict[str, Fit]:
    out: dict[str, Fit] = {}
    for name in fits:
        if name == "absorbing":
            T, states, C = _fit_absorbing(dtrajs, lag, n_micro)
            m_req = n_macro + 1
        else:
            T, states, C, msm_model = _fit_msm(dtrajs, lag, n_micro, name == "reversible")
            m_req = n_macro
        eta = _eta_for(name, states, dtrajs, n_micro, eta_mode)
        ev = _sorted_eigs(T)[:n_eigs]
        if name == "reversible":
            pc = msm_model.pcca(min(m_req, msm_model.n_states))
            chi = np.asarray(pc.memberships, dtype=float)
            cT = np.real(np.asarray(pc.coarse_grained_transition_matrix))
            method = "pcca+"
        else:
            g = gpcca(T, min(m_req, len(states)), eta)
            chi, cT, method = g.memberships, g.coarse_T, "gpcca"
        out[name] = Fit(name=name, lag=lag, T=T, states=states, counts=C, chi=chi, coarse_T=cT,
                        eigenvalues=ev, method=method, n_sets_requested=m_req, eta=eta)
    return out


# ============================================================================ checks


def timescales_from_eigs(ev: np.ndarray, lag: int, skip_unit: int = 1) -> np.ndarray:
    a = np.abs(np.asarray(ev))[skip_unit:]
    with np.errstate(divide="ignore"):
        return np.where(a > 0, -lag / np.log(np.clip(a, 1e-300, 1 - 1e-15)), 0.0)


def its_table(name, dtrajs, n_micro, lags, n_its) -> pd.DataFrame:
    build = transition_builder(name, dtrajs, n_micro)
    rows = []
    for lag in lags:
        try:
            T, _ = build(lag)
        except ValueError:
            continue
        ev = _sorted_eigs(T)
        for i, (t, e) in enumerate(zip(timescales_from_eigs(ev, lag)[:n_its], ev[1:n_its + 1])):
            rows.append({"fit": name, "lag": int(lag), "index": i + 1, "timescale": float(t),
                         "eig_real": float(np.real(e)), "eig_imag": float(np.imag(e))})
    return pd.DataFrame(rows, columns=["fit", "lag", "index", "timescale", "eig_real", "eig_imag"])


def ck_table(fit: Fit, dtrajs, n_micro, mlags: int) -> pd.DataFrame:
    """Chapman-Kolmogorov test with the same definition for every fit.

    For each set ``i``, start from ``p0 ∝ eta * chi_i`` and compare
    ``p0 T(lag)^k chi`` with ``p0 T(k lag) chi``.  Matrices at other lags are
    embedded in the full state space; mass that leaves a lag's active set is
    dropped and each row renormalized.
    """
    cols = ["fit", "lag", "from_set", "to_set", "predicted", "estimated"]
    N = n_micro + (1 if fit.name == "absorbing" else 0)
    m = fit.chi.shape[1]
    chi = np.zeros((N, m))
    chi[fit.states] = fit.chi
    eta = np.zeros(N)
    eta[fit.states] = fit.eta

    def embed(T, st):
        E = np.zeros((N, N))
        E[np.ix_(st, st)] = T
        return E

    Tb = embed(fit.T, fit.states)
    build = transition_builder(fit.name, dtrajs, n_micro)
    P0 = (eta[:, None] * chi).T
    P0 = P0 / np.maximum(P0.sum(axis=1, keepdims=True), 1e-300)
    rows, Tk = [], np.eye(N)
    for k in range(1, mlags + 1):
        Tk = Tk @ Tb
        try:
            TL, stL = build(k * fit.lag)
        except ValueError:
            break
        pred, est = P0 @ Tk @ chi, P0 @ embed(TL, stL) @ chi
        pred /= np.maximum(pred.sum(axis=1, keepdims=True), 1e-300)
        est /= np.maximum(est.sum(axis=1, keepdims=True), 1e-300)
        for i in range(m):
            for j in range(m):
                rows.append({"fit": fit.name, "lag": k * fit.lag, "from_set": i, "to_set": j,
                             "predicted": float(pred[i, j]), "estimated": float(est[i, j])})
    return pd.DataFrame(rows, columns=cols)


def micro_map(fit: Fit, n_micro: int, centers: np.ndarray | None) -> np.ndarray:
    """Macrostate of every microstate (absorbing: ``done`` excluded); nearest kept center or -1."""
    keep = fit.states < n_micro
    st = fit.states[keep]
    asg = np.argmax(fit.chi[keep], axis=1)
    mp = np.full(n_micro, -1, dtype=np.int64)
    mp[st] = asg
    dropped = np.setdiff1d(np.arange(n_micro), st)
    if centers is not None and len(dropped) and len(st):
        d2 = ((centers[dropped, None, :] - centers[None, st, :]) ** 2).sum(-1)
        mp[dropped] = mp[st[np.argmin(d2, axis=1)]]
    return mp


def irreversibility(fit_nr: Fit, fit_rv: Fit) -> dict:
    """How far microstate counts are from detailed balance, on the non-reversible fit's active set.

    Microstate counts are sparse, so sampling noise alone makes them
    asymmetric and inflates the likelihood ratio and entropy production (a
    reversible synthetic chain scores high too); read these next to
    :func:`macro_irreversibility`, which tests the well-sampled coarse chain.
    """
    C, T = fit_nr.counts, fit_nr.T
    off = ~np.eye(len(C), dtype=bool)
    tot = C[off].sum()
    asym = float(0.5 * np.abs(C - C.T)[off].sum() / tot) if tot else float("nan")
    one_way = float(C[off & (C > 0) & (C.T == 0)].sum() / tot) if tot else float("nan")
    # stationary distribution of the non-reversible fit
    w, V = np.linalg.eig(T.T)
    pi = np.abs(np.real(V[:, np.argmin(np.abs(w - 1))]))
    pi /= pi.sum()
    F = pi[:, None] * T
    both = off & (F > 0) & (F.T > 0)
    ep = float(0.5 * np.sum((F - F.T)[both] * np.log(F[both] / F.T[both])))
    # likelihood ratio of detailed balance on the same counts (both fits use the same active set)
    same = np.array_equal(fit_nr.states, fit_rv.states)
    lr = df = p = float("nan")
    if same:
        Cr, Tr = fit_rv.counts, fit_rv.T
        nz = C > 0
        ll_nr = float(np.sum(C[nz] * np.log(T[nz])))
        ll_rv = float(np.sum(Cr[nz] * np.log(np.clip(Tr[nz], 1e-300, None))))
        lr = 2.0 * (ll_nr - ll_rv)
        k_nr = int(nz.sum()) - len(C)
        und = np.triu(nz | nz.T)
        k_rv = int(und.sum()) - 1
        df = max(k_nr - k_rv, 1)
        p = float(chi2.sf(lr, df))
    return {"count_asymmetry": asym, "one_way_count_share": one_way, "entropy_production_per_lag": ep,
            "lr_detailed_balance": lr, "lr_df": df, "lr_p_value": p}


def cv_loglik(dtrajs, n_micro, lag, folds=3, seed=0, prior: float = 1.0) -> dict:
    """Held-out log-likelihood per transition, reversible vs non-reversible, folds over proofs.

    Both models are fit on the training proofs' largest strongly connected set
    with the same Dirichlet-style prior, ``prior / n`` pseudo-counts added to
    every entry (``prior`` per row), so neither gets probability zero on a
    transition the training folds never saw.  Held-out transitions with an end
    outside that set are skipped; the share scored is reported.
    """
    from deeptime.markov.msm import MaximumLikelihoodMSM

    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(dtrajs)) % folds
    out = {"reversible": [], "nonreversible": []}
    covered = []
    for f in range(folds):
        tr = [d for d, k in zip(dtrajs, fold) if k != f]
        te = [d for d, k in zip(dtrajs, fold) if k == f]
        try:
            _, st, C, _ = _fit_msm(tr, lag, n_micro, False)
        except ValueError:
            continue
        Cp = C + prior / len(st)
        T_nr = Cp / Cp.sum(axis=1, keepdims=True)
        T_rv = np.asarray(MaximumLikelihoodMSM(reversible=True).fit(Cp).fetch_model().transition_matrix)
        Cte = dense_counts(te, lag, n_micro)
        sub = Cte[np.ix_(st, st)]
        n = max(sub.sum(), 1)
        out["nonreversible"].append(float(np.sum(sub * np.log(T_nr)) / n))
        out["reversible"].append(float(np.sum(sub * np.log(np.clip(T_rv, 1e-300, None))) / n))
        covered.append(float(sub.sum() / max(Cte.sum(), 1)))
    res = {k: float(np.nanmean(v)) if v else float("nan") for k, v in out.items()}
    res["nonrev_minus_rev"] = res["nonreversible"] - res["reversible"]
    res["per_fold"] = out
    res["prior_per_row"] = prior
    res["share_heldout_transitions_scored"] = float(np.mean(covered)) if covered else float("nan")
    return res


def _macro_counts(seqs, k, lag) -> np.ndarray:
    N = np.zeros((k, k))
    for g in seqs:
        if len(g) > lag:
            a, b = g[:-lag], g[lag:]
            ok = (a >= 0) & (b >= 0)
            np.add.at(N, (a[ok], b[ok]), 1.0)
    return N


def _lr_detailed_balance(N: np.ndarray) -> dict:
    """Likelihood ratio of the non-reversible against the reversible MLE on one count matrix."""
    from deeptime.markov.msm import MaximumLikelihoodMSM

    keep = (N.sum(axis=1) > 0) & (N.sum(axis=0) > 0)
    N = N[np.ix_(keep, keep)]
    T_nr = N / N.sum(axis=1, keepdims=True)
    T_rv = np.asarray(MaximumLikelihoodMSM(reversible=True).fit(N).fetch_model().transition_matrix)
    nz = N > 0
    lr = 2.0 * float(np.sum(N[nz] * (np.log(T_nr[nz]) - np.log(np.clip(T_rv[nz], 1e-300, None)))))
    df = max(int(nz.sum()) - len(N) - (int(np.triu(nz | nz.T).sum()) - 1), 1)
    w, V = np.linalg.eig(T_nr.T)
    pi = np.abs(np.real(V[:, np.argmin(np.abs(w - 1))]))
    pi /= pi.sum()
    F = pi[:, None] * T_nr
    both = ~np.eye(len(N), dtype=bool) & (F > 0) & (F.T > 0)
    ep = float(0.5 * np.sum((F - F.T)[both] * np.log(F[both] / F.T[both])))
    return {"lr": max(lr, 0.0), "df": df, "p_value": float(chi2.sf(max(lr, 0.0), df)),
            "entropy_production_per_lag": ep, "n_transitions": float(N.sum())}


def macro_irreversibility(seqs, k, lag, folds=3, seed=0, prior=1.0) -> dict:
    """Detailed balance tested on the macrostate chain, where counts are dense.

    Likelihood ratio (chi-square reference), entropy production of the
    non-reversible coarse chain, and held-out log-likelihood per transition
    (folds over proofs, ``prior / k`` pseudo-counts per entry for both models).
    """
    from deeptime.markov.msm import MaximumLikelihoodMSM

    out = _lr_detailed_balance(_macro_counts(seqs, k, lag))
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(seqs)) % folds
    diffs = []
    for f in range(folds):
        Ntr = _macro_counts([g for g, j in zip(seqs, fold) if j != f], k, lag) + prior / k
        Nte = _macro_counts([g for g, j in zip(seqs, fold) if j == f], k, lag)
        T_nr = Ntr / Ntr.sum(axis=1, keepdims=True)
        T_rv = np.asarray(MaximumLikelihoodMSM(reversible=True).fit(Ntr).fetch_model().transition_matrix)
        n = max(Nte.sum(), 1)
        diffs.append(float(np.sum(Nte * (np.log(T_nr) - np.log(np.clip(T_rv, 1e-300, None)))) / n))
    out["cv_nonrev_minus_rev_per_transition"] = float(np.mean(diffs))
    out["cv_per_fold"] = diffs
    return out


def macro_sequences(dtrajs, mp) -> list[np.ndarray]:
    return [mp[d] for d in dtrajs]


def net_flux(seqs, n_macro) -> np.ndarray:
    """``(n_AB - n_BA) / (n_AB + n_BA)`` of lag-1 macrostate changes; NaN where neither occurs."""
    n = np.zeros((n_macro, n_macro))
    for g in seqs:
        a, b = g[:-1], g[1:]
        ok = (a >= 0) & (b >= 0) & (a != b)
        np.add.at(n, (a[ok], b[ok]), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (n - n.T) / (n + n.T)


def absorbing_analysis(dtrajs, n_micro, mp, n_macro) -> dict:
    """Forward questions on the lag-1 absorbing chain, each next to its empirical value.

    * ``hit[A][B]``: probability, from a step in macrostate ``A``, of reaching
      ``B`` before ``no goals``;
    * ``stay[A]``: probability of finishing without leaving ``A``;
    * ``switch``: probability a proof changes macrostate at least once (from
      its first step);
    * ``steps_left[A]``: expected steps until ``no goals``, the current one included.
    """
    C = absorbing_counts(dtrajs, 1, n_micro)
    vis = np.flatnonzero(C[:n_micro].sum(axis=1) > 0)
    tot = C[vis].sum(axis=1)
    Q = C[np.ix_(vis, vis)] / tot[:, None]
    r = C[vis, n_micro] / tot
    w = visits(dtrajs, n_micro)[vis]
    g = mp[vis]
    I = np.eye(len(vis))
    steps_left = np.linalg.solve(I - Q, np.ones(len(vis)))
    H = np.full((n_macro, n_macro), np.nan)
    for B in range(n_macro):
        inB = g == B
        if not inB.any():
            continue
        S = ~inB
        h = np.ones(len(vis))
        if S.any():
            h[S] = np.linalg.solve(I[np.ix_(S, S)] - Q[np.ix_(S, S)], Q[np.ix_(S, inB)].sum(axis=1))
        for A in range(n_macro):
            inA = g == A
            if A != B and inA.any():
                H[A, B] = float(np.sum(w[inA] * h[inA]) / w[inA].sum())
    stay_micro = np.full(len(vis), np.nan)
    stay = np.full(n_macro, np.nan)
    for A in range(n_macro):
        inA = g == A
        if not inA.any():
            continue
        s = np.linalg.solve(I[np.ix_(inA, inA)] - Q[np.ix_(inA, inA)], r[inA])
        stay_micro[inA] = s
        stay[A] = float(np.sum(w[inA] * s) / w[inA].sum())
    sl = np.array([float(np.sum(w[g == A] * steps_left[g == A]) / w[g == A].sum()) if (g == A).any()
                   else np.nan for A in range(n_macro)])
    pos = {int(x): i for i, x in enumerate(vis)}
    starts = np.array([pos[int(d[0])] for d in dtrajs if len(d)])
    sw_model = float(np.nanmean(1.0 - stay_micro[starts])) if len(starts) else float("nan")

    # empirical counterparts
    He_n = np.zeros((n_macro, n_macro))
    st_n = np.zeros(n_macro)
    a_n = np.zeros(n_macro)
    sl_e = np.zeros(n_macro)
    switched = 0
    for d in dtrajs:
        seq = mp[d]
        L = len(seq)
        if L == 0:
            continue
        valid = seq[seq >= 0]
        switched += int(len(np.unique(valid)) > 1)
        last = {int(B): int(np.flatnonzero(seq == B)[-1]) for B in np.unique(valid)}
        # first index after t holding a macrostate other than seq[t]
        for t in range(L):
            A = int(seq[t])
            if A < 0:
                continue
            a_n[A] += 1
            sl_e[A] += L - t
            fut = seq[t + 1:]
            st_n[A] += bool(np.all(fut == A))
            for B, lb in last.items():
                if B != A and lb > t:
                    He_n[A, B] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        He = He_n / a_n[:, None]
        st_e = st_n / a_n
        sl_e = sl_e / a_n
    np.fill_diagonal(He, np.nan)
    diff = np.abs(H - He)
    return {
        "hit_model": H.tolist(), "hit_empirical": He.tolist(),
        "stay_model": stay.tolist(), "stay_empirical": st_e.tolist(),
        "steps_left_model": sl.tolist(), "steps_left_empirical": sl_e.tolist(),
        "switch_model": sw_model, "switch_empirical": float(switched / max(len(dtrajs), 1)),
        "max_abs_hit_error": float(np.nanmax(diff)) if np.isfinite(diff).any() else float("nan"),
        "max_abs_stay_error": float(np.nanmax(np.abs(stay - st_e))),
        "mean_steps_to_done_model": float(np.sum(w * steps_left) / w.sum()),
    }


# ============================================================================ comparison


@dataclass
class NonrevConfig:
    eta: str = "empirical"          # GPCCA weights: "empirical" (share of steps) or "uniform"
    its_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    n_its: int = 10
    ck_mlags: int = 4
    ck_tol: float = 0.10
    min_gap_ratio: float = 1.5
    min_membership: float = 0.6
    absorbing_tol: float = 0.10     # max |model - empirical| hitting/stay probability
    cv_folds: int = 3
    seed: int = 0


@dataclass
class NonrevResult:
    report: dict
    timescales: pd.DataFrame
    ck: pd.DataFrame
    assignments: pd.DataFrame       # proof_id, step, microstate, macro_<fit>
    occupancy: pd.DataFrame
    fits: dict[str, Fit]

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.timescales.to_csv(out / "implied_timescales.csv", index=False)
        self.ck.to_csv(out / "ck_test.csv", index=False)
        self.occupancy.to_csv(out / "occupancy.csv", index=False)
        self.assignments.to_parquet(out / "assignments.parquet", index=False)
        for name, f in self.fits.items():
            np.save(out / f"coarse_T_{name}.npy", f.coarse_T)
        (out / "nonrev_report.json").write_text(json.dumps(self.report, indent=2, default=M._jsonable))
        return out


def _gap_ratio(ts: np.ndarray, n_slow: int) -> float:
    return float(ts[n_slow - 1] / ts[n_slow]) if len(ts) > n_slow and ts[n_slow] > 0 else float("nan")


def _scores(a, b) -> dict:
    from .robustness import codes, scores

    ca, na = codes(a)
    cb, nb = codes(b)
    return scores(ca, na, cb, nb)


def compare_dtrajs(dtrajs: list[np.ndarray], n_micro: int, lag: int, n_macro: int, *,
                   keys: pd.DataFrame | None = None, labels: np.ndarray | None = None,
                   centers: np.ndarray | None = None, config: NonrevConfig | None = None,
                   meta: dict | None = None) -> NonrevResult:
    """Every comparison on microstate trajectories (one int array per proof).

    ``keys`` (proof_id, step per concatenated row) and ``labels`` (one per
    row, NaN where unlabelled) are optional; ``centers`` places microstates
    outside a fit's active set on the nearest kept one.
    """
    cfg = config or NonrevConfig()
    dtrajs = [np.asarray(d, dtype=np.int64) for d in dtrajs]
    fits = fit_all(dtrajs, n_micro, lag, n_macro, cfg.eta, n_eigs=max(cfg.n_its, n_macro + 2) + 1)
    lags = sorted(set(cfg.its_lags) | {lag})
    its = pd.concat([its_table(n, dtrajs, n_micro, lags, cfg.n_its) for n in FITS], ignore_index=True)
    ck = pd.concat([ck_table(fits[n], dtrajs, n_micro, cfg.ck_mlags) for n in FITS], ignore_index=True)
    micro = np.concatenate(dtrajs)
    maps = {n: micro_map(fits[n], n_micro, centers) for n in FITS}
    step_macro = {n: maps[n][micro] for n in FITS}
    if keys is None:
        keys = pd.DataFrame({"proof_id": np.repeat(np.arange(len(dtrajs)), [len(d) for d in dtrajs]).astype(str),
                             "step": np.concatenate([np.arange(len(d)) for d in dtrajs])})
    asg = keys[["proof_id", "step"]].reset_index(drop=True).copy()
    asg["microstate"] = micro
    for n in FITS:
        asg[f"macro_{n}"] = step_macro[n]

    per_fit, occ_rows = {}, []
    for n in FITS:
        f = fits[n]
        n_sets = f.chi.shape[1]
        ts = timescales_from_eigs(f.eigenvalues, lag)
        n_slow = n_sets - 1
        ckf = ck[ck["fit"] == n]
        ck_err = float((ckf["predicted"] - ckf["estimated"]).abs().max()) if len(ckf) else float("nan")
        member = np.max(f.chi, axis=1)
        w = f.eta
        sm = step_macro[n]
        sizes = np.bincount(sm[sm >= 0], minlength=n_sets)
        share = sizes / max(sizes.sum(), 1)
        for i, s in enumerate(share):
            occ_rows.append({"fit": n, "macrostate": i, "share_of_steps": float(s)})
        slow_ev = f.eigenvalues[1:n_sets]
        rel = M.its_relative_change(its[its["fit"] == n][["lag", "index", "timescale"]], max(n_slow, 1))
        gap = _gap_ratio(ts, n_slow)
        per_fit[n] = {
            "method": f.method,
            "n_states_active": int(len(f.states)),
            "n_sets": int(n_sets),
            "n_sets_requested": int(f.n_sets_requested),
            "timescales_steps": [float(t) for t in ts[: cfg.n_its]],
            "complex_slow_eigenvalues": int(np.sum(np.abs(np.imag(slow_ev)) > 1e-9)),
            "largest_macrostate_share": float(share.max()) if len(share) else float("nan"),
            "macrostate_shares": [float(s) for s in share],
            "coarse_T": np.real(f.coarse_T).tolist(),
            "checks": {
                "spectral_gap": M._check(gap >= cfg.min_gap_ratio, gap, cfg.min_gap_ratio,
                                         f"timescale {n_slow} / timescale {n_slow + 1}"),
                "crisp": M._check(float(np.sum(w * member)) >= cfg.min_membership, float(np.sum(w * member)),
                                  cfg.min_membership, "step-weighted mean of each state's largest membership"),
                "chapman_kolmogorov": M._check(bool(np.isfinite(ck_err) and ck_err <= cfg.ck_tol), ck_err,
                                               cfg.ck_tol, f"max |predicted - estimated| up to {cfg.ck_mlags} x lag"),
                "its_converged": M._check(bool(lag in rel.index and rel.loc[lag] < 0.15),
                                          float(rel.loc[lag]) if lag in rel.index else float("nan"), 0.15,
                                          f"max relative change of the {max(n_slow, 1)} slowest timescales to the next lag"),
            },
        }
        if n == "absorbing":
            done_set = int(np.argmax(f.chi[-1]))
            per_fit[n]["done_set"] = done_set
            per_fit[n]["done_set_share_of_steps"] = float(share[done_set]) if done_set < len(share) else 0.0
    occupancy = pd.DataFrame(occ_rows)

    pairs = {}
    for i, a in enumerate(FITS):
        for b in FITS[i + 1:]:
            ok = (step_macro[a] >= 0) & (step_macro[b] >= 0)
            pairs[f"{a}_vs_{b}"] = _scores(step_macro[a][ok], step_macro[b][ok])
    # method effect alone: GPCCA on the reversible matrix against PCCA+
    rv = fits["reversible"]
    g_rv = gpcca(rv.T, rv.chi.shape[1], rv.eta)
    mp_g = np.full(n_micro, -1, dtype=np.int64)
    mp_g[rv.states] = g_rv.assignments
    sg = mp_g[micro]
    ok = (sg >= 0) & (step_macro["reversible"] >= 0)
    pairs["pcca_vs_gpcca_on_reversible"] = _scores(step_macro["reversible"][ok], sg[ok])

    vs_labels = None
    if labels is not None:
        lab = pd.Series(labels).reset_index(drop=True)
        vs_labels = {}
        for n in FITS:
            s = pd.Series(step_macro[n]).where(step_macro[n] >= 0)
            okl = lab.notna() & s.notna()
            vs_labels[n] = _scores(s[okl].astype(int).to_numpy(), lab[okl].astype(str).to_numpy()) \
                if okl.any() else None

    nmacro_nr = fits["nonreversible"].chi.shape[1]
    seqs = macro_sequences(dtrajs, maps["nonreversible"])
    absorb = {n: absorbing_analysis(dtrajs, n_micro, maps[n], fits[n].chi.shape[1]) for n in FITS}
    irr = {"microstate": irreversibility(fits["nonreversible"], fits["reversible"])}
    for n in ("reversible", "nonreversible"):
        k_n = fits[n].chi.shape[1]
        sq = macro_sequences(dtrajs, maps[n])
        irr[f"macrostate_{n}"] = macro_irreversibility(sq, k_n, lag, cfg.cv_folds, cfg.seed)
        irr[f"macrostate_{n}"]["net_flux"] = net_flux(sq, k_n).tolist()
    report = {
        "time_unit": M.TIME_UNIT,
        "config": asdict(cfg),
        "input": meta or {},
        "n_proofs": int(len(dtrajs)),
        "n_steps": int(len(micro)),
        "n_microstates": int(n_micro),
        "lag": int(lag),
        "n_macrostates": int(n_macro),
        "n_sets_nonreversible_spectral_gap": int(choose_n_sets(fits["nonreversible"].T)),
        "fits": per_fit,
        "agreement": pairs,
        "vs_labels": vs_labels,
        "irreversibility": irr,
        "cv_loglik_per_transition_microstates": cv_loglik(dtrajs, n_micro, lag, cfg.cv_folds, cfg.seed),
        "absorbing_view": absorb,
        "checks": {
            f"absorbing_markov_{n}": M._check(
                bool(absorb[n]["max_abs_hit_error"] <= cfg.absorbing_tol
                     and absorb[n]["max_abs_stay_error"] <= cfg.absorbing_tol),
                max(absorb[n]["max_abs_hit_error"], absorb[n]["max_abs_stay_error"]), cfg.absorbing_tol,
                "max |model - empirical| probability of reaching another macrostate (or none) before no goals")
            for n in FITS
        },
    }
    return NonrevResult(report=report, timescales=its, ck=ck, assignments=asg, occupancy=occupancy, fits=fits)


def dtrajs_from_assignments(asg: pd.DataFrame) -> tuple[list[np.ndarray], pd.DataFrame]:
    a = asg.sort_values(["proof_id", "step"], kind="stable").reset_index(drop=True)
    return M.split_trajectories(a, a["microstate"].to_numpy()), a[["proof_id", "step"]]


def compare_saved(msm_dir: str | Path, config: NonrevConfig | None = None, n_macrostates: int | None = None,
                  lag: int | None = None, labels: pd.Series | None = None) -> NonrevResult:
    """Run the comparison on a saved ``msm.run`` output (assignments.parquet, msm_report.json, centers.npy)."""
    d = Path(msm_dir)
    asg = pd.read_parquet(d / "assignments.parquet")
    asg["proof_id"] = asg["proof_id"].astype(str)
    rep = json.loads((d / "msm_report.json").read_text())
    dtrajs, keys = dtrajs_from_assignments(asg)
    centers = np.load(d / "centers.npy") if (d / "centers.npy").exists() else None
    n_micro = int(rep["n_microstates"])
    cfg = config or NonrevConfig()
    mcfg = rep.get("config", {})
    cfg.its_lags = list(mcfg.get("its_lags", cfg.its_lags))
    cfg.ck_mlags = int(mcfg.get("ck_mlags", cfg.ck_mlags))
    cfg.n_its = int(mcfg.get("n_its", cfg.n_its))
    cfg.seed = int(mcfg.get("seed", cfg.seed))
    lab = None if labels is None else labels.reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
    return compare_dtrajs(dtrajs, n_micro, int(lag or rep["lag"]), int(n_macrostates or rep["n_macrostates"]),
                          keys=keys, labels=lab, centers=centers, config=cfg,
                          meta={"msm_dir": str(d), "msm_lag": rep["lag"], "msm_n_macrostates": rep["n_macrostates"]})


def compare_steps(steps: pd.DataFrame, msm_config: M.MSMConfig | None = None, config: NonrevConfig | None = None,
                  label_col: str | None = None, labels: pd.Series | None = None) -> tuple[NonrevResult, M.MSMResult]:
    """Fit the usual pipeline on ``steps`` (microstates, lag, number of macrostates), then compare."""
    mcfg = msm_config or M.MSMConfig()
    res = M.run(steps, mcfg)
    a = res.assignments
    dtrajs, keys = dtrajs_from_assignments(a)
    cfg = config or NonrevConfig(its_lags=list(mcfg.its_lags), ck_mlags=mcfg.ck_mlags, n_its=mcfg.n_its,
                                 ck_tol=mcfg.ck_tol, min_gap_ratio=mcfg.min_gap_ratio,
                                 min_membership=mcfg.min_membership, seed=mcfg.seed)
    lab = None
    if label_col is not None and label_col in steps.columns:
        s = steps.astype({"proof_id": str}).set_index(["proof_id", "step"])[label_col]
        lab = s.reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
    elif labels is not None:
        lab = labels.reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
    out = compare_dtrajs(dtrajs, res.report["n_microstates"], res.lag, res.report["n_macrostates"],
                         keys=keys, labels=lab, centers=res.centers, config=cfg,
                         meta={"msm_lag": res.lag, "msm_n_macrostates": res.report["n_macrostates"],
                               "msm_checks_failed": [k for k, c in res.checks.items() if not c["passed"]]})
    return out, res


# ============================================================================ synthetic directed proofs


def directed_area_matrix(n_areas: int, p_stay: float, forward: float) -> np.ndarray:
    """Cyclic area chain: stay, else step to the next area with probability ``forward``, back otherwise.

    ``forward = 0.5`` is symmetric, hence reversible; ``forward = 1`` only moves forward.
    """
    T = np.zeros((n_areas, n_areas))
    for i in range(n_areas):
        T[i, i] = p_stay
        T[i, (i + 1) % n_areas] += (1 - p_stay) * forward
        T[i, (i - 1) % n_areas] += (1 - p_stay) * (1 - forward)
    return T


def generate_directed(n_proofs: int = 300, forward: float = 1.0, p_stay: float = 0.8, min_steps: int = 8,
                      max_steps: int = 24, areas: list[str] | None = None, noise: float = 0.35,
                      leak: float = 0.05, carryover: float = 0.2, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic proofs over a directed (``forward`` > 0.5: non-reversible) area chain.

    Same goal-state text model as ``synthetic.generate``; proofs start in a
    uniformly random area (the chain's stationary distribution).
    """
    from . import synthetic as S

    areas = list(areas or S.AREA_VOCAB)
    A = len(areas)
    T = directed_area_matrix(A, p_stay, forward)
    rng = np.random.default_rng(seed)
    vocab = {a: S._area_tokens(a) for a in areas}
    rows = []
    for p in range(n_proofs):
        n_steps = int(rng.integers(min_steps, max_steps + 1))
        a = int(rng.integers(A))
        folder = S.AREA_VOCAB[areas[a]]["folder"][0]
        theorem = f"Directed.{folder}.thm_{p}"
        prev: list[str] = []
        for s in range(n_steps):
            n_tok = int(rng.integers(8, 21))
            n_keep = min(len(prev), int(round(carryover * n_tok)))
            toks = list(rng.choice(prev, size=n_keep, replace=False)) if n_keep else []
            for r in rng.random(n_tok - n_keep):
                if r < noise:
                    toks.append(S.SHARED_NOISE[rng.integers(len(S.SHARED_NOISE))])
                elif r < noise + leak:
                    o = areas[int(rng.integers(A))]
                    toks.append(vocab[o][rng.integers(len(vocab[o]))])
                else:
                    toks.append(vocab[areas[a]][rng.integers(len(vocab[areas[a]]))])
            rng.shuffle(toks)
            rows.append({"proof_id": f"directed:{theorem}", "step": s, "source": "synthetic", "split": "train",
                         "theorem": theorem, "file_path": f"Mathlib/{folder}/Directed.lean", "tactic": "simp",
                         "state_before": S._format_goal(toks, rng), "state_after": "",
                         "true_area": areas[a]})
            prev = toks
            a = int(rng.choice(A, p=T[a]))
        rows[-1]["state_after"] = "no goals"
    steps = pd.DataFrame(rows)
    steps["step"] = steps["step"].astype("int64")
    return steps, T


# ============================================================================ figures


def figures(result: NonrevResult, out_dir: str | Path) -> list[Path]:
    """``nonrev_comparison.pdf`` (timescales, CK, occupancy, agreement) and ``nonrev_absorbing.pdf``."""
    from .figures import BLUE_RAMP, CATEGORICAL, TEXT_WIDTH, _blue_cmap, paper_rc, save_pdf  # noqa: F401
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    out = Path(out_dir)
    color = dict(zip(FITS, CATEGORICAL[:3]))
    rep = result.report
    paths = []
    with matplotlib.rc_context(paper_rc()):
        fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 4.6), layout="constrained")
        ax = axes[0, 0]
        its = result.timescales
        for n in FITS:
            k = rep["fits"][n]["n_sets"] - 1
            sub = its[(its["fit"] == n) & (its["index"] <= max(k, 1))]
            for i, g in sub.groupby("index"):
                ax.plot(g["lag"], g["timescale"], color=color[n], lw=1.2 if i == 1 else 0.6,
                        alpha=1.0 if i == 1 else 0.45, marker="o" if i == 1 else None, ms=2.5,
                        label=FIT_LABEL[n] if i == 1 else None)
        lg = np.array(sorted(its["lag"].unique()))
        ax.fill_between(lg, 0, lg, color="#e5e5e5", lw=0)
        ax.set_yscale("log")
        ax.set_xlabel("Lag (proof steps)")
        ax.set_ylabel("Implied timescale (steps)")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=6, handlelength=1.2,
                  columnspacing=0.8, borderaxespad=0.2)

        ax = axes[0, 1]
        ck = result.ck
        for n in FITS:
            s = ck[ck["fit"] == n]
            ax.scatter(s["estimated"], s["predicted"], s=4, color=color[n], alpha=0.6, lw=0,
                       label=f"{FIT_LABEL[n]} (max err {rep['fits'][n]['checks']['chapman_kolmogorov']['value']:.3f})")
        ax.plot([0, 1], [0, 1], color="#666666", lw=0.6)
        ax.set_xlabel("Estimated set probability")
        ax.set_ylabel("Predicted (CK)")
        ax.legend(loc="upper left", fontsize=6)

        ax = axes[1, 0]
        occ = result.occupancy.copy()
        occ["rank"] = occ.groupby("fit")["share_of_steps"].rank(ascending=False, method="first").astype(int)
        occ["fit_label"] = occ["fit"].map(FIT_LABEL)
        sns.barplot(data=occ, x="rank", y="share_of_steps", hue="fit_label", ax=ax,
                    palette=[color[n] for n in FITS], hue_order=[FIT_LABEL[n] for n in FITS])
        ax.set_xlabel("Macrostate (by size)")
        ax.set_ylabel("Share of steps")
        ax.legend(fontsize=6, title=None, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
                  handlelength=1.0, columnspacing=0.8, borderaxespad=0.2)

        ax = axes[1, 1]
        a = result.assignments
        ok = (a["macro_reversible"] >= 0) & (a["macro_nonreversible"] >= 0)
        ct = pd.crosstab(a.loc[ok, "macro_reversible"], a.loc[ok, "macro_nonreversible"], normalize="index")
        sns.heatmap(ct, ax=ax, cmap=_blue_cmap(), vmin=0, vmax=1, cbar_kws={"label": "Share of row"},
                    linewidths=0.3, linecolor="white")
        ax.set_xlabel("Non-reversible macrostate")
        ax.set_ylabel("Reversible macrostate")
        ar = rep["agreement"]["reversible_vs_nonreversible"]
        fig.nmi = ar["nmi"]
        fig.ari = ar["ari"]
        paths.append(save_pdf(fig, out / "nonrev_comparison.pdf"))
        plt.close(fig)

        ab = rep["absorbing_view"]["nonreversible"]
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6), layout="constrained")
        Hs = [np.array(ab[k], dtype=float) for k in ("hit_model", "hit_empirical")]
        vmax = max([float(np.nanmax(h)) for h in Hs if np.isfinite(h).any()] + [1e-6])
        for ax, H, title in [(axes[0], Hs[0], "Model"), (axes[1], Hs[1], "Empirical")]:
            sns.heatmap(H, ax=ax, cmap=_blue_cmap(), vmin=0, vmax=vmax,
                        cbar_kws={"label": "P(reach column before no goals)"}, linewidths=0.3, linecolor="white",
                        mask=~np.isfinite(H))
            ax.set_xlabel("To macrostate")
            ax.set_ylabel("From macrostate")
            ax.text(0.0, 1.02, title, transform=ax.transAxes, fontsize=7)
        paths.append(save_pdf(fig, out / "nonrev_absorbing.pdf"))
        plt.close(fig)
    return paths


# ============================================================================ notes and CLI


def summary_md(result: NonrevResult) -> str:
    r = result.report
    lines = [f"# Reversible vs non-reversible fit", "",
             f"{r['n_proofs']:,} proofs, {r['n_steps']:,} steps, {r['n_microstates']} microstates, "
             f"lag {r['lag']} (proof steps), {r['n_macrostates']} macrostates "
             f"(absorbing fit: {r['n_macrostates']} + 1 for the end state).", "",
             "| fit | sets | slowest timescale | gap ratio | CK max error | largest macrostate | crisp | failed |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for n in FITS:
        f = r["fits"][n]
        c = f["checks"]
        failed = ", ".join(k for k, v in c.items() if not v["passed"]) or "none"
        lines.append(f"| {FIT_LABEL[n]} | {f['n_sets']} | {f['timescales_steps'][0]:.1f} | "
                     f"{c['spectral_gap']['value']:.2f} | {c['chapman_kolmogorov']['value']:.3f} | "
                     f"{f['largest_macrostate_share']:.1%} | {c['crisp']['value']:.2f} | {failed} |")
    ag = r["agreement"]
    irr = r["irreversibility"]["macrostate_nonreversible"]
    mi = r["irreversibility"]["microstate"]
    cv = r["cv_loglik_per_transition_microstates"]
    lines += ["", "Agreement (step-level NMI / ARI): " + "; ".join(
        f"{k.replace('_vs_', ' vs ')} {v['nmi']:.3f} / {v['ari']:.3f}" for k, v in ag.items()), "",
        f"Detailed balance on the non-reversible macrostate chain: LR {irr['lr']:.1f} on {irr['df']} df "
        f"(p = {irr['p_value']:.3g}), entropy production {irr['entropy_production_per_lag']:.4f} nats per lag, "
        f"held-out log-likelihood non-reversible minus reversible {irr['cv_nonrev_minus_rev_per_transition']:+.4f} "
        f"nats per transition.", "",
        f"On microstates (sparse counts, so noisy): count asymmetry {mi['count_asymmetry']:.3f}, one-way count share "
        f"{mi['one_way_count_share']:.3f}; held-out log-likelihood non-reversible minus reversible "
        f"{cv['nonrev_minus_rev']:+.4f} nats per transition.", ""]
    for n in FITS:
        ab = r["absorbing_view"][n]
        lines.append(f"Absorbing view on {FIT_LABEL[n]} macrostates: P(proof switches at least once) model "
                     f"{ab['switch_model']:.3f} vs empirical {ab['switch_empirical']:.3f}; max hitting error "
                     f"{ab['max_abs_hit_error']:.3f}; mean steps to no goals {ab['mean_steps_to_done_model']:.1f}.")
    if r.get("vs_labels"):
        lines += ["", "Against labels (NMI / ARI): " + "; ".join(
            f"{FIT_LABEL[n]} {v['nmi']:.3f} / {v['ari']:.3f}" for n, v in r["vs_labels"].items() if v)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="markov-lean nonrev-check",
                                description="Fit reversible, non-reversible and absorbing MSMs on the same "
                                            "microstates and compare them (no API calls).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--msm", help="saved model folder from `pilot`/`full` (uses its microstates and lag)")
    src.add_argument("--steps", help="steps parquet: fit the usual pipeline first")
    src.add_argument("--synthetic", choices=["reversible", "directed"],
                     help="synthetic proofs on a symmetric (reversible) or one-way (directed) area chain")
    p.add_argument("--config", help="YAML with an `msm:` section (MSMConfig fields), for --steps/--synthetic")
    p.add_argument("--out", required=True)
    p.add_argument("--labels", help="judge labels table (proof_id, step, judge, msc) to score each fit against")
    p.add_argument("--judges", nargs="+", default=None, help="judges for the consensus (default: first two)")
    p.add_argument("--label-col", default=None, help="label column in the steps table (e.g. true_area)")
    p.add_argument("--n-macrostates", type=int, default=None)
    p.add_argument("--lag", type=int, default=None)
    p.add_argument("--eta", choices=["empirical", "uniform"], default="empirical")
    p.add_argument("--n-proofs", type=int, default=300, help="synthetic proofs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    cfg = NonrevConfig(eta=a.eta, seed=a.seed)

    labels = None
    if a.labels:
        from .robustness import _read_labels, consensus, expand_labels

        lab = _read_labels(Path(a.labels))
        judges = a.judges or [j for j in dict.fromkeys(lab["judge"].astype(str)) if j != "consensus"][:2]
        cons = consensus(lab, judges)

        def labels_for(keys: pd.DataFrame) -> pd.Series:
            return pd.Series(expand_labels(keys, cons).to_numpy(), index=pd.MultiIndex.from_frame(keys))

    if a.msm:
        if a.labels:
            asg = pd.read_parquet(Path(a.msm) / "assignments.parquet", columns=["proof_id", "step"])
            asg["proof_id"] = asg["proof_id"].astype(str)
            labels = labels_for(asg.sort_values(["proof_id", "step"]).reset_index(drop=True))
        res = compare_saved(a.msm, cfg, a.n_macrostates, a.lag, labels)
    else:
        mkw = {}
        if a.config:
            import yaml

            mkw = (yaml.safe_load(Path(a.config).read_text()) or {}).get("msm", {})
        if a.n_macrostates:
            mkw["n_macrostates"] = a.n_macrostates
        if a.lag:
            mkw["lag"] = a.lag
        mkw.setdefault("seed", a.seed)
        if a.synthetic:
            steps, _ = generate_directed(n_proofs=a.n_proofs, forward=1.0 if a.synthetic == "directed" else 0.5,
                                         seed=a.seed)
            mkw.setdefault("n_microstates", 60)
            mkw.setdefault("n_macrostates", 6)
            mkw.setdefault("lag", 1)
            mkw.setdefault("error_samples", 0)
            label_col = a.label_col or "true_area"
        else:
            steps = pd.read_parquet(a.steps)
            label_col = a.label_col
        if a.labels:
            keys = steps.astype({"proof_id": str})[["proof_id", "step"]].sort_values(["proof_id", "step"])
            labels = labels_for(keys.reset_index(drop=True))
        mcfg = M.MSMConfig(**mkw)
        cfg.its_lags, cfg.ck_mlags, cfg.n_its = list(mcfg.its_lags), mcfg.ck_mlags, mcfg.n_its
        res, _ = compare_steps(steps, mcfg, cfg, label_col=label_col, labels=labels)
    out = res.save(a.out)
    if not a.no_figures:
        figures(res, out)
    md = summary_md(res)
    (out / "SUMMARY.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
