"""Area prior for proof search (workstream B).

Turns the Markov state model's areas into weights over candidate tactics, so
a prover can lean toward (or away from) areas of mathematics while it
searches.

The pieces work on plain arrays so they don't depend on ``msm.py``:

* ``centroids``: k-means microstate centers, shape ``(n_micro, d)``
  (``MSMResult.centers``).
* ``membership``: PCCA+ membership matrix, shape ``(n_micro, n_macro)``. Rows
  of NaN (microstates outside the connected set) are read as "no
  information" and spread evenly over the macrostates.
* ``featurizer``: maps one goal state to a ``(d,)`` vector. Either a plain
  callable ``str -> vector`` or an object with ``transform(list[str])``,
  such as ``featurize.TfidfSvdFeaturizer``.

The search loop (workstream A) calls ``prior(state_text, candidates)`` and
gets one weight per candidate, summing to 1. ``make_prior`` builds that
callable. Each candidate's area is read from the goal state it leads to.

Scoring, step by step:

1. ``area_distribution`` soft-assigns a state vector to its nearest
   microstates and pushes that through the membership matrix, giving a
   distribution over macrostates ("areas").
2. A per-area target is built from the current state's area (moved one step
   by ``coarse_T`` when given), times ``diversity_bonus`` (favor areas the
   search has visited less) and ``stuck_escape`` (a metadynamics-style
   penalty on areas the search has sat in for ``window`` steps without
   progress).
3. A candidate's area score is the expected target under its resulting
   state's area distribution. A candidate that closes the goal gets the top
   score; one whose resulting state is unknown gets a neutral score.
4. ``reweight_prior`` mixes the LM's distribution over candidates with the
   normalized area scores: ``(1 - lam) * p_lm + lam * p_area``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

ArrayLike = Any
CLOSED_STATES = frozenset({"no goals", ""})
_EPS = 1e-12


# --------------------------------------------------------------------------
# small helpers


def _normalize(p: ArrayLike, name: str = "p") -> np.ndarray:
    """Clip to non-negative and normalize; all-zero (or empty) gives uniform."""
    p = np.asarray(p, dtype=float).ravel()
    if p.size == 0:
        return p
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} has non-finite entries")
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= _EPS:
        return np.full(p.size, 1.0 / p.size)
    return p / s


def _clean_membership(membership: ArrayLike) -> np.ndarray:
    M = np.asarray(membership, dtype=float)
    if M.ndim != 2 or M.shape[1] == 0:
        raise ValueError("membership must be (n_micro, n_macro) with n_macro >= 1")
    M = M.copy()
    bad = ~np.all(np.isfinite(M), axis=1)
    M[bad] = 1.0
    M = np.clip(M, 0.0, None)
    s = M.sum(axis=1, keepdims=True)
    zero = s[:, 0] <= _EPS
    M[zero] = 1.0
    s[zero] = M.shape[1]
    return M / s


# --------------------------------------------------------------------------
# 1. area of one state


def area_distribution(
    state_vec: ArrayLike,
    centroids: ArrayLike,
    membership: ArrayLike,
    temperature: float = 0.1,
    k: int | None = None,
) -> np.ndarray:
    """Distribution over macrostates for one featurized goal state.

    Soft-assigns ``state_vec`` to microstates with a softmax over negative
    squared distances to ``centroids`` (scaled by ``temperature``), keeps the
    ``k`` nearest when ``k`` is given, then maps through ``membership``.

    A zero or non-finite vector carries no information (an empty goal under
    TF-IDF features is all zeros), so it returns the uniform distribution.
    ``temperature <= 0`` means hard assignment to the nearest microstate.
    """
    C = np.asarray(centroids, dtype=float)
    M = _clean_membership(membership)
    if C.ndim != 2 or C.shape[0] != M.shape[0]:
        raise ValueError(
            f"centroids {C.shape} and membership {M.shape} disagree on n_micro"
        )
    n_macro = M.shape[1]
    x = np.asarray(state_vec, dtype=float).ravel()
    if x.shape[0] != C.shape[1]:
        raise ValueError(f"state_vec has dim {x.shape[0]}, centroids have {C.shape[1]}")
    if not np.all(np.isfinite(x)) or np.linalg.norm(x) <= _EPS:
        return np.full(n_macro, 1.0 / n_macro)

    d2 = ((C - x) ** 2).sum(axis=1)
    if temperature is None or temperature <= 0:
        w = np.zeros_like(d2)
        w[int(np.argmin(d2))] = 1.0
    else:
        logits = -(d2 - d2.min()) / float(temperature)
        if k is not None and 0 < k < len(d2):
            keep = np.argpartition(d2, k - 1)[:k]
            mask = np.full(len(d2), -np.inf)
            mask[keep] = 0.0
            logits = logits + mask
        w = np.exp(logits)
        w /= w.sum()
    return _normalize(w @ M)


# --------------------------------------------------------------------------
# 2. mixing


def reweight_prior(p_lm: ArrayLike, area_scores: ArrayLike, lam: float) -> np.ndarray:
    """``(1 - lam) * p_LM + lam * p_area``, renormalized.

    ``p_lm`` and ``area_scores`` are both normalized first (negative scores
    clip to 0; all-zero becomes uniform). ``lam = 0`` returns ``p_lm``
    unchanged and ``lam = 1`` returns the area distribution alone.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    p = _normalize(p_lm, "p_lm")
    a = _normalize(area_scores, "area_scores")
    if p.shape != a.shape:
        raise ValueError(f"p_lm {p.shape} and area_scores {a.shape} differ in length")
    if p.size == 0:
        return p
    return _normalize((1.0 - lam) * p + lam * a)


def logprobs_to_probs(logprobs: ArrayLike) -> np.ndarray:
    """Stable softmax for LM log-probabilities (``-inf`` entries get 0)."""
    lp = np.asarray(logprobs, dtype=float).ravel()
    if lp.size == 0:
        return lp
    if np.all(np.isneginf(lp)):
        return np.full(lp.size, 1.0 / lp.size)
    w = np.exp(lp - np.max(lp[np.isfinite(lp)]))
    return _normalize(w, "logprobs")


# --------------------------------------------------------------------------
# 3. search-history biases


def diversity_bonus(visited_area_counts: ArrayLike, alpha: float = 0.5) -> np.ndarray:
    """Per-area multiplier ``(1 + count) ** -alpha`` in ``(0, 1]``.

    Areas the search hasn't visited get 1; heavily visited areas shrink
    toward 0. Counts may be fractional (sums of soft area distributions).
    ``alpha = 0`` switches it off.
    """
    c = np.asarray(visited_area_counts, dtype=float).ravel()
    if c.size and (not np.all(np.isfinite(c)) or np.any(c < 0)):
        raise ValueError("visited_area_counts must be finite and non-negative")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    return (1.0 + c) ** (-alpha)


def _as_area_vector(area: Any, n_areas: int) -> np.ndarray:
    if np.ndim(area) == 0:
        a = int(area)
        if not 0 <= a < n_areas:
            raise ValueError(f"area {a} out of range for {n_areas} areas")
        v = np.zeros(n_areas)
        v[a] = 1.0
        return v
    v = np.asarray(area, dtype=float).ravel()
    if v.size != n_areas:
        raise ValueError(f"area vector has length {v.size}, expected {n_areas}")
    return _normalize(v, "area")


def stuck_escape(
    history: Sequence[tuple[Any, float]],
    window: int,
    threshold: float,
    n_areas: int,
    strength: float = 2.0,
) -> np.ndarray:
    """Metadynamics-style escape bias, as a per-area multiplier in ``(0, 1]``.

    ``history`` holds one ``(area, progress)`` pair per search step, oldest
    first. ``area`` is a macrostate index or a distribution over areas;
    ``progress`` is any "did this step help" number (goals closed, drop in
    goal size, score gain).

    If the last ``window`` steps made less than ``threshold`` total progress,
    each area gets ``exp(-strength * occupancy)``, where ``occupancy`` is the
    fraction of those steps spent in it: the hills pile up where the search
    has been stuck and push it elsewhere. Otherwise (or with fewer than
    ``window`` steps of history) every area gets 1.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if n_areas < 1:
        raise ValueError("n_areas must be >= 1")
    ones = np.ones(n_areas)
    if len(history) < window:
        return ones
    recent = list(history)[-window:]
    total = float(sum(float(prog) for _, prog in recent))
    if total >= threshold:
        return ones
    occ = sum(_as_area_vector(area, n_areas) for area, _ in recent) / window
    return np.exp(-strength * occ)


# --------------------------------------------------------------------------
# 4. the callable the search loop uses


def _get(c: Any, key: str, default: Any = None) -> Any:
    if isinstance(c, dict):
        return c.get(key, default)
    return getattr(c, key, default)


def default_next_state(state_text: str, candidate: Any) -> str | None:
    """Read the resulting goal state off a candidate.

    Accepts a ``(tactic, state_after)`` tuple, a dict or object with a
    ``state_after`` field, or a bare tactic string (unknown result -> None).
    """
    if isinstance(candidate, tuple) and len(candidate) >= 2:
        return candidate[1]
    return _get(candidate, "state_after")


def default_lm_logprob(candidate: Any) -> float | None:
    """LM log-probability from a ``logprob`` field, or None when absent."""
    if isinstance(candidate, tuple) and len(candidate) >= 3:
        return candidate[2]
    return _get(candidate, "logprob")


def _featurize_one(featurizer: Any, text: str) -> np.ndarray:
    if hasattr(featurizer, "transform"):
        return np.asarray(featurizer.transform([text]), dtype=float)[0]
    return np.asarray(featurizer(text), dtype=float).ravel()


@dataclass
class AreaPrior:
    """``prior(state_text, candidates) -> weights`` with search-history state.

    Call ``record(state_text, progress)`` after each expanded step so the
    diversity and stuck-escape terms see the search history; ``reset()``
    between theorems.
    """

    featurizer: Any
    centroids: np.ndarray
    membership: np.ndarray
    lam: float = 0.3
    coarse_T: np.ndarray | None = None
    temperature: float = 0.1
    k: int | None = None
    diversity_alpha: float = 0.5
    window: int = 8
    threshold: float = 1.0
    escape_strength: float = 2.0
    next_state: Callable[[str, Any], str | None] = default_next_state
    lm_logprob: Callable[[Any], float | None] = default_lm_logprob
    history: list[tuple[np.ndarray, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {self.lam}")
        self.centroids = np.asarray(self.centroids, dtype=float)
        self.membership = _clean_membership(self.membership)
        if self.centroids.shape[0] != self.membership.shape[0]:
            raise ValueError("centroids and membership disagree on n_micro")
        if self.coarse_T is not None:
            T = np.asarray(self.coarse_T, dtype=float)
            if T.shape != (self.n_areas, self.n_areas):
                raise ValueError(f"coarse_T must be ({self.n_areas}, {self.n_areas})")
            self.coarse_T = np.vstack([_normalize(r, "coarse_T row") for r in T])
        self._cache: dict[str, np.ndarray] = {}

    @property
    def n_areas(self) -> int:
        return self.membership.shape[1]

    def area_of(self, state_text: str) -> np.ndarray:
        """Area distribution of a goal state (cached by text)."""
        hit = self._cache.get(state_text)
        if hit is None:
            vec = _featurize_one(self.featurizer, state_text)
            hit = area_distribution(vec, self.centroids, self.membership, self.temperature, self.k)
            self._cache[state_text] = hit
        return hit

    def visited_counts(self) -> np.ndarray:
        if not self.history:
            return np.zeros(self.n_areas)
        return np.sum([a for a, _ in self.history], axis=0)

    def target(self, state_text: str) -> np.ndarray:
        """Per-area desirability, scaled so its maximum is 1."""
        base = self.area_of(state_text)
        if self.coarse_T is not None:
            base = base @ self.coarse_T
        else:
            base = np.ones(self.n_areas)
        t = (
            base
            * diversity_bonus(self.visited_counts(), self.diversity_alpha)
            * stuck_escape(self.history, self.window, self.threshold, self.n_areas,
                           self.escape_strength)
        )
        m = t.max()
        return t / m if m > _EPS else np.ones(self.n_areas)

    def area_scores(self, state_text: str, candidates: Sequence[Any]) -> np.ndarray:
        t = self.target(state_text)
        neutral = float(t.mean())
        scores = np.empty(len(candidates))
        for i, c in enumerate(candidates):
            nxt = self.next_state(state_text, c)
            if nxt is None:
                scores[i] = neutral
            elif nxt.strip() in CLOSED_STATES:
                scores[i] = 1.0  # closing the goal is never penalized
            else:
                scores[i] = float(self.area_of(nxt) @ t)
        return scores

    def lm_probs(self, candidates: Sequence[Any]) -> np.ndarray:
        lps = [self.lm_logprob(c) for c in candidates]
        if any(lp is None for lp in lps):
            return np.full(len(candidates), 1.0 / max(len(candidates), 1))
        return logprobs_to_probs(lps)

    def __call__(self, state_text: str, candidates: Sequence[Any]) -> np.ndarray:
        if len(candidates) == 0:
            return np.zeros(0)
        return reweight_prior(self.lm_probs(candidates),
                              self.area_scores(state_text, candidates), self.lam)

    def record(self, state_text: str, progress: float) -> None:
        """Add one expanded step to the search history."""
        self.history.append((self.area_of(state_text), float(progress)))

    def reset(self) -> None:
        self.history.clear()


def make_prior(
    featurizer: Any,
    centroids: ArrayLike,
    membership: ArrayLike,
    lam: float = 0.3,
    **kwargs: Any,
) -> AreaPrior:
    """Build the ``prior(state_text, candidates) -> weights`` callable.

    Keyword arguments go to ``AreaPrior`` (``coarse_T``, ``temperature``,
    ``k``, ``diversity_alpha``, ``window``, ``threshold``,
    ``escape_strength``, ``next_state``, ``lm_logprob``).
    """
    return AreaPrior(featurizer, np.asarray(centroids, dtype=float),
                     np.asarray(membership, dtype=float), lam=lam, **kwargs)


__all__ = [
    "AreaPrior",
    "area_distribution",
    "default_lm_logprob",
    "default_next_state",
    "diversity_bonus",
    "logprobs_to_probs",
    "make_prior",
    "reweight_prior",
    "stuck_escape",
]
