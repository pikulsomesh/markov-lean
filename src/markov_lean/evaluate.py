"""Scoring macrostates against labels.

* ``agreement``: NMI and ARI of a predicted partition against any label column
  (``true_area`` on synthetic data, a judge's ``msc`` on real data).
* ``next_area_topk``: does the coarse transition matrix predict where a proof
  goes next?  Reported next to a "stay where you are" baseline and a
  marginal-frequency baseline, because sticky chains make "stay" strong.
* ``kappa``: Cohen's kappa between two label columns (two judges, or a judge
  and the synthetic truth).
* ``folder_labels`` / ``folder_baseline``: the top-level Mathlib folder from
  ``file_path`` as a one-label-per-proof baseline partition.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score

KEY = ["proof_id", "step"]


def _valid(a: pd.Series, b: pd.Series) -> np.ndarray:
    a_ok = a.notna() & (a.astype(str) != "")
    b_ok = b.notna() & (b.astype(str) != "")
    if pd.api.types.is_numeric_dtype(a):
        a_ok &= a != -1
    return (a_ok & b_ok).to_numpy()


def agreement(pred, true) -> dict:
    """NMI (arithmetic mean normalization) and ARI on rows where both sides are set.

    ``pred == -1`` (a step outside the connected set) and missing labels are
    left out; ``n`` says how many rows were scored.
    """
    pred, true = pd.Series(pred).reset_index(drop=True), pd.Series(true).reset_index(drop=True)
    ok = _valid(pred, true)
    p, t = pred[ok].astype(str), true[ok].astype(str)
    if len(p) == 0:
        return {"nmi": float("nan"), "ari": float("nan"), "n": 0, "coverage": 0.0}
    return {
        "nmi": float(normalized_mutual_info_score(t, p)),
        "ari": float(adjusted_rand_score(t, p)),
        "n": int(len(p)),
        "coverage": float(ok.mean()),
    }


def score_assignments(assignments: pd.DataFrame, labels: pd.DataFrame, label_col: str,
                      pred_col: str = "macrostate") -> dict:
    """Join assignments to a table holding ``label_col`` on (proof_id, step), then score.

    ``labels`` can be the ``steps`` table (``true_area``) or one judge's slice of
    the ``labels`` table (``msc``).  Whole-proof labels (``step == -1``) are
    broadcast to every step of that proof.
    """
    lab = labels[KEY + [label_col]].copy()
    whole = lab[lab["step"] == -1]
    per_step = lab[lab["step"] != -1]
    m = assignments.merge(per_step, on=KEY, how="left")
    if len(whole):
        fill = assignments[["proof_id"]].merge(whole[["proof_id", label_col]], on="proof_id", how="left")[label_col]
        m[label_col] = m[label_col].where(m[label_col].notna(), fill.values)
    return agreement(m[pred_col], m[label_col])


def majority_map(pred, true) -> dict:
    """Map each predicted cluster to its most common label."""
    df = pd.DataFrame({"p": pd.Series(pred).values, "t": pd.Series(true).values}).dropna()
    return {p: Counter(g["t"]).most_common(1)[0][0] for p, g in df.groupby("p")}


def _pairs(seq_df: pd.DataFrame, col: str, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """(current, next-at-lag) pairs of ``col`` within each proof."""
    cur, nxt = [], []
    for _, g in seq_df.sort_values(KEY).groupby("proof_id", sort=False):
        v = g[col].to_numpy()
        if len(v) > lag:
            cur.append(v[:-lag])
            nxt.append(v[lag:])
    if not cur:
        return np.array([]), np.array([])
    return np.concatenate(cur), np.concatenate(nxt)


def next_state_topk(assignments: pd.DataFrame, coarse_T: np.ndarray, k: int = 1, lag: int = 1,
                    col: str = "macrostate") -> dict:
    """Top-k accuracy of predicting the macrostate ``lag`` steps ahead from ``coarse_T[current]``.

    Also returns the "stay" baseline (predict the current state) and a
    marginal baseline (the k most frequent states overall).
    """
    cur, nxt = _pairs(assignments, col, lag)
    ok = (cur >= 0) & (nxt >= 0)
    cur, nxt = cur[ok].astype(int), nxt[ok].astype(int)
    if len(cur) == 0:
        return {"topk": float("nan"), "stay": float("nan"), "marginal": float("nan"), "n": 0, "k": k}
    T = np.asarray(coarse_T)
    T_lag = np.linalg.matrix_power(T, lag)
    topk = np.argsort(-T_lag, axis=1)[:, :k]
    hit = (topk[cur] == nxt[:, None]).any(axis=1)
    freq = np.bincount(nxt, minlength=T.shape[0])
    marg = set(np.argsort(-freq)[:k].tolist())
    return {
        "topk": float(hit.mean()),
        "stay": float((cur == nxt).mean()),
        "marginal": float(np.mean([x in marg for x in nxt])),
        "n": int(len(cur)),
        "k": int(k),
    }


def next_area_topk(assignments: pd.DataFrame, labels: pd.DataFrame, label_col: str, coarse_T: np.ndarray,
                   k: int = 1, lag: int = 1) -> dict:
    """Top-k accuracy of predicting the next *area label* from the coarse MSM.

    Each macrostate is mapped to its majority label; the predicted areas are
    the labels of the k likeliest next macrostates (deduplicated, keeping
    order, so fewer than k distinct areas can come out).  Baselines: "stay"
    (next area = current area label) and the k most frequent areas.
    """
    m = assignments.merge(labels[KEY + [label_col]], on=KEY, how="left")
    mapping = majority_map(m.loc[m["macrostate"] >= 0, "macrostate"], m.loc[m["macrostate"] >= 0, label_col])
    T_lag = np.linalg.matrix_power(np.asarray(coarse_T), lag)
    order = np.argsort(-T_lag, axis=1)
    cur_ms, _ = _pairs(m, "macrostate", lag)
    cur_lab, nxt_lab = _pairs(m, label_col, lag)
    ok = (cur_ms >= 0) & pd.notna(nxt_lab) & pd.notna(cur_lab)
    cur_ms, cur_lab, nxt_lab = cur_ms[ok].astype(int), cur_lab[ok], nxt_lab[ok]
    if len(cur_ms) == 0:
        return {"topk": float("nan"), "stay": float("nan"), "marginal": float("nan"), "n": 0, "k": k}
    pred_sets = []
    for row in order:
        seen: list = []
        for j in row:
            lab = mapping.get(j)
            if lab is not None and lab not in seen:
                seen.append(lab)
            if len(seen) == k:
                break
        pred_sets.append(seen)
    hit = np.array([nl in pred_sets[c] for c, nl in zip(cur_ms, nxt_lab)])
    top = [a for a, _ in Counter(nxt_lab).most_common(k)]
    return {
        "topk": float(hit.mean()),
        "stay": float(np.mean(cur_lab == nxt_lab)),
        "marginal": float(np.mean([x in top for x in nxt_lab])),
        "n": int(len(cur_ms)),
        "k": int(k),
        "macrostate_to_label": {int(a): b for a, b in mapping.items()},
    }


def kappa(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Cohen's kappa between two label columns, on rows where both are set."""
    ok = _valid(df[col_a], df[col_b])
    if ok.sum() == 0:
        return {"kappa": float("nan"), "n": 0, "raw_agreement": float("nan")}
    a, b = df[col_a][ok].astype(str), df[col_b][ok].astype(str)
    return {"kappa": float(cohen_kappa_score(a, b)), "n": int(ok.sum()), "raw_agreement": float((a == b).mean())}


def judge_kappa(labels: pd.DataFrame, judge_a: str, judge_b: str, label_col: str = "msc") -> dict:
    """Kappa between two judges in the long ``labels`` table, matched on (proof_id, step)."""
    a = labels[labels["judge"] == judge_a][KEY + [label_col]].rename(columns={label_col: "a"})
    b = labels[labels["judge"] == judge_b][KEY + [label_col]].rename(columns={label_col: "b"})
    return kappa(a.merge(b, on=KEY, how="inner"), "a", "b")


def folder_label(file_path: str | None) -> str | None:
    """Top-level Mathlib folder: ``Mathlib/NumberTheory/Padics/X.lean`` -> ``NumberTheory``.

    A file directly under ``Mathlib/`` gives its stem; anything outside
    Mathlib or empty gives ``None``.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    parts = [p for p in file_path.replace("\\", "/").split("/") if p and p != "."]
    if "Mathlib" not in parts:
        return None
    parts = parts[parts.index("Mathlib") + 1:]
    if not parts:
        return None
    head = parts[0]
    return head[:-5] if head.endswith(".lean") else head


def folder_labels(steps: pd.DataFrame) -> pd.Series:
    return steps["file_path"].map(folder_label)


def folder_baseline(steps: pd.DataFrame, label_col: str) -> dict:
    """Score the folder partition (one label per proof) against ``label_col``."""
    return agreement(folder_labels(steps), steps[label_col])
