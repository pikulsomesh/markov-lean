"""Bridge steps: which tactics and lemmas sit where a proof changes area.

``markov-lean bridges --run results/pilot3 --out results/bridges``

A *move* is a pair of consecutive steps ``t`` and ``t + 1`` of one proof whose
goal states both carry an area. The judges label the goal *before* each step, so
the tactic at step ``t`` is the one that turns the step-``t`` goal into the
step-``t + 1`` goal: the move is credited to that tactic and to the lemmas it
cites. A move is a *switch* when the two areas differ.

Two area sources, each analysed on its own:

* ``judge``: step-level MSC labels from ``<run>/judge/labels.parquet``. The
  ``consensus`` judge is used when present, else the two judges' consensus
  (rows where both agree), else the one judge there is (``--judge`` picks one).
* ``macrostate``: PCCA+ macrostates from ``<run>/msm/assignments.parquet``.

The tactic texts come from ``<run>/judge/sample_steps.parquet`` (the proofs the
judges saw), or any ``steps`` table given with ``--steps``; macrostate moves
need the steps table to cover the proofs, so pass the pilot's full steps table
to use every assigned proof.

For a feature ``f`` (a tactic name, or a cited lemma) the **lift** is

    P(f | switch) / P(f | move)  =  P(switch | f) / P(switch),

so a lift of 3 means moves that use ``f`` switch area three times as often as
moves overall. Per area pair ``a -> b`` the base rate is the moves out of ``a``:
``P(f | a -> b switch) / P(f | move from a)``, which keeps tactics that are
simply common in area ``a`` from looking like bridges. Intervals are percentile
bootstraps over whole proofs (moves inside a proof are not independent).
Overall, a feature is ranked once it is used on ``--min-count`` moves (so
tactics that avoid switches show up with lift below 1); within a pair it needs
that many switches of that pair, and the pair ``--min-pair-switches``.
``p_boot`` is the bootstrap share of resamples with lift <= 1 (one-sided, with
the usual +1 correction) and ``q_value`` its Benjamini-Hochberg adjustment
within each source, scope and kind.

Cited lemmas are read from the tactic text, because the ``steps`` schema keeps
LeanDojo's tactic text but not its premise annotations: an identifier counts
when it is namespaced (``Finset.sum_comm``) or snake_case (``mul_pos``), is not
a tactic or keyword, is not a hypothesis or variable of the goal before the
step, and is not a name the tactic itself binds (``have h_key``, ``intro x``,
``obtain ⟨x, hx⟩``, ``fun x =>``). A ``premises`` column in the steps table, if
present (a list or a space-separated string), is used instead.

Outputs in ``--out``:

| file | what it holds |
|---|---|
| ``bridges.parquet`` / ``.csv`` | one row per area source x scope (``all`` or ``pair``) x feature, with counts, lift, interval, ``p_boot``, ``q_value`` and ``rank`` (1 = highest lift within its group) |
| ``moves.parquet`` / ``.csv`` | every move: proof, step, areas, switch flag, tactic name, cited lemmas |
| ``summary.json`` | counts, settings and inputs used |
| ``REPORT.md`` | the top tactics and lemmas overall and for the busiest area pairs |
| ``bridges.pdf`` | lift with 95% intervals for the top tactics and lemmas (needs the ``figures`` extra) |

Parquet is written when pyarrow is installed; CSV always.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

KEY = ["proof_id", "step"]
SOURCES = ("judge", "macrostate")
KINDS = ("tactic", "premise")

# Tactic names and Lean words that look like lemma names (snake_case) or that
# would otherwise be read as cited names.
TACTIC_WORDS = frozenset("""
abel abel_nf absurd ac_rfl aesop aesop_cat all_goals any_goals apply apply_assumption apply_fun apply_mod_cast
apply_rules arith assumption assumption_mod_cast bound by_cases by_contra by_contra' by_contra! calc case case'
cases cases' change choose clear clear_value compute_degree congr congr! congrm constructor continuity contrapose
contrapose! conv conv_lhs conv_rhs convert convert_to dsimp decide delta exact exact? exact_mod_cast exacts
exfalso exists existsi ext ext1 field_simp fin_cases filter_upwards first focus fun_prop funext gcongr gcongr?
generalize group haveI have induction induction' infer_instance interval_cases intro intros introv iterate left
lift let letI linarith linarith! linear_combination measurability mono monicity native_decide next nlinarith
nlinarith! noncomm_ring nontriviality norm_cast norm_num norm_num1 nth_rewrite nth_rw obtain omega on_goal
positivity polyrith push_cast push_neg qify rcases refine refine' refine_lift rename rename_i repeat revert rfl
right ring ring1 ring_nf ring_exp rintro rify rw rw? rw_mod_cast rwa rewrite set set_option show simp simp?
simp_all simp_all? simp_arith simp_intro simp_rw simpa simpa? slim_check solve_by_elim specialize split split_ifs
subst subst_vars suffices swap symm tauto tfae_finish tfae_have trans trivial triv try unfold unfold_let use
use! wlog zify exact? apply? decreasing_tactic decreasing_trivial positivity_core
""".split())
LEAN_WORDS = frozenset("""
at by do else from fun if in let match mut only then this using with where Type Sort Prop true false
maxHeartbeats maxRecDepth config decide discharger contextual zeta beta eta iota proj
""".split())
# Hypothesis-style names (h, h1, h₀, h_main, hx, ih, this) are local, never lemmas.
_HYP = re.compile(r"^(?:h(?:[_\d₀-₉'].*)?|ih[\w']*|IH[\w']*|this|hyp[\w']*|key|claim|aux)$")
_IDENT = re.compile(r"(?<![\w.'!?])(?:_root_\.)?[^\W\d][\w'!?]*(?:\.(?:[^\W\d][\w'!?]*|\d+))*", re.UNICODE)
_HEAD = re.compile(r"[^\W\d][\w'!?]*", re.UNICODE)
_BULLET = re.compile(r"^(?:[·•]\s*|\.\s+|(?:case|case'|next)\b[^=]*=>\s*|on_goal\s+-?\d+\s*=>\s*)+")
_COMMENT = re.compile(r"--[^\n]*|/-.*?-/", re.DOTALL)
# Names a tactic binds itself (``have h_key : ...``, ``intro x y``, ``fun x =>``).
_BIND_ONE = re.compile(
    r"\b(?:have|haveI|let|letI|set|suffices|generalize|by_cases|by_contra!?|by_contra'|specialize|choose!?)\s+"
    r"([^\W\d][\w'!?]*)", re.UNICODE)
_BIND_REST = re.compile(r"\b(?:intro|intros|introv|rintro|ext|ext1|funext|rename_i)\b([^:=⊢<;]*)", re.UNICODE)
_BIND_BINDER = re.compile(r"(?:\bfun\b|λ|∀|∃!?|∑|∏|⋃|⋂)\s*([^=,:]*?)(?:=>|↦|,|:|\bin\b)", re.UNICODE)
_BIND_ANGLE = re.compile(r"(?:\bobtain\b|\brcases\b|\brintro\b|\bwith\b|\bchoose\b)\s*[^⟨]*?(⟨[^⊢:=]*⟩)", re.UNICODE)
_HYP_LINE = re.compile(r"^(?!⊢|case\b)([^\s:][^:]*?)\s:\s?", re.UNICODE)


# --------------------------------------------------------------------------- parsing


def _clean(text: Any) -> str:
    return _COMMENT.sub(" ", "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text))


def tactic_name(text: Any) -> str:
    """The head tactic of a tactic text: ``simp only [foo] at h`` -> ``simp``, ``· nlinarith [..]`` -> ``nlinarith``."""
    t = _BULLET.sub("", _clean(text).strip()).lstrip("( ").strip()
    m = _HEAD.match(t)
    if m:
        return m.group(0)
    return t.split()[0] if t.split() else ""


def local_names(state_before: Any) -> set[str]:
    """Hypothesis and variable names in a pretty-printed goal state (``x y : ℕ``, ``h : x < y``)."""
    names: set[str] = set()
    for line in _clean(state_before).splitlines():
        if not line or line[0].isspace():
            continue  # wrapped continuation of a long type
        m = _HYP_LINE.match(line)
        if m:
            names.update(n.rstrip("✝") for n in m.group(1).split())
    return names


def bound_names(tactic: Any) -> set[str]:
    """Names the tactic text binds itself; these are local, never cited lemmas."""
    t = _clean(tactic)
    out = set(_BIND_ONE.findall(t))
    for rest in _BIND_REST.findall(t) + _BIND_BINDER.findall(t) + _BIND_ANGLE.findall(t):
        out.update(m.group(0) for m in _IDENT.finditer(rest))
    return out


def cited_names(tactic: Any, state_before: Any = "") -> list[str]:
    """Lemma and definition names a tactic cites, read from its text (sorted, unique)."""
    t = _clean(tactic)
    local = local_names(state_before) | bound_names(t)
    found = set()
    for m in _IDENT.finditer(t):
        name = m.group(0)
        if name.startswith("_root_."):
            name = name[len("_root_."):]
        if "." not in name and "_" not in name.strip("_"):
            continue
        if name.startswith("_") or name in TACTIC_WORDS or name in LEAN_WORDS:
            continue
        head = name.split(".")[0]
        if head in local or _HYP.match(head) or name in local:
            continue
        found.add(name)
    return sorted(found)


def _premise_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            try:
                return sorted({str(v) for v in json.loads(value)})
            except json.JSONDecodeError:
                pass
        return sorted(set(value.split()))
    return sorted({str(v) for v in value})


# --------------------------------------------------------------------------- inputs


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix in (".jsonl", ".json"):
        return pd.read_json(p, lines=p.suffix == ".jsonl")
    raise ValueError(f"unsupported table format: {p}")


def _keyed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["proof_id"] = df["proof_id"].astype(str)
    df["step"] = pd.to_numeric(df["step"]).astype("int64")
    return df


def judge_areas(labels: pd.DataFrame, judge: str | None = None, min_confidence: float = 0.0) -> tuple[pd.DataFrame, str]:
    """Step-level MSC areas from a ``labels`` table, and a note on which judge rows were used."""
    lab = _keyed(labels)
    lab = lab[lab["step"] >= 0]
    judges = [j for j in dict.fromkeys(lab["judge"].astype(str))]
    if judge:
        lab, used = lab[lab["judge"].astype(str) == judge], judge
    elif "consensus" in judges:
        lab, used = lab[lab["judge"].astype(str) == "consensus"], "consensus"
    elif len(judges) >= 2:
        from .judge import consensus

        lab, used = _keyed(consensus(lab, judges[0], judges[1])), f"consensus of {judges[0]} and {judges[1]}"
    elif judges:
        used = judges[0]
    else:
        return pd.DataFrame(columns=KEY + ["area"]), "no step labels"
    if min_confidence > 0 and "confidence" in lab.columns:
        lab = lab[lab["confidence"].astype(float) >= min_confidence]
    out = lab.drop_duplicates(KEY)[KEY].assign(area=lab.drop_duplicates(KEY)["msc"].astype(str).str.zfill(2).values)
    return out.reset_index(drop=True), used


def macrostate_areas(assignments: pd.DataFrame, min_membership: float = 0.0) -> pd.DataFrame:
    a = _keyed(assignments)
    if min_membership > 0 and "membership" in a.columns:
        a = a[a["membership"].astype(float) >= min_membership]
    a = a.drop_duplicates(KEY)
    return pd.DataFrame({"proof_id": a["proof_id"].values, "step": a["step"].values,
                         "area": a["macrostate"].astype("int64").astype(str).values})


def build_moves(steps: pd.DataFrame, areas: pd.DataFrame, source: str) -> pd.DataFrame:
    """One row per pair of consecutive area-labelled steps, credited to the tactic at the first step."""
    cols = ["area_source", "proof_id", "step", "from_area", "to_area", "switch", "tactic_name", "premises", "tactic"]
    if areas.empty:
        return pd.DataFrame(columns=cols)
    st = _keyed(steps)
    keep = [c for c in ("proof_id", "step", "tactic", "state_before", "premises") if c in st.columns]
    st = st[keep].drop_duplicates(KEY)
    ar = areas.copy()
    nxt = ar.assign(step=ar["step"] - 1).rename(columns={"area": "to_area"})
    m = ar.rename(columns={"area": "from_area"}).merge(nxt, on=KEY).merge(st, on=KEY)
    if m.empty:
        return pd.DataFrame(columns=cols)
    if "premises" in m.columns:
        prem = [" ".join(_premise_list(v)) for v in m["premises"]]
    else:
        prem = [" ".join(cited_names(t, s)) for t, s in zip(m["tactic"], m["state_before"])]
    out = pd.DataFrame({
        "area_source": source,
        "proof_id": m["proof_id"].values,
        "step": m["step"].values,
        "from_area": m["from_area"].astype(str).values,
        "to_area": m["to_area"].astype(str).values,
        "switch": (m["from_area"].astype(str) != m["to_area"].astype(str)).values,
        "tactic_name": [tactic_name(t) for t in m["tactic"]],
        "premises": prem,
        "tactic": m["tactic"].astype(str).values,
    })
    return out.sort_values(KEY).reset_index(drop=True)


# --------------------------------------------------------------------------- lift


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (nan stays nan)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    n = len(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(n)
    qq[order] = np.minimum(ranked, 1.0)
    q[ok] = qq
    return q


def _compact(keys: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Index of each key in the sorted ``keep`` array, or -1."""
    if len(keep) == 0:
        return np.full(len(keys), -1)
    pos = np.searchsorted(keep, keys)
    pos[pos >= len(keep)] = 0
    return np.where(keep[pos] == keys, pos, -1)


def lift_table(moves: pd.DataFrame, n_boot: int = 1000, seed: int = 0, min_count: int = 5,
               min_pair_switches: int = 10, ci: float = 0.95) -> pd.DataFrame:
    """Lift of every tactic and cited lemma at area switches, overall and per area pair, for one area source."""
    columns = ["area_source", "scope", "from_area", "to_area", "kind", "feature", "n_switch", "n_switch_total",
               "n_moves", "n_moves_total", "p_given_switch", "p_base", "switch_rate_given_feature",
               "base_switch_rate", "lift", "lift_lo", "lift_hi", "p_boot", "q_value", "rank"]
    if moves.empty or not moves["switch"].any():
        return pd.DataFrame(columns=columns)
    source = str(moves["area_source"].iloc[0])
    mv = moves.reset_index(drop=True)
    proofs, m_proof = np.unique(mv["proof_id"].astype(str).values, return_inverse=True)
    n_proofs = len(proofs)
    areas, codes = np.unique(np.concatenate([mv["from_area"].astype(str).values, mv["to_area"].astype(str).values]),
                             return_inverse=True)
    m_from, m_to = codes[: len(mv)], codes[len(mv):]
    A = len(areas)
    m_sw = mv["switch"].astype(bool).values
    m_pair = m_from * A + m_to

    # events: one per (move, feature), deduplicated within the move
    e_move, e_kind, e_feat = [], [], []
    for i, (tac, prem) in enumerate(zip(mv["tactic_name"], mv["premises"])):
        if tac:
            e_move.append(i), e_kind.append("tactic"), e_feat.append(str(tac))
        for p in set(str(prem).split()) if isinstance(prem, str) else set(prem or []):
            e_move.append(i), e_kind.append("premise"), e_feat.append(p)
    e_move = np.asarray(e_move, dtype=np.int64)
    feats = pd.Series([f"{k}\x00{f}" for k, f in zip(e_kind, e_feat)])
    fnames, e_f = np.unique(feats.values, return_inverse=True)
    F = len(fnames)
    e_proof, e_from, e_pair, e_sw = m_proof[e_move], m_from[e_move], m_pair[e_move], m_sw[e_move]

    # point counts and the candidates worth bootstrapping
    b_all = np.bincount(e_f, minlength=F)
    keep_all = np.flatnonzero(b_all >= min_count)
    pair_sw = np.bincount(m_pair[m_sw], minlength=A * A)
    good_pairs = np.flatnonzero(pair_sw >= min_pair_switches)
    sk = e_pair[e_sw] * F + e_f[e_sw]
    sk_keys, sk_counts = np.unique(sk, return_counts=True)
    cand = sk_keys[(sk_counts >= min_count) & np.isin(sk_keys // F, good_pairs)]
    base_keys = np.unique((cand // F) // A * F + cand % F)  # (from, feature) keys the pairs need

    i_all = _compact(e_f, keep_all)
    i_pair = np.where(e_sw, _compact(e_pair * F + e_f, cand), -1)
    i_base = _compact(e_from * F + e_f, base_keys)
    cand_pair, cand_feat = cand // F, cand % F
    cand_from = cand_pair // A
    cand_base = _compact(cand_from * F + cand_feat, base_keys)

    def counts(w: np.ndarray) -> dict[str, np.ndarray]:
        wm, we = w[m_proof], w[e_proof]
        return {
            "S": wm[m_sw].sum(), "M": wm.sum(),
            "s_all": np.bincount(i_all[(i_all >= 0) & e_sw], we[(i_all >= 0) & e_sw], len(keep_all)),
            "b_all": np.bincount(i_all[i_all >= 0], we[i_all >= 0], len(keep_all)),
            "S_pair": np.bincount(m_pair[m_sw], wm[m_sw], A * A)[cand_pair],
            "M_from": np.bincount(m_from, wm, A)[cand_from],
            "s_pair": np.bincount(i_pair[i_pair >= 0], we[i_pair >= 0], len(cand)),
            "b_pair": np.bincount(i_base[i_base >= 0], we[i_base >= 0], len(base_keys))[cand_base],
        }

    def lifts(c: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        with np.errstate(divide="ignore", invalid="ignore"):
            la = (c["s_all"] / c["S"]) / (c["b_all"] / c["M"])
            lp = (c["s_pair"] / c["S_pair"]) / (c["b_pair"] / c["M_from"])
        return la, lp

    point = counts(np.ones(n_proofs))
    la0, lp0 = lifts(point)
    rng = np.random.default_rng(seed)
    boot_a = np.full((n_boot, len(keep_all)), np.nan)
    boot_p = np.full((n_boot, len(cand)), np.nan)
    for b in range(n_boot):
        w = rng.multinomial(n_proofs, np.full(n_proofs, 1.0 / n_proofs)).astype(float)
        boot_a[b], boot_p[b] = lifts(counts(w))

    alpha = (1 - ci) / 2 * 100

    def interval(boot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if boot.shape[1] == 0 or n_boot == 0:
            nan = np.full(boot.shape[1], np.nan)
            return nan, nan, nan
        with np.errstate(invalid="ignore"), _quiet():
            lo = np.nanpercentile(boot, alpha, axis=0)
            hi = np.nanpercentile(boot, 100 - alpha, axis=0)
        valid = np.isfinite(boot).sum(0)
        le1 = (np.where(np.isfinite(boot), boot <= 1, False)).sum(0)
        p = np.where(valid > 0, (le1 + 1) / (valid + 1), np.nan)
        return lo, hi, p

    rows: list[pd.DataFrame] = []
    lo, hi, p = interval(boot_a)
    kinds, names = zip(*(f.split("\x00", 1) for f in fnames[keep_all])) if len(keep_all) else ((), ())
    rows.append(pd.DataFrame({
        "scope": "all", "from_area": None, "to_area": None, "kind": list(kinds), "feature": list(names),
        "n_switch": point["s_all"], "n_switch_total": point["S"], "n_moves": point["b_all"], "n_moves_total": point["M"],
        "lift": la0, "lift_lo": lo, "lift_hi": hi, "p_boot": p,
    }))
    lo, hi, p = interval(boot_p)
    kinds, names = zip(*(f.split("\x00", 1) for f in fnames[cand_feat])) if len(cand) else ((), ())
    rows.append(pd.DataFrame({
        "scope": "pair", "from_area": areas[cand_from] if len(cand) else [], "to_area": areas[cand_pair % A] if len(cand) else [],
        "kind": list(kinds), "feature": list(names),
        "n_switch": point["s_pair"], "n_switch_total": point["S_pair"], "n_moves": point["b_pair"],
        "n_moves_total": point["M_from"], "lift": lp0, "lift_lo": lo, "lift_hi": hi, "p_boot": p,
    }))
    out = pd.concat([r for r in rows if len(r)], ignore_index=True) if any(len(r) for r in rows) else pd.DataFrame(
        columns=columns)
    if out.empty:
        return pd.DataFrame(columns=columns)
    out.insert(0, "area_source", source)
    for c in ("n_switch", "n_switch_total", "n_moves", "n_moves_total"):
        out[c] = out[c].round().astype("int64")
    out["p_given_switch"] = out["n_switch"] / out["n_switch_total"]
    out["p_base"] = out["n_moves"] / out["n_moves_total"]
    out["switch_rate_given_feature"] = out["n_switch"] / out["n_moves"]
    out["base_switch_rate"] = out["n_switch_total"] / out["n_moves_total"]
    group = ["scope", "from_area", "to_area", "kind"]
    g = out.fillna({"from_area": "", "to_area": ""}).groupby(group, sort=False)
    out["rank"] = g["lift"].rank(ascending=False, method="first").astype("int64")
    out["q_value"] = np.nan
    for _, idx in out.groupby(["scope", "kind"]).groups.items():
        out.loc[idx, "q_value"] = _bh(out.loc[idx, "p_boot"].values)
    out = out.sort_values(["scope", "from_area", "to_area", "kind", "rank"], na_position="first")
    return out[columns].reset_index(drop=True)


class _quiet:
    """Silence numpy's all-nan slice warnings inside nanpercentile."""

    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


# --------------------------------------------------------------------------- figure


def bridge_figure(table: pd.DataFrame, source: str, top: int = 12):
    """Lift with intervals for the top tactics and cited lemmas at switches (overall scope), two panels."""
    from . import figures as fg

    matplotlib, plt = fg._plt()
    sub = table[(table["area_source"] == source) & (table["scope"] == "all")]
    with matplotlib.rc_context(fg.paper_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(fg.TEXT_WIDTH, 0.22 * top + 0.9), layout="constrained")
        for ax, kind, color, title in ((axes[0], "tactic", fg.color_for(0), "Tactics"),
                                       (axes[1], "premise", fg.color_for(1), "Cited lemmas")):
            d = sub[sub["kind"] == kind].nsmallest(top, "rank").iloc[::-1]
            y = np.arange(len(d))
            vals = d[["lift", "lift_lo", "lift_hi"]].to_numpy(dtype=float)
            hi_all = np.nanmax(vals) if len(d) and np.isfinite(vals).any() else 2.0
            floor = min(0.25, 2 ** np.floor(np.log2(max(np.nanmin(vals[vals > 0]) if (vals > 0).any() else 0.25,
                                                            1 / 64))))
            for yi, (lift, lo, hi), s in zip(y, vals, d["lift_lo"].to_numpy(dtype=float) > 1):
                c = color if s else fg.OTHER
                lo_c, hi_c = max(lo, floor), max(hi, floor)
                ax.plot([lo_c, hi_c], [yi, yi], color=c, lw=1.2, solid_capstyle="butt")
                ax.plot([max(lift, floor)], [yi], "<" if lift < floor else "o", color=c, ms=3.5)
            ax.axvline(1.0, color=fg.TEXT_SECONDARY, lw=0.8, ls="--")
            ax.set_yticks(y, [_short(f) for f in d["feature"]])
            ax.set_xscale("log", base=2)
            top_x = 2 ** np.ceil(np.log2(max(hi_all, 2.0) * 1.05))
            ax.set_xlim(floor / 1.15, top_x)
            ticks = [2.0 ** k for k in range(int(np.log2(floor)), int(np.log2(top_x)) + 1)]
            ax.set_xticks(ticks, [f"{t:g}" if t >= 1 else f"1/{int(round(1 / t))}" for t in ticks])
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.set_xlabel("lift at area switches")
            ax.set_title(title, loc="left", fontsize="medium")
            ax.grid(axis="x", color=fg.GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if not len(d):
                ax.text(0.5, 0.5, "no feature reaches the count threshold", transform=ax.transAxes,
                        ha="center", va="center", fontsize="small", color=fg.TEXT_SECONDARY)
    return fig


def _short(s: str, n: int = 30) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# --------------------------------------------------------------------------- report


def _area_name(source: str, code: Any) -> str:
    if source == "judge":
        try:
            from . import msc

            return f"{code} {msc.title(str(code))}"
        except Exception:  # unknown code: show it bare
            return str(code)
    return f"macrostate {code}"


def _fmt_row(r: pd.Series) -> str:
    return (f"| `{r['feature']}` | {r['n_switch']} of {r['n_moves']} | {r['switch_rate_given_feature']:.2f} | "
            f"{r['lift']:.2f} [{r['lift_lo']:.2f}, {r['lift_hi']:.2f}] | {r['q_value']:.3f} |")


def render_report(table: pd.DataFrame, summary: dict, top: int = 15, top_pairs: int = 5, per_pair: int = 3) -> str:
    lines = ["# Bridge steps: tactics and lemmas at area switches", "",
             "A move is two consecutive labelled steps of a proof, credited to the tactic at the first. "
             "Lift = P(switch | feature) / P(switch); for an area pair the base is moves out of the first area. "
             f"Intervals are {int(summary['settings']['ci'] * 100)}% bootstraps over proofs "
             f"({summary['settings']['n_boot']} resamples); q is the Benjamini-Hochberg adjusted bootstrap p of "
             f"lift <= 1. Features need at least {summary['settings']['min_count']} moves overall, and "
             f"{summary['settings']['min_count']} switches within a pair.", ""]
    for src, info in summary["sources"].items():
        lines += [f"## Areas from {src} ({info['areas_from']})", ""]
        if not info.get("n_moves"):
            lines += [f"No moves: {info.get('note', 'no consecutive labelled steps')}.", ""]
            continue
        lines += [f"{info['n_proofs']} proofs, {info['n_moves']} moves, {info['n_switches']} switches "
                  f"(switch rate {info['switch_rate']:.3f}); {info['n_area_pairs']} area pairs seen.", ""]
        sub = table[table["area_source"] == src]
        for kind, label in (("tactic", "Tactics"), ("premise", "Cited lemmas")):
            d = sub[(sub["scope"] == "all") & (sub["kind"] == kind)].nsmallest(top, "rank")
            lines += [f"### {label}, all switches", ""]
            if d.empty:
                lines += ["None reach the count threshold.", ""]
                continue
            lines += ["| feature | switches of moves | switch rate | lift [CI] | q |", "|---|---|---|---|---|"]
            lines += [_fmt_row(r) for _, r in d.iterrows()] + [""]
        pairs = sub[sub["scope"] == "pair"].drop_duplicates(["from_area", "to_area"]).nlargest(top_pairs, "n_switch_total")
        if len(pairs):
            lines += ["### Busiest area pairs", ""]
            for _, pr in pairs.iterrows():
                d = sub[(sub["scope"] == "pair") & (sub["from_area"] == pr["from_area"]) &
                        (sub["to_area"] == pr["to_area"])]
                lines += [f"**{_area_name(src, pr['from_area'])} → {_area_name(src, pr['to_area'])}** "
                          f"({pr['n_switch_total']} switches of {pr['n_moves_total']} moves out)", ""]
                for kind, label in (("tactic", "tactics"), ("premise", "lemmas")):
                    dk = d[d["kind"] == kind].nsmallest(per_pair, "rank")
                    if len(dk):
                        lines.append(f"- {label}: " + ", ".join(
                            f"`{r['feature']}` {r['lift']:.1f} [{r['lift_lo']:.1f}, {r['lift_hi']:.1f}]"
                            for _, r in dk.iterrows()))
                lines.append("")
    lines += ["Cited lemmas are read from the tactic text (see `markov_lean.bridges`), not from LeanDojo's "
              "premise annotations, which the steps table does not keep.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- run


def write_table(df: pd.DataFrame, path_stem: Path) -> list[Path]:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    out = [path_stem.with_suffix(".csv")]
    df.to_csv(out[0], index=False)
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return out
    pq = path_stem.with_suffix(".parquet")
    df.to_parquet(pq, index=False)
    return [pq] + out


def _first(*paths: Path | None) -> Path | None:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def run(run_dir: str | Path, out_dir: str | Path, steps_path: str | Path | None = None,
        labels_path: str | Path | None = None, msm_dir: str | Path | None = None, sources: Iterable[str] = SOURCES,
        judge: str | None = None, min_confidence: float = 0.0, min_membership: float = 0.0, n_boot: int = 1000,
        seed: int = 0, min_count: int = 5, min_pair_switches: int = 10, ci: float = 0.95, top: int = 15,
        figure: bool = True, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    run_dir, out = Path(run_dir), Path(out_dir)
    sp = _first(Path(steps_path) if steps_path else None, run_dir / "judge" / "sample_steps.parquet",
                run_dir / "judge" / "sample_steps.csv", run_dir / "steps.parquet")
    if sp is None:
        raise FileNotFoundError(f"no steps table: pass --steps, or run the pilot so {run_dir}/judge/sample_steps.parquet exists")
    steps = read_table(sp)
    summary: dict[str, Any] = {
        "inputs": {"run": str(run_dir), "steps": str(sp)},
        "settings": {"n_boot": n_boot, "seed": seed, "min_count": min_count, "min_pair_switches": min_pair_switches,
                     "ci": ci, "min_confidence": min_confidence, "min_membership": min_membership},
        "sources": {},
    }
    all_moves, tables = [], []
    for src in sources:
        if src == "judge":
            lp = _first(Path(labels_path) if labels_path else None, run_dir / "judge" / "labels.parquet",
                        run_dir / "judge" / "labels.csv")
            if lp is None:
                summary["sources"][src] = {"areas_from": "missing", "note": "no judge labels yet"}
                say("bridges: no judge labels found, skipping the judge source")
                continue
            areas, used = judge_areas(read_table(lp), judge, min_confidence)
            summary["inputs"]["labels"] = str(lp)
        elif src == "macrostate":
            md = Path(msm_dir) if msm_dir else run_dir / "msm"
            ap = _first(md / "assignments.parquet", md / "assignments.csv")
            if ap is None:
                summary["sources"][src] = {"areas_from": "missing", "note": f"no {md}/assignments.parquet"}
                say(f"bridges: no {md}/assignments.parquet, skipping the macrostate source")
                continue
            areas, used = macrostate_areas(read_table(ap), min_membership), "PCCA+ macrostates"
            summary["inputs"]["assignments"] = str(ap)
        else:
            raise ValueError(f"unknown area source {src!r}; choose from {SOURCES}")
        moves = build_moves(steps, areas, src)
        info: dict[str, Any] = {"areas_from": used, "n_labelled_steps": int(len(areas))}
        if moves.empty:
            info["note"] = "no consecutive labelled steps match the steps table"
            summary["sources"][src] = info
            say(f"bridges: {src}: no moves")
            continue
        tab = lift_table(moves, n_boot=n_boot, seed=seed, min_count=min_count,
                         min_pair_switches=min_pair_switches, ci=ci)
        info.update({
            "n_proofs": int(moves["proof_id"].nunique()), "n_moves": int(len(moves)),
            "n_switches": int(moves["switch"].sum()), "switch_rate": float(moves["switch"].mean()),
            "n_area_pairs": int(moves.loc[moves["switch"], ["from_area", "to_area"]].drop_duplicates().shape[0]),
            "n_moves_with_premise": int((moves["premises"].str.len() > 0).sum()),
            "n_features_ranked": {k: int(((tab["scope"] == "all") & (tab["kind"] == k)).sum()) for k in KINDS},
        })
        summary["sources"][src] = info
        say(f"bridges: {src}: {info['n_proofs']} proofs, {info['n_moves']} moves, {info['n_switches']} switches; "
            f"{len(tab)} rows ranked")
        all_moves.append(moves)
        tables.append(tab)
    moves = pd.concat(all_moves, ignore_index=True) if all_moves else build_moves(steps, pd.DataFrame(), "none")
    table = pd.concat(tables, ignore_index=True) if tables else lift_table(pd.DataFrame())
    files = write_table(table, out / "bridges") + write_table(moves, out / "moves")
    (out / "REPORT.md").write_text(render_report(table, summary, top=top))
    files.append(out / "REPORT.md")
    if figure and not table.empty:
        src = next((s for s in ("judge", "macrostate") if (table["area_source"] == s).any()), None)
        try:
            from .figures import save_pdf

            fig = bridge_figure(table, src, top=min(top, 12))
            files.append(save_pdf(fig, out / "bridges.pdf"))
            summary["figure_source"] = src
        except ImportError as exc:
            say(f"bridges: figure skipped ({exc}); install the figures extra")
    summary["outputs"] = [str(f) for f in files]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    say(f"bridges: wrote {out}/REPORT.md and {len(files)} files")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="markov-lean bridges", description=__doc__.split("\n\n")[0])
    p.add_argument("--run", default="results/pilot3", help="pilot output folder (judge/, msm/)")
    p.add_argument("--out", default="results/bridges")
    p.add_argument("--steps", default=None, help="steps table; default <run>/judge/sample_steps.parquet")
    p.add_argument("--labels", default=None, help="labels table; default <run>/judge/labels.parquet")
    p.add_argument("--msm", default=None, help="model folder with assignments.parquet; default <run>/msm")
    p.add_argument("--source", choices=["judge", "macrostate", "both"], default="both")
    p.add_argument("--judge", default=None, help="use this judge's labels instead of the consensus")
    p.add_argument("--min-confidence", type=float, default=0.0, help="drop judge labels below this confidence")
    p.add_argument("--min-membership", type=float, default=0.0, help="drop macrostate assignments below this membership")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-count", type=int, default=5,
                   help="moves a feature needs to be ranked overall, and switches within an area pair")
    p.add_argument("--min-pair-switches", type=int, default=10, help="switches an area pair needs to be analysed")
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--top", type=int, default=15, help="rows per table in REPORT.md")
    p.add_argument("--no-figure", action="store_true")
    a = p.parse_args(argv)
    sources = SOURCES if a.source == "both" else (a.source,)
    try:
        summary = run(a.run, a.out, a.steps, a.labels, a.msm, sources, a.judge, a.min_confidence, a.min_membership,
                      a.n_boot, a.seed, a.min_count, a.min_pair_switches, a.ci, a.top, figure=not a.no_figure)
    except FileNotFoundError as exc:
        print(f"bridges: {exc}", file=sys.stderr)
        return 2
    return 0 if any(v.get("n_moves") for v in summary["sources"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
