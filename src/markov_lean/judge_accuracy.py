"""Score MSC judges against the gold set (`markov-lean judge-accuracy`).

The gold set (``gold.py``, ``data/gold/msc_gold.csv``) gives each benchmark
problem an MSC top-level code from the benchmark's own metadata. This module
takes one or more ``labels`` tables (``docs/schema.md``; the API judges', the
local Ollama judge's, or any file with the same columns) and reports, per judge:

* **coverage**: the share of gold problems the judge labelled;
* on **strict** problems (one known class): accuracy, macro-F1 over the gold
  classes, balanced accuracy (mean per-class recall), per-class precision,
  recall and F1, and the confusion matrix;
* on **strict and family** problems: family accuracy, the share whose code is
  in the accepted group;
* the **majority-class baseline** (always answer the commonest gold class) on
  the same problems, for scale.

The label scored for a problem is the judge's proof-level label (``step -1``),
the area of the theorem's statement. A judge that only gave step labels is
scored on its label for the first step instead, and the report says how many
problems that was.

Intervals are 95% percentile bootstraps over problems (``--n-boot``, seed
``--seed``). Problems the judge did not label are left out, not counted wrong;
coverage shows how many that was.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from markov_lean import gold as G
from markov_lean import msc

PROOF_LEVEL_STEP = -1
OTHER = "other"


# ---------------------------------------------------------------- inputs

def read_labels(paths: Sequence[str | Path]) -> pd.DataFrame:
    parts = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix in (".jsonl", ".json"):
            df = pd.read_json(p, lines=p.suffix == ".jsonl", dtype=False)
        else:
            df = pd.read_csv(p, dtype=str)
        missing = {"proof_id", "step", "judge", "msc"} - set(df.columns)
        if missing:
            raise ValueError(f"{p} is missing columns {sorted(missing)}")
        parts.append(df[["proof_id", "step", "judge", "msc"]])
    return clean_labels(pd.concat(parts, ignore_index=True))


def clean_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Integer steps, string ids, and codes cut to the two-digit level (None when invalid)."""
    labels = labels.copy()
    labels["step"] = pd.to_numeric(labels.step, errors="coerce").fillna(PROOF_LEVEL_STEP).astype(int)
    labels["msc"] = labels.msc.map(msc.normalize)
    labels["proof_id"] = labels.proof_id.astype(str)
    labels["judge"] = labels.judge.astype(str)
    return labels


def problem_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """One label per (judge, problem): the proof-level label, else the first step's.

    Columns: judge, proof_id, pred, from_step (False for proof-level labels).
    Invalid codes become ``None`` and are scored as wrong.
    """
    out = []
    for (judge, pid), g in labels.groupby(["judge", "proof_id"], sort=False):
        proof = g[g.step == PROOF_LEVEL_STEP]
        if len(proof):
            out.append((judge, pid, proof.msc.iloc[-1], False))
        else:
            first = g.sort_values("step").iloc[0]
            out.append((judge, pid, first.msc, True))
    return pd.DataFrame(out, columns=["judge", "proof_id", "pred", "from_step"])


# ---------------------------------------------------------------- metrics

def _f1_parts(y: np.ndarray, p: np.ndarray, classes: Sequence[str], w: np.ndarray | None = None):
    """Per-class precision, recall and F1 (weights ``w`` give bootstrap resamples)."""
    w = np.ones(len(y)) if w is None else w
    prec, rec, f1 = [], [], []
    for c in classes:
        tp = float(w[(y == c) & (p == c)].sum())
        fp = float(w[(y != c) & (p == c)].sum())
        fn = float(w[(y == c) & (p != c)].sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else float("nan")
        prec.append(pr)
        rec.append(rc)
        f1.append(2 * pr * rc / (pr + rc) if (pr + rc) and rc == rc else 0.0)
    return np.array(prec), np.array(rec), np.array(f1)


def strict_metrics(y: np.ndarray, p: np.ndarray, classes: Sequence[str], w: np.ndarray | None = None) -> dict:
    w = np.ones(len(y)) if w is None else w
    present = [c for c in classes if w[y == c].sum() > 0]
    _, rec, f1 = _f1_parts(y, p, present, w)
    n = w.sum()
    return {
        "accuracy": float(w[y == p].sum() / n) if n else float("nan"),
        "macro_f1": float(np.mean(f1)) if present else float("nan"),
        "balanced_accuracy": float(np.nanmean(rec)) if present else float("nan"),
    }


def _ci(samples: list[float]) -> list[float]:
    a = np.asarray([s for s in samples if s == s])
    if a.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def _boot_weights(n: int, n_boot: int, rng: np.random.Generator):
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n, n), minlength=n).astype(float)


def confusion(y: np.ndarray, p: np.ndarray, classes: Sequence[str]) -> pd.DataFrame:
    """Rows: gold classes. Columns: the gold classes, then ``other`` for any other code
    (or an invalid answer)."""
    cols = list(classes) + [OTHER]
    pc = np.where(np.isin(p, list(classes)), p, OTHER)
    m = pd.DataFrame(0, index=list(classes), columns=cols)
    for a, b in zip(y, pc):
        m.loc[a, b] += 1
    m.index.name = "gold"
    m.columns.name = "judge"
    return m


def score_judge(pl: pd.DataFrame, gold: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Score one judge's problem labels (from :func:`problem_labels`) against the gold table."""
    scored = gold[gold.status != "excluded"]
    strict_all = scored[scored.status == "strict"]
    classes = sorted(strict_all.msc.unique())
    m = scored.merge(pl, left_on="id", right_on="proof_id", how="inner")
    m["pred"] = [x if isinstance(x, str) and x else None for x in m.pred]
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {
        "n_gold": int(len(scored)),
        "n_labelled": int(len(m)),
        "coverage": float(len(m) / len(scored)) if len(scored) else float("nan"),
        "n_from_step_label": int(m.from_step.sum()),
        "n_invalid_code": int(sum(not isinstance(x, str) for x in m.pred)),
    }

    s = m[m.status == "strict"]
    y = s.msc.to_numpy(dtype=object)
    p = np.array([x if isinstance(x, str) else "" for x in s.pred], dtype=object)
    st: dict[str, Any] = {"n": int(len(s)), "n_gold": int(len(strict_all)), "classes": classes}
    if len(s):
        st.update(strict_metrics(y, p, classes))
        majority = strict_all.msc.value_counts().index[0]
        st["majority_class"] = majority
        st["majority_baseline_accuracy"] = float((y == majority).mean())
        boots: dict[str, list[float]] = {"accuracy": [], "macro_f1": [], "balanced_accuracy": []}
        per_class_rec: dict[str, list[float]] = {c: [] for c in classes}
        for w in _boot_weights(len(s), n_boot, rng):
            for k, v in strict_metrics(y, p, classes, w).items():
                boots[k].append(v)
            _, rec, _ = _f1_parts(y, p, classes, w)
            for c, r in zip(classes, rec):
                per_class_rec[c].append(r)
        for k in boots:
            st[f"{k}_ci"] = _ci(boots[k])
        prec, rec, f1 = _f1_parts(y, p, classes)
        st["per_class"] = {
            c: {"n": int((y == c).sum()), "title": msc.MSC2020_TOP[c], "precision": float(prec[i]),
                "recall": float(rec[i]), "recall_ci": _ci(per_class_rec[c]), "f1": float(f1[i])}
            for i, c in enumerate(classes) if (y == c).sum()
        }
        cm = confusion(y, p, classes)
        st["confusion"] = {"rows": list(cm.index), "cols": list(cm.columns), "counts": cm.to_numpy().tolist()}
        # what "other" was, so a judge that answers 26 for every number-theory problem is visible
        st["other_codes"] = {k: int(v) for k, v in
                             pd.Series([x or "invalid" for x in p[~np.isin(p, classes)]]).value_counts().items()}
        st["by_source"] = {src: {"n": int(len(g)), "accuracy": float((g.msc == g.pred).mean())}
                           for src, g in s.groupby("source")}
    out["strict"] = st

    fam_sets = G.family_codes(m)
    hit = np.array([isinstance(pr, str) and pr in fs for pr, fs in zip(m.pred, fam_sets)], dtype=bool)
    fa: dict[str, Any] = {"n": int(len(m)), "n_family_only": int((m.status == "family").sum())}
    if len(m):
        fa["accuracy"] = float(hit.mean())
        fa["accuracy_ci"] = _ci([float((hit * w).sum() / w.sum()) for w in _boot_weights(len(m), n_boot, rng)])
        fam = m.status == "family"
        if fam.any():
            fa["family_only_accuracy"] = float(hit[fam.to_numpy()].mean())
        fa["by_rule"] = {r: {"n": int(len(g)), "accuracy": float(hit[g.index.to_numpy()].mean())}
                         for r, g in m.reset_index(drop=True).groupby("rule")}
    out["family"] = fa
    return out


def report(labels: pd.DataFrame, gold: pd.DataFrame, judges: Sequence[str] | None = None,
           n_boot: int = 2000, seed: int = 0, eval_split: str = "all") -> dict:
    """Score every judge. ``eval_split`` ("all", "dev" or "test") keeps only that half of the
    fixed dev/test split (``gold.eval_split``); report on "test" for a prompt designed on "dev"."""
    gold = G.select_split(gold, eval_split)
    pl = problem_labels(clean_labels(labels))
    pl = pl[pl.proof_id.isin(set(gold.id))]
    judges = list(judges) if judges else sorted(pl.judge.unique())
    if not judges:
        raise ValueError("no judge labels match a gold problem id; were the labels made from `gold-set steps`?")
    rep: dict[str, Any] = {
        "gold": G.balance(gold),
        "settings": {"n_boot": n_boot, "seed": seed, "resampling_unit": "problem", "label": "proof-level (step -1)",
                     "eval_split": eval_split, "eval_split_seed": G.EVAL_SPLIT_SEED},
        "judges": {},
    }
    for i, j in enumerate(judges):
        rep["judges"][j] = score_judge(pl[pl.judge == j], gold, n_boot, seed + 7 * i)
    return rep


# ---------------------------------------------------------------- output

def _f(x: Any, d: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def _ci_s(ci: Sequence[float]) -> str:
    return f"[{_f(ci[0])}, {_f(ci[1])}]"


def markdown(rep: dict) -> str:
    g = rep["gold"]
    lines = [
        "# Judge accuracy on the gold MSC set",
        "",
        f"Gold set: {g['status'].get('strict', 0)} strict problems in {len(g['strict_classes'])} classes and "
        f"{g['status'].get('family', 0)} family problems (from {g['n_problems']}; "
        f"{g['status'].get('excluded', 0)} excluded). Intervals: 95% bootstrap over problems, "
        f"{rep['settings']['n_boot']} resamples."
        + _split_note(rep["settings"].get("eval_split", "all")),
        "",
        "| judge | coverage | accuracy | macro-F1 | balanced acc. | majority baseline | family acc. |",
        "|---|---|---|---|---|---|---|",
    ]
    for j, r in rep["judges"].items():
        s, fa = r["strict"], r["family"]
        lines.append(
            f"| {j} | {r['n_labelled']}/{r['n_gold']} | {_f(s.get('accuracy'))} {_ci_s(s.get('accuracy_ci', [None, None]))} "
            f"| {_f(s.get('macro_f1'))} {_ci_s(s.get('macro_f1_ci', [None, None]))} "
            f"| {_f(s.get('balanced_accuracy'))} {_ci_s(s.get('balanced_accuracy_ci', [None, None]))} "
            f"| {_f(s.get('majority_baseline_accuracy'))} "
            f"| {_f(fa.get('accuracy'))} {_ci_s(fa.get('accuracy_ci', [None, None]))} |")
    for j, r in rep["judges"].items():
        s = r["strict"]
        if not s.get("per_class"):
            continue
        lines += ["", f"## {j}", "", f"Strict problems scored: {s['n']} of {s['n_gold']}. "
                  f"Scored on a step label (no proof label): {r['n_from_step_label']}. "
                  f"Invalid codes: {r['n_invalid_code']}.", "",
                  "| class | n | precision | recall [95% CI] | F1 |", "|---|---|---|---|---|"]
        for c, v in s["per_class"].items():
            lines.append(f"| {c} {v['title']} | {v['n']} | {_f(v['precision'])} | "
                         f"{_f(v['recall'])} {_ci_s(v['recall_ci'])} | {_f(v['f1'])} |")
        if s.get("other_codes"):
            lines += ["", "Codes outside the gold classes: "
                      + ", ".join(f"{k} ({v})" for k, v in s["other_codes"].items()) + "."]
        if s.get("by_source"):
            lines += ["", "Accuracy by source: " + ", ".join(
                f"{k} {_f(v['accuracy'])} (n={v['n']})" for k, v in s["by_source"].items()) + "."]
    return "\n".join(lines) + "\n"


def _split_note(which: str) -> str:
    if which == "all":
        return ""
    return (f" Problems: the {which} half of the fixed dev/test split "
            f"(`data/gold/msc_gold_eval_split.csv`, seed {G.EVAL_SPLIT_SEED}).")


def figure(rep: dict):
    """Left: accuracy, macro-F1 and family accuracy per judge with 95% intervals, and the
    majority baseline. Right: each judge's confusion matrix on strict problems, rows
    normalized to recall."""
    from markov_lean import figures as F

    matplotlib, plt = F._plt()
    sns = F._sns()
    judges = [j for j, r in rep["judges"].items() if r["strict"].get("confusion")]
    with matplotlib.rc_context(F.paper_rc()):
        n = len(judges)
        fig = plt.figure(figsize=(F.TEXT_WIDTH, 2.6 if n <= 2 else 2.6 * ((n + 1) // 2)), layout="constrained")
        gs = fig.add_gridspec(max(1, (n + 1) // 2), 1 + min(n, 2), width_ratios=[1.0] + [1.25] * min(n, 2))
        ax = fig.add_subplot(gs[:, 0])
        metrics = [("accuracy", "Accuracy", "strict"), ("macro_f1", "Macro-\nF1", "strict"),
                   ("accuracy", "Family\naccuracy", "family")]
        width = 0.8 / max(1, len(judges))
        for k, j in enumerate(judges):
            r = rep["judges"][j]
            xs, vals, lo, hi = [], [], [], []
            for i, (key, _, part) in enumerate(metrics):
                v = r[part].get(key)
                ci = r[part].get(f"{key}_ci", [v, v])
                xs.append(i - 0.4 + width * (k + 0.5))
                vals.append(v)
                lo.append(v - ci[0])
                hi.append(ci[1] - v)
            ax.bar(xs, vals, width=width * 0.9, color=F.color_for(k), label=_short(j))
            ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=F.TEXT, elinewidth=0.7, capsize=1.5)
        base = rep["judges"][judges[0]]["strict"].get("majority_baseline_accuracy") if judges else None
        if base is not None:
            ax.axhline(base, color=F.TEXT_SECONDARY, lw=0.7, ls="--")
            ax.text(-0.45, base + 0.01, "majority class", ha="left", va="bottom",
                    fontsize="x-small", color=F.TEXT_SECONDARY)
        ax.set_xticks(range(len(metrics)), [m[1] for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score on gold problems")
        ax.legend(fontsize="x-small", loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
        cmap = F._blue_cmap()
        for k, j in enumerate(judges):
            cm = rep["judges"][j]["strict"]["confusion"]
            df = pd.DataFrame(cm["counts"], index=cm["rows"], columns=cm["cols"]).astype(float)
            norm = df.div(df.sum(axis=1).replace(0, np.nan), axis=0)
            axc = fig.add_subplot(gs[k // 2, 1 + k % 2])
            sns.heatmap(norm, ax=axc, cmap=cmap, vmin=0, vmax=1, square=True, linewidths=0.3,
                        linecolor="white", xticklabels=True, yticklabels=True, cbar=k % 2 == 1 or k == n - 1,
                        cbar_kws={"label": "Share of gold class", "shrink": 0.8})
            F._style_colorbar(axc)
            axc.set_xlabel(f"{_short(j)} label")
            axc.set_ylabel("Gold MSC class")
            axc.tick_params(length=0)
            axc.set_xticklabels(axc.get_xticklabels(), rotation=90)
            axc.set_yticklabels(axc.get_yticklabels(), rotation=0)
    return fig


def _short(judge: str) -> str:
    return judge.split("/")[-1]


def write(rep: dict, out: str | Path, figures: bool = True) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = [out / "judge_accuracy.json", out / "judge_accuracy.md"]
    written[0].write_text(json.dumps(rep, indent=1, default=str) + "\n")
    written[1].write_text(markdown(rep))
    for j, r in rep["judges"].items():
        cm = r["strict"].get("confusion")
        if cm:
            p = out / f"confusion_{j.replace('/', '_').replace(':', '_')}.csv"
            pd.DataFrame(cm["counts"], index=pd.Index(cm["rows"], name="gold"), columns=cm["cols"]).to_csv(p)
            written.append(p)
    if figures:
        try:
            fig = figure(rep)
        except ImportError as exc:  # pragma: no cover - figures extra missing
            print(f"no figure: {exc}", file=sys.stderr)
        else:
            fdir = out / "figures"
            fdir.mkdir(exist_ok=True)
            for ext in ("pdf", "png"):
                fig.savefig(fdir / f"judge_accuracy.{ext}", bbox_inches="tight")
                written.append(fdir / f"judge_accuracy.{ext}")
            import matplotlib.pyplot as plt

            plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="markov-lean judge-accuracy",
                                 description="Accuracy, macro-F1 and confusion of MSC judges on the gold set.")
    ap.add_argument("--gold", default=str(G.DEFAULT_GOLD), help="gold table (default data/gold/msc_gold.csv)")
    ap.add_argument("--labels", nargs="+", required=True, help="one or more labels tables (parquet, csv, jsonl)")
    ap.add_argument("--judges", nargs="+", default=None, help="judges to score (default: every judge found)")
    ap.add_argument("--out", default=None, help="write the report, confusion CSVs and figure here")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-split", choices=["all", *G.EVAL_SPLITS], default="all",
                    help="score only the dev or test half of the fixed dev/test split (default all); "
                         "use test for any prompt that was designed on dev")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    gold = G.read_gold(a.gold)
    rep = report(read_labels(a.labels), gold, a.judges, a.n_boot, a.seed, a.eval_split)
    print(markdown(rep))
    if a.out:
        for p in write(rep, a.out, figures=not a.no_figures):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
