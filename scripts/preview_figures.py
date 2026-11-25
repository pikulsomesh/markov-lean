"""Render every paper figure from made-up numbers, to preview the layout.

Nothing here comes from a real run. The numbers are shaped to look plausible
so the page layout, sizes and captions can be judged before results exist.
Each figure carries a small "synthetic preview" stamp, and the paper prints a
red note under any figure taken from paper/figures/preview/. Real figures
saved to paper/figures/ take precedence automatically.

    python scripts/preview_figures.py            # writes paper/figures/preview/*.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from markov_lean import figures as F

AREAS = ["05", "11", "13", "26", "54", "60"]  # MSC 2020 top-level codes
STAMP = "synthetic preview"


def stamp(fig):
    import matplotlib

    with matplotlib.rc_context(F.paper_rc()):  # text fonts are fixed when the text is made
        fig.text(0.995, 0.005, STAMP, ha="right", va="bottom", fontsize=5, color="#999999")
    return fig


def sticky_matrix(m: int, p_stay: float, rng) -> np.ndarray:
    off = rng.dirichlet(np.ones(m - 1), size=m) * (1 - p_stay)
    T = np.zeros((m, m))
    for i in range(m):
        T[i, np.arange(m) != i] = off[i]
        T[i, i] = p_stay
    return T


def make_all(rng) -> list[tuple[str, object]]:
    out = []

    # implied timescales that level off after a few steps of lag
    lags = np.arange(1, 11)
    t_inf = np.array([22.0, 9.0, 4.5])
    ts = t_inf * (1 - 0.55 * np.exp(-lags[:, None] / 2.5))
    out.append(("implied_timescales", F.implied_timescales(lags, ts, ts * 0.88, ts * 1.12)))

    # macrostate transition matrix at the chosen lag
    T4 = sticky_matrix(4, 0.78, rng)
    out.append(("transition_matrix", F.transition_heatmap(T4, ["M1", "M2", "M3", "M4"])))

    # Chapman-Kolmogorov: model T^k against noisy direct estimates
    T3 = sticky_matrix(3, 0.85, rng)
    ks = np.arange(1, 6)
    pred = np.stack([np.linalg.matrix_power(T3, k) for k in ks])
    est = np.clip(pred + rng.normal(0, 0.02, pred.shape), 0, 1)
    err = np.full_like(est, 0.03)
    out.append(("chapman_kolmogorov",
                F.chapman_kolmogorov(ks * 2, pred, est, err, labels=["M1", "M2", "M3"])))

    # macrostates against judge MSC codes: mostly diagonal, with confusion
    n = 3000
    truth = rng.integers(0, len(AREAS), n)
    noisy = np.where(rng.random(n) < 0.35, rng.integers(0, len(AREAS), n), truth)
    clusters = np.array([f"M{i + 1}" for i in noisy])
    judge = np.array(AREAS)[truth]
    out.append(("judge_confusion", F.judge_cluster_confusion(clusters, judge)))

    # within-proof moves between areas from a sticky chain
    T6 = sticky_matrix(len(AREAS), 0.85, rng)
    rows = []
    for pid in range(400):
        a = rng.integers(len(AREAS))
        for step in range(int(rng.integers(8, 30))):
            rows.append((pid, step, AREAS[a]))
            a = rng.choice(len(AREAS), p=T6[a])
    assign = pd.DataFrame(rows, columns=["proof_id", "step", "macrostate"])
    out.append(("area_flow", F.area_flow(F.area_transitions(assign), top=10)))

    # search: problems solved per seed, two benchmarks, three priors
    base = {"miniF2F": 48.0, "ProofNet": 21.0}
    lift = {"LM prior": 0.0, "+ area prior": 3.0, "+ random partition": 0.5}
    res = [(m, b, base[b] + d + rng.normal(0, 1.2))
           for b in base for m, d in lift.items() for _ in range(5)]
    df = pd.DataFrame(res, columns=["method", "benchmark", "solved"])
    out.append(("search_ablation", F.search_ablation(df, group_col="benchmark")))

    # case studies: a few short proofs that cross areas, from a synthetic judged sample
    from markov_lean import cases as C

    picks = C.select(*C.synthetic_case_inputs(seed=0))
    out.append((C.FIGURE_NAME, C.case_strips(picks.steps, max_steps=picks.config.max_steps)))

    return [(name, stamp(fig)) for name, fig in out]


def write_tables(outdir) -> list[Path]:
    """The LaTeX tables that go with the previews (the case-study table)."""
    from markov_lean import cases as C

    picks = C.select(*C.synthetic_case_inputs(seed=0))
    path = Path(outdir) / f"{C.TABLE_NAME}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(C.latex_table(picks.picked))
    return [path]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=Path(__file__).resolve().parents[1] / "paper" / "figures" / "preview")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    paths = F.save_all(make_all(np.random.default_rng(args.seed)), args.out) + write_tables(args.out)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
