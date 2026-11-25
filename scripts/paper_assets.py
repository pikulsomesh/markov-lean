"""Build the paper's figures and tables from real run outputs.

Each asset is built only when its inputs exist; the rest are listed as skipped,
so this can run after any stage of REPRODUCE.md.  Nothing here calls an API.

    python scripts/paper_assets.py                       # defaults below
    python scripts/paper_assets.py --pilot results/pilot3 --strict

Writes ``paper/figures/<name>.pdf`` (``\\maybefigure`` in the paper uses these
instead of the synthetic previews) and ``paper/tables/<name>.tex`` (table bodies
to paste in place of the ``\\tbd`` rows, or ``\\input``).

| asset | inputs |
|---|---|
| figures/implied_timescales.pdf | ``<msm>/implied_timescales.csv`` |
| figures/chapman_kolmogorov.pdf | ``<msm>/ck_test.csv`` |
| figures/transition_matrix.pdf | ``<msm>/coarse_T.npy`` |
| figures/judge_confusion.pdf | ``<pilot>/msm/assignments.parquet`` + ``<pilot>/judge/labels.parquet`` |
| figures/area_flow.pdf | ``<pilot>/judge/labels.parquet`` (consensus step labels) |
| figures/search_ablation.pdf | ``<search>/fig_pass_at_1.pdf`` from ``markov-lean search-report`` |
| tables/data.tex | ``<pilot>/pilot_report.json``, ``data/processed/*.parquet``, ``<synthetic>/pilot_report.json`` |
| tables/areas.tex | pilot assignments and labels (+ the synthetic run) |
| tables/search.tex | ``<search>/pass_at_k.tex`` |

``<msm>`` is ``results/full/msm`` when it exists, else ``<pilot>/msm``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ["proof_id", "step"]


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _consensus_steps(labels_path: Path) -> pd.DataFrame:
    from markov_lean.pilot import consensus_labels

    labels = pd.read_parquet(labels_path)
    labels["proof_id"] = labels["proof_id"].astype(str)
    cons, _ = consensus_labels(labels)
    cons = cons[cons["step"] >= 0]
    return cons[KEY + ["msc"]].astype({"step": "int64"})


def _labelled(pilot: Path) -> pd.DataFrame | None:
    a, lab = pilot / "msm" / "assignments.parquet", pilot / "judge" / "labels.parquet"
    if not (a.exists() and lab.exists()):
        return None
    assign = pd.read_parquet(a)
    assign["proof_id"] = assign["proof_id"].astype(str)
    return assign.merge(_consensus_steps(lab), on=KEY, how="inner")


def _tex_num(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return r"\tbd"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", "{,}")
    return f"{float(x):.3f}"


def _count(path: Path) -> tuple[int | None, int | None]:
    """(proofs, steps) in a processed steps parquet; steps is None for statement-only files."""
    if not path.exists():
        return None, None
    df = pd.read_parquet(path, columns=["proof_id", "step"])
    proofs = int(df["proof_id"].nunique())
    return proofs, (int(len(df)) if (df["step"] >= 0).any() else None)


# --------------------------------------------------------------------------- figures


def figure_msm(msm: Path, out: Path, F) -> dict[str, str]:
    done: dict[str, str] = {}
    its_p, ck_p, T_p = msm / "implied_timescales.csv", msm / "ck_test.csv", msm / "coarse_T.npy"
    if its_p.exists():
        its = pd.read_csv(its_p)
        piv = its.pivot(index="lag", columns="index", values="timescale").sort_index()
        piv = piv[[c for c in piv.columns if int(c) < 3]]  # the three slowest processes
        F.save_pdf(F.implied_timescales(piv.index.to_numpy(), piv.to_numpy()), out / "implied_timescales.pdf")
        done["implied_timescales"] = _rel(its_p)
    if ck_p.exists():
        ck = pd.read_csv(ck_p)
        lags = sorted(ck["lag"].unique())
        m = int(max(ck["from_set"].max(), ck["to_set"].max())) + 1
        pred = np.zeros((len(lags), m, m))
        est = np.zeros_like(pred)
        for li, L in enumerate(lags):
            sub = ck[ck["lag"] == L]
            i, j = sub["from_set"].to_numpy(int), sub["to_set"].to_numpy(int)
            pred[li, i, j] = sub["predicted"].to_numpy()
            est[li, i, j] = sub["estimated"].to_numpy()
        labels = [f"M{k + 1}" for k in range(m)]
        F.save_pdf(F.chapman_kolmogorov(lags, pred, est, labels=labels), out / "chapman_kolmogorov.pdf")
        done["chapman_kolmogorov"] = _rel(ck_p)
    if T_p.exists():
        T = np.load(T_p)  # PCCA+ coarse matrices can hold tiny negative entries, so no stochastic check
        labels = [f"M{k + 1}" for k in range(T.shape[0])]
        F.save_pdf(F.transition_heatmap(T, labels, annotate=T.shape[0] <= 10, check_stochastic=False),
                   out / "transition_matrix.pdf")
        done["transition_matrix"] = _rel(T_p)
    return done


def figure_judges(pilot: Path, out: Path, F) -> dict[str, str]:
    done: dict[str, str] = {}
    lab = pilot / "judge" / "labels.parquet"
    m = _labelled(pilot)
    if m is not None and len(m):
        F.save_pdf(F.judge_cluster_confusion([f"M{int(x) + 1}" for x in m["macrostate"]], m["msc"]),
                   out / "judge_confusion.pdf")
        done["judge_confusion"] = _rel(lab)
    if lab.exists():
        cons = _consensus_steps(lab)
        tr = F.area_transitions(cons, area_col="msc")
        if len(tr):
            F.save_pdf(F.area_flow(tr, top=10), out / "area_flow.pdf")
            done["area_flow"] = _rel(lab)
    return done


def figure_search(search: Path, out: Path) -> dict[str, str]:
    src = search / "fig_pass_at_1.pdf"
    if not src.exists():
        return {}
    shutil.copyfile(src, out / "search_ablation.pdf")
    return {"search_ablation": _rel(src)}


# --------------------------------------------------------------------------- tables


def table_data(pilot: Path, synthetic: Path, processed: Path) -> str | None:
    rep_p = pilot / "pilot_report.json"
    if not rep_p.exists():
        return None
    d = json.loads(rep_p.read_text()).get("data", {})
    wb = _count(processed / "workbook_train.parquet")
    syn = json.loads((synthetic / "pilot_report.json").read_text()).get("data", {}) \
        if (synthetic / "pilot_report.json").exists() else {}
    mini = _count(processed / "minif2f_test.parquet")[0]
    put = _count(processed / "putnam_test.parquet")[0]
    rows = [
        ("LeanDojo Benchmark 4", d.get("n_proofs"), d.get("n_steps"), "transitions"),
        ("Lean Workbook", wb[0], wb[1], "judge validation"),
        ("Synthetic (planted)", syn.get("n_proofs"), syn.get("n_steps"), "sanity check"),
        None,
        ("miniF2F (test)", mini, "--", "search"),
        ("PutnamBench", put, "--", "search"),
    ]
    lines = ["% generated by scripts/paper_assets.py from " + _rel(rep_p)]
    for r in rows:
        if r is None:
            lines.append(r"\midrule")
            continue
        name, p, s, use = r
        lines.append(f"{name:<22}& {_tex_num(p)} & {s if s == '--' else _tex_num(s)} & {use} \\\\")
    return "\n".join(lines) + "\n"


def table_areas(pilot: Path, synthetic: Path, seed: int = 0) -> str | None:
    m = _labelled(pilot)
    if m is None or not len(m):
        return None
    from markov_lean import evaluate as ev

    steps_p = pilot / "judge" / "sample_steps.parquet"
    macro = ev.agreement(m["macrostate"], m["msc"])
    folder = {"nmi": None, "ari": None}
    if steps_p.exists():
        st = pd.read_parquet(steps_p, columns=KEY + ["file_path"])
        st["proof_id"] = st["proof_id"].astype(str)
        f = m.merge(st, on=KEY, how="left")
        folder = ev.agreement(ev.folder_labels(f), f["msc"])
    # random partition with the macrostates' sizes: shuffle the macrostate column
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(m["macrostate"].to_numpy())
    rand = ev.agreement(shuffled, m["msc"])
    syn = {"nmi": None, "ari": None}
    if (synthetic / "pilot_report.json").exists():
        syn = json.loads((synthetic / "pilot_report.json").read_text()).get("eval", {}).get("macrostates", syn)
    k = int(m["macrostate"].max()) + 1
    rows = [(f"Macrostates ($m = {k}$)", macro), ("Mathlib folder", folder), ("Random partition", rand), None,
            ("Synthetic, planted areas", syn)]
    lines = [f"% generated by scripts/paper_assets.py; {len(m):,} steps with a consensus label"]
    for r in rows:
        if r is None:
            lines.append(r"\midrule")
            continue
        name, v = r
        lines.append(f"{name:<25}& {_tex_num(v.get('nmi'))} & {_tex_num(v.get('ari'))} \\\\")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main


def build(pilot: Path, msm: Path | None, synthetic: Path, search: Path, processed: Path, out: Path,
          figures: bool = True) -> tuple[dict[str, str], list[str]]:
    fig_dir, tab_dir = out / "figures", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    msm = msm or (ROOT / "results" / "full" / "msm" if (ROOT / "results" / "full" / "msm").exists()
                  else pilot / "msm")
    built: dict[str, str] = {}
    if figures:
        from markov_lean import figures as F

        built.update({f"figures/{k}.pdf": v for k, v in figure_msm(msm, fig_dir, F).items()})
        built.update({f"figures/{k}.pdf": v for k, v in figure_judges(pilot, fig_dir, F).items()})
        built.update({f"figures/{k}.pdf": v for k, v in figure_search(search, fig_dir).items()})
    for name, text in (("data", table_data(pilot, synthetic, processed)), ("areas", table_areas(pilot, synthetic))):
        if text is not None:
            (tab_dir / f"{name}.tex").write_text(text)
            built[f"tables/{name}.tex"] = "computed"
    if (search / "pass_at_k.tex").exists():
        shutil.copyfile(search / "pass_at_k.tex", tab_dir / "search.tex")
        built["tables/search.tex"] = _rel(search / "pass_at_k.tex")
    wanted = [f"figures/{n}.pdf" for n in ("implied_timescales", "chapman_kolmogorov", "transition_matrix",
                                          "judge_confusion", "area_flow", "search_ablation")]
    wanted += [f"tables/{n}.tex" for n in ("data", "areas", "search")]
    skipped = [w for w in wanted if w not in built and (figures or not w.startswith("figures/"))]
    return built, skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pilot", default=str(ROOT / "results" / "pilot3"))
    p.add_argument("--msm", default=None, help="model folder (default results/full/msm, else <pilot>/msm)")
    p.add_argument("--synthetic", default=str(ROOT / "results" / "pilot_synthetic"))
    p.add_argument("--search", default=str(ROOT / "results" / "search_eval" / "report"))
    p.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    p.add_argument("--out", default=str(ROOT / "paper"))
    p.add_argument("--no-figures", action="store_true", help="tables only (no matplotlib needed)")
    p.add_argument("--strict", action="store_true", help="exit 1 if any asset was skipped")
    a = p.parse_args(argv)
    built, skipped = build(Path(a.pilot), Path(a.msm) if a.msm else None, Path(a.synthetic), Path(a.search),
                           Path(a.processed), Path(a.out), figures=not a.no_figures)
    for k, v in built.items():
        print(f"built   {k}  <- {v}")
    for k in skipped:
        print(f"skipped {k}  (inputs not found)")
    return 1 if (a.strict and skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
