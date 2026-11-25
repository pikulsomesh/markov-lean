"""Phase 2 pilot: do Markov-model macrostates over goal states line up with MSC areas?

``markov-lean pilot`` runs in two passes so that nothing spends money by itself.

1. **First pass** (no API calls): load the steps, fit the Markov state model, run
   its checks, write a judge sample and its cost estimate, and print the judge
   commands to run next.
2. **Second pass**: once ``<out>/judge/labels.parquet`` exists, the same command
   scores the macrostates against the judges' consensus MSC labels, compares
   them with the Mathlib-folder baseline, and makes the gate 1 call.

On ``data.source: synthetic`` the hidden ``true_area`` stands in for the judge
labels, so the whole chain runs end to end offline.  That is what the tests use.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import evaluate as ev
from .msm import MSMConfig, MSMResult, map_microstates
from .msm import run as run_msm

ROOT = Path(__file__).resolve().parents[2]
KEY = ["proof_id", "step"]


# --------------------------------------------------------------------------- config


def _path(p: str | Path | None, base: Path = ROOT) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else base / p


def msm_config(section: dict | None) -> MSMConfig:
    """Build an ``MSMConfig`` from the ``msm`` section, refusing unknown keys."""
    section = dict(section or {})
    known = {f.name for f in fields(MSMConfig)}
    unknown = sorted(set(section) - known)
    if unknown:
        raise ValueError(f"unknown msm settings in the pilot config: {unknown}")
    if "tokens_per_state" in section:
        section["tokens_per_state"] = tuple(section["tokens_per_state"])
    return MSMConfig(**section)


# --------------------------------------------------------------------------- data


def load_steps(data: dict) -> pd.DataFrame:
    """The steps table the pilot runs on, from processed parquet or the synthetic generator."""
    source = data.get("source", "leandojo")
    if source == "synthetic":
        from .synthetic import generate

        steps, _ = generate(**(data.get("synthetic") or {}))
        return steps
    from .data import processed_path, read_steps

    split = data.get("split", "train")
    processed = _path(data.get("processed_dir", "data/processed"))
    name = "leandojo" if source == "leandojo" else source
    path = processed_path(processed, name, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run ./scripts/fetch_data.sh first (it writes data/processed/).")
    return read_steps(path)


def sample_proofs(steps: pd.DataFrame, max_proofs: int | None, min_steps: int, seed: int) -> pd.DataFrame:
    """Keep proofs with at least ``min_steps`` steps, then a random ``max_proofs`` of them."""
    sizes = steps.groupby("proof_id").size()
    ids = sorted(sizes[sizes >= min_steps].index.astype(str))
    if max_proofs is not None and len(ids) > max_proofs:
        ids = sorted(random.Random(seed).sample(ids, max_proofs))
    keep = steps[steps["proof_id"].astype(str).isin(set(ids))]
    return keep.reset_index(drop=True)


# --------------------------------------------------------------------------- judge stage


def write_judge_sample(steps: pd.DataFrame, n: int, seed: int, out: Path, judge_config: Path) -> dict:
    """Write the proofs to judge and a cost estimate; makes no API call."""
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(steps["proof_id"].astype(str).unique())
    if n < len(ids):
        ids = sorted(random.Random(seed + 1).sample(ids, n))
    sample = steps[steps["proof_id"].astype(str).isin(set(ids))].reset_index(drop=True)
    sample_path = out / "sample_steps.parquet"
    sample.to_parquet(sample_path, index=False)
    info: dict[str, Any] = {"sample_steps": str(sample_path), "n_proofs": len(ids), "n_steps": int(len(sample))}
    try:
        from .judge import build_requests, estimate_cost, load_config, proofs_from_steps

        cfg = load_config(judge_config)
        proofs = proofs_from_steps(sample, cfg.max_steps_per_proof, cfg.max_state_chars)
        reqs, man = build_requests(proofs, cfg)
        est = estimate_cost(reqs, man, cfg)
        info["estimate"] = est.summary()
    except Exception as exc:  # the estimate is advice; the pilot goes on without it
        info["estimate"] = f"could not estimate: {exc}"
    run_dir = out / "run"
    info["commands"] = [
        f"python -m markov_lean.judge --config {judge_config} estimate --steps {sample_path}",
        f"python -m markov_lean.judge --config {judge_config} submit --steps {sample_path} --out {run_dir}",
        f"python -m markov_lean.judge --config {judge_config} collect --out {run_dir}",
        f"cp {run_dir / 'labels.parquet'} {out / 'labels.parquet'}",
    ]
    (out / "NEXT.md").write_text(
        "# Judge the pilot sample\n\n"
        f"{info['n_proofs']} proofs, {info['n_steps']} steps.\n\n"
        f"Estimate:\n\n```\n{info['estimate']}\n```\n\n"
        "Set ANTHROPIC_API_KEY in your shell first (never in a file in this repo), then:\n\n```\n"
        + "\n".join(info["commands"]) + "\n```\n\nThen rerun `markov-lean pilot` to score and make the gate call.\n")
    return info


def consensus_labels(labels: pd.DataFrame, judges: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Consensus rows of two judges, plus their agreement.  One judge: its labels as they are."""
    present = list(dict.fromkeys(labels["judge"].astype(str)))
    present = [j for j in present if j != "consensus"]
    if "consensus" in set(labels["judge"]) and not judges:
        return labels[labels["judge"] == "consensus"], {"judges": ["consensus"]}
    judges = judges or present[:2]
    if len(judges) < 2:
        return labels[labels["judge"] == judges[0]], {"judges": judges}
    from .judge import agreement, consensus

    return consensus(labels, judges[0], judges[1]), {"judges": judges, **agreement(labels, judges[0], judges[1])}


# --------------------------------------------------------------------------- scoring


def macrostate_sweep(res: MSMResult, ks: list[int]) -> dict[int, np.ndarray]:
    """PCCA+ macrostates of the fitted model for each k, one value per row of ``res.assignments``.

    The gate is called on the configured ``n_macrostates``; the sweep shows how
    agreement with the labels changes with k, so the paper can report the curve.
    """
    n_micro = res.centers.shape[0]
    out: dict[int, np.ndarray] = {}
    for k in sorted(set(int(k) for k in ks)):
        if k < 2 or k > res.msm.n_states:
            continue
        pcca = res.msm.pcca(k)
        macro, _, _ = map_microstates(n_micro, res.connected, pcca.assignments, pcca.memberships, res.centers,
                                      "nearest")
        out[k] = macro[res.microstates]
    return out


def score(res: MSMResult, steps: pd.DataFrame, labels: pd.DataFrame, label_col: str,
          topk: list[int], gate_nmi: float, sweep: list[int] | None = None) -> dict:
    """Macrostates and the folder baseline against the same labelled steps, then gate 1."""
    a = res.assignments
    lab = labels[KEY + [label_col]].copy()
    lab["proof_id"] = lab["proof_id"].astype(str)
    lab["step"] = lab["step"].astype("int64")
    # whole-proof labels (step -1) count for every step of that proof
    whole = lab[lab["step"] == -1]
    if len(whole):
        expanded = a[KEY].merge(whole[["proof_id", label_col]], on="proof_id")
        lab = pd.concat([lab[lab["step"] != -1], expanded]).drop_duplicates(KEY, keep="first")
    scored = a.merge(lab, on=KEY, how="inner")
    if scored.empty:
        raise ValueError("no labelled steps match the MSM assignments; was the judge sample taken from this run?")
    folder_tab = scored.merge(steps[KEY + ["file_path"]].assign(proof_id=lambda d: d["proof_id"].astype(str)),
                              on=KEY, how="left")
    macro = ev.agreement(scored["macrostate"], scored[label_col])
    micro = ev.agreement(scored["microstate"], scored[label_col])
    folder = ev.agreement(ev.folder_labels(folder_tab), folder_tab[label_col])
    nxt = {f"top{k}": ev.next_area_topk(a, lab, label_col, res.coarse_T, k=k, lag=res.lag) for k in topk}
    for v in nxt.values():
        v.pop("macrostate_to_label", None)
    by_k = {}
    if sweep:
        rows = a[KEY].reset_index(drop=True)
        for k, pred in macrostate_sweep(res, sweep).items():
            m = rows.assign(_pred=pred).merge(lab, on=KEY, how="inner")
            by_k[k] = ev.agreement(m["_pred"], m[label_col])
    folder_nmi = folder["nmi"] if np.isfinite(folder["nmi"]) else 0.0
    passed = bool(np.isfinite(macro["nmi"]) and macro["nmi"] >= gate_nmi and macro["nmi"] > folder_nmi)
    return {
        "label_col": label_col,
        "n_labelled_steps": int(len(scored)),
        "macrostates": macro,
        "microstates": micro,
        "folder_baseline": folder,
        "next_area": nxt,
        "macrostate_sweep": {str(k): v for k, v in by_k.items()},
        "gate1": {
            "passed": passed,
            "rule": f"macrostate NMI >= {gate_nmi} and above the folder baseline, at the configured k",
            "k": int(res.report["n_macrostates"]),
            "macrostate_nmi": macro["nmi"],
            "folder_nmi": folder["nmi"],
            "threshold": gate_nmi,
        },
    }


# --------------------------------------------------------------------------- entry point


def run(config: dict, out_dir: Path | str, steps: pd.DataFrame | None = None, verbose: bool = True) -> int:
    """Run the pilot.  Returns 0 when it finished (gate passed or labels pending), 1 when gate 1 fails."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = config.get("data") or {}
    jcfg = config.get("judge") or {}
    ecfg = config.get("eval") or {}
    seed = int(data.get("seed", 0))
    say = print if verbose else (lambda *a, **k: None)

    steps = load_steps(data) if steps is None else steps
    steps = sample_proofs(steps, data.get("max_proofs"), int(data.get("min_steps", 2)), seed)
    if steps.empty:
        raise ValueError("no proofs left after filtering; lower data.min_steps")
    lens = steps.groupby("proof_id").size()
    say(f"pilot: {len(lens)} proofs, {len(steps)} steps, median proof length {lens.median():g} steps")

    res = run_msm(steps, msm_config(config.get("msm")))
    res.save(out / "msm")
    rep = res.report
    failed = [k for k, c in rep["checks"].items() if not c["passed"]]
    say(f"msm: {rep['n_microstates']} microstates, lag {rep['lag']}, {rep['n_macrostates']} macrostates; "
        f"checks failed: {failed or 'none'}")

    report: dict[str, Any] = {
        "data": {"source": data.get("source", "leandojo"), "split": data.get("split", "train"),
                 "n_proofs": int(steps["proof_id"].nunique()), "n_steps": int(len(steps))},
        "msm": {k: rep[k] for k in ("n_microstates", "lag", "n_macrostates", "all_checks_passed")},
        "msm_checks_failed": failed,
    }
    topk = list(ecfg.get("topk", [1, 3]))
    sweep = list(ecfg.get("macrostate_sweep", []))
    gate_nmi = float(ecfg.get("gate_nmi", 0.15))

    if "true_area" in steps.columns and steps["true_area"].notna().any() and data.get("source") == "synthetic":
        report["eval"] = score(res, steps, steps.assign(proof_id=steps["proof_id"].astype(str)),
                               "true_area", topk, gate_nmi, sweep)
        report["labels"] = "true_area (synthetic)"
    else:
        judge_dir = out / "judge"
        labels_path = _path(jcfg.get("labels"), out) or judge_dir / "labels.parquet"
        if labels_path.exists():
            labels = pd.read_parquet(labels_path)
            cons, agree = consensus_labels(labels, jcfg.get("judges"))
            report["judge_agreement"] = agree
            report["eval"] = score(res, steps, cons, "msc", topk, gate_nmi, sweep)
            report["labels"] = str(labels_path)
        else:
            info = write_judge_sample(steps, int(jcfg.get("sample_proofs", 1000)), seed, judge_dir,
                                      _path(jcfg.get("config", "configs/judge.yaml")))
            report["judge"] = info
            report["gate1"] = "waiting for judge labels"
            say(f"judge sample written: {info['n_proofs']} proofs. Next steps are in {judge_dir / 'NEXT.md'}")
            say(info["estimate"])

    (out / "pilot_report.json").write_text(json.dumps(report, indent=2, default=str))
    if "eval" in report:
        g = report["eval"]["gate1"]
        say(f"gate 1 {'PASSED' if g['passed'] else 'FAILED'} at k={g['k']}: macrostate NMI {g['macrostate_nmi']:.3f}, "
            f"folder NMI {g['folder_nmi']:.3f}, threshold {g['threshold']}")
        for k, v in report["eval"]["macrostate_sweep"].items():
            say(f"  k={k}: NMI {v['nmi']:.3f}, ARI {v['ari']:.3f}")
        return 0 if g["passed"] else 1
    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin wrapper over the CLI
    from .cli import main as cli_main

    return cli_main(["pilot", *(argv or sys.argv[1:])])
