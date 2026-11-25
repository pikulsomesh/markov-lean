"""Command line entry point: `markov-lean <command>`.

Commands import their modules lazily so a missing optional dependency
(for example the anthropic SDK) only affects the command that needs it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_fetch(args: argparse.Namespace) -> int:
    return subprocess.call([str(ROOT / "scripts" / "fetch_data.sh"), *args.sources])


def cmd_pilot(args: argparse.Namespace) -> int:
    from markov_lean import pilot

    return pilot.run(_load_config(args.config), out_dir=Path(args.out))


def cmd_robustness(args: argparse.Namespace) -> int:
    from markov_lean import robustness as rb

    kw = {k: v for k, v in {
        "lags": args.lags, "macrostates": args.macrostates, "microstates": args.microstates,
        "featurizers": args.featurizers, "n_permutations": args.permutations, "n_bootstrap": args.bootstrap,
        "judges": args.judges, "jobs": args.jobs, "seed": args.seed,
    }.items() if v is not None}
    kw["figures"] = not args.no_figures
    kw["sweep"] = not args.no_sweep
    cfg = rb.RobustnessConfig.quick(**kw) if args.quick else rb.RobustnessConfig(**kw)
    summary = rb.run(args.pilot, args.out, cfg, labels=args.labels, steps=args.steps, gate_nmi=args.gate_nmi)
    return 0 if summary["verdict"]["robust_pass"] else 1


def cmd_full(args: argparse.Namespace) -> int:
    from markov_lean import full

    cfg = _load_config(args.config)
    if args.min_steps is not None:
        cfg.setdefault("data", {})["min_steps"] = args.min_steps
    return full.run(cfg, out_dir=Path(args.out), n_jobs=args.jobs, refit_features=args.refit_features,
                    skip_projection=args.no_project)


def cmd_project(args: argparse.Namespace) -> int:
    from markov_lean import full

    return full.run_project_only(_load_config(args.config), args.msm, Path(args.out), extra=args.dataset,
                                 n_jobs=args.jobs)


def _search_eval(rest: list[str]) -> int:
    from markov_lean import search_eval

    return search_eval.main(rest)


def _search_report(rest: list[str]) -> int:
    from markov_lean import search_report

    return search_report.main(rest)


def _cases(rest: list[str]) -> int:
    from markov_lean import cases

    return cases.main(rest)


def _bridges(rest: list[str]) -> int:
    from markov_lean import bridges

    return bridges.main(rest)


# Commands that parse their own options.
PASSTHROUGH = {
    "cases": (_cases, "pick readable proofs that cross areas; write the case-study figure and table"),
    "search-eval": (_search_eval, "proof search on miniF2F/PutnamBench under prior conditions (EC2)"),
    "search-report": (_search_report, "pass@k tables, paired tests and figures from search-eval runs"),
    "bridges": (_bridges, "tactics and lemmas at the steps where a proof switches area, with lift and CIs"),
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in PASSTHROUGH:
        return PASSTHROUGH[argv[0]][0](argv[1:])
    p = argparse.ArgumentParser(prog="markov-lean")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download datasets into data/raw (run on the Mac Mini or EC2)")
    f.add_argument("sources", nargs="*", help="subset of sources; default all")
    f.set_defaults(func=cmd_fetch)

    pl = sub.add_parser("pilot", help="run the Phase 2 Markov model pilot")
    pl.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    pl.add_argument("--out", default=str(ROOT / "results" / "pilot"))
    pl.set_defaults(func=cmd_pilot)

    rb = sub.add_parser("robustness", help="baselines, sweeps and memory tests for gate 1 (no API calls)")
    rb.add_argument("--pilot", required=True, help="pilot output folder (has msm/ and judge/labels.parquet)")
    rb.add_argument("--out", default=None, help="default: <pilot>/robustness")
    rb.add_argument("--labels", default=None, help="labels table; default <pilot>/judge/labels.parquet")
    rb.add_argument("--steps", default=None, help="steps table; default <pilot>/judge/sample_steps.parquet")
    rb.add_argument("--judges", nargs="+", default=None, help="the two judges to build the consensus from")
    rb.add_argument("--jobs", type=int, default=None, help="sweep cells run in parallel (default 1)")
    rb.add_argument("--quick", action="store_true", help="small grid and fewer resamples, for a first look")
    rb.add_argument("--lags", type=int, nargs="+", default=None)
    rb.add_argument("--macrostates", type=int, nargs="+", default=None)
    rb.add_argument("--microstates", type=int, nargs="+", default=None)
    rb.add_argument("--featurizers", nargs="+", default=None, help="e.g. tfidf_svd:64 tfidf_svd:128")
    rb.add_argument("--permutations", type=int, default=None)
    rb.add_argument("--bootstrap", type=int, default=None)
    rb.add_argument("--seed", type=int, default=None)
    rb.add_argument("--gate-nmi", type=float, default=0.15)
    rb.add_argument("--no-sweep", action="store_true", help="baselines and memory test only")
    rb.add_argument("--no-figures", action="store_true")
    rb.set_defaults(func=cmd_robustness)

    fu = sub.add_parser("full", help="fit the model on all LeanDojo train proofs and project the other datasets")
    fu.add_argument("--config", default=str(ROOT / "configs" / "full.yaml"))
    fu.add_argument("--out", default=str(ROOT / "results" / "full"))
    fu.add_argument("--jobs", type=int, default=None, help="worker processes for featurizing (config: features.n_jobs)")
    fu.add_argument("--min-steps", type=int, default=None, help="override data.min_steps for the fit")
    fu.add_argument("--refit-features", action="store_true", help="ignore the cached features and featurizer")
    fu.add_argument("--no-project", action="store_true", help="fit and save the model only")
    fu.set_defaults(func=cmd_full)

    pr = sub.add_parser("project", help="place datasets on a saved model and measure area moves (no refit)")
    pr.add_argument("--config", default=str(ROOT / "configs" / "full.yaml"))
    pr.add_argument("--msm", default=str(ROOT / "results" / "full" / "msm"), help="model directory from `full`")
    pr.add_argument("--out", default=str(ROOT / "results" / "full" / "project"))
    pr.add_argument("--dataset", action="append", default=[], metavar="NAME=PATH",
                    help="add a steps parquet (e.g. a search run's steps.parquet); repeatable")
    pr.add_argument("--jobs", type=int, default=None)
    pr.set_defaults(func=cmd_project)

    for name, (_, help_) in PASSTHROUGH.items():
        sub.add_parser(name, help=help_ + "; see `markov-lean " + name + " --help`")

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
