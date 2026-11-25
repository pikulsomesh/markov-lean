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


# Commands that parse their own options.
PASSTHROUGH = {
    "search-eval": (_search_eval, "proof search on miniF2F/PutnamBench under prior conditions (EC2)"),
    "search-report": (_search_report, "pass@k tables, paired tests and figures from search-eval runs"),
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
