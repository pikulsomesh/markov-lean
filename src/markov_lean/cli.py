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


def main(argv: list[str] | None = None) -> int:
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

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
