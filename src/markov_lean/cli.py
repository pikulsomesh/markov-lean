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

    for name, (_, help_) in PASSTHROUGH.items():
        sub.add_parser(name, help=help_ + "; see `markov-lean " + name + " --help`")

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
