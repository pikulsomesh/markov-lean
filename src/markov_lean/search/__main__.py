"""``python -m markov_lean.search``: best-first search over a benchmark file."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m markov_lean.search")
    p.add_argument("--config", default=str(ROOT / "configs" / "search.yaml"))
    p.add_argument("--benchmark", default=str(ROOT / "data/raw/deepseek_benchmarks/minif2f.jsonl"))
    p.add_argument("--source", default="minif2f", help="value of the steps table's source column")
    p.add_argument("--split", action="append", help="val or test; repeatable; default all")
    p.add_argument("--name", action="append", help="only these theorem names; repeatable")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", default=str(ROOT / "results" / "search" / "minif2f"))
    p.add_argument("--endpoint", help="override prover.base_url, e.g. http://1.2.3.4:8000")
    p.add_argument("--backend", choices=["vllm", "llama_cpp"],
                   help="override prover.backend: vllm (default) or llama_cpp for a local GGUF (Mac smoke)")
    p.add_argument("--model-path", help="override prover.model_path, the GGUF file for --backend llama_cpp")
    p.add_argument("--resume", action="store_true", help="skip theorems already in summary.jsonl")
    p.add_argument("--check", action="store_true", help="only check the endpoint and Lean, then exit")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.endpoint:
        config.setdefault("prover", {})["base_url"] = args.endpoint
    if args.backend:
        config.setdefault("prover", {})["backend"] = args.backend
    if args.model_path:
        config.setdefault("prover", {})["model_path"] = args.model_path

    if args.check:
        return _check(config)

    from markov_lean.search import runner

    stats = runner.run(config, args.benchmark, args.source, args.out,
                       splits=args.split, names=args.name, limit=args.limit, resume=args.resume)
    print(f"solved {stats['solved']}/{stats['theorems']} -> {args.out}")
    return 0


def _check(config: dict) -> int:
    import time

    from markov_lean.search.lean_env import DEFAULT_HEADER, LeanEnv, default_repl_command
    from markov_lean.search.prover import make_proposer
    from markov_lean.search.runner import _project_dir

    prover = make_proposer(config.get("prover"), seed=0)
    try:
        print("models served:", prover.models())
        t0 = time.monotonic()
        cands = prover.propose("theorem t (a b : ℕ) : a + b = b + a :=", [], "a b : ℕ\n⊢ a + b = b + a", n=4)
        print(f"sample tactics ({time.monotonic() - t0:.1f}s for 4 samples):", [c.tactic for c in cands])
    finally:
        prover.close()
    lean = dict(config.get("lean") or {})
    cmd = lean.get("command") or default_repl_command(lean["repl_dir"])
    t0 = time.monotonic()
    with LeanEnv(cmd, cwd=_project_dir(lean), header=lean.get("header", DEFAULT_HEADER),
                 startup_timeout=float(lean.get("startup_timeout", 900))) as env:
        print(f"lean REPL up, Mathlib imported in {time.monotonic() - t0:.0f}s")
        root = env.init_theorem("theorem t (a b : ℕ) : a + b = b + a :=")
        res = env.run_tactic(root.proof_state, "omega")
        print("lean ok:", res.solved, res.error or "")
        return 0 if res.solved else 1


if __name__ == "__main__":
    sys.exit(main())
