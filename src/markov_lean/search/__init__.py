"""Prover harness and proof search (workstream A).

* :mod:`~markov_lean.search.lean_env`: tactic-mode wrapper around leanprover-community/repl.
* :mod:`~markov_lean.search.prover`: DeepSeek-Prover-V1.5 tactic proposals over vLLM's
  OpenAI-compatible HTTP API (``httpx`` only; install the ``search`` extra).
* :mod:`~markov_lean.search.best_first`: best-first search with a pluggable prior.
* :mod:`~markov_lean.search.runner`: benchmark runs, ``python -m markov_lean.search``.

Submodules import lazily so ``import markov_lean.search`` works without ``httpx``.
"""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "LeanEnv": "lean_env",
    "LeanError": "lean_env",
    "LeanTimeout": "lean_env",
    "TacticResult": "lean_env",
    "Candidate": "prover",
    "ProverClient": "prover",
    "build_prompt": "prover",
    "BestFirstSearch": "best_first",
    "SearchConfig": "best_first",
    "SearchResult": "best_first",
    "Theorem": "best_first",
    "lm_prior": "best_first",
    "make_lm_prior": "best_first",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(name)
