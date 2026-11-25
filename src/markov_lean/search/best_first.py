"""Best-first proof search over Lean goal states with a pluggable prior.

One expansion of a node:

1. ask the prover for tactic candidates at the node's goal state;
2. run each candidate in Lean (best LM logprob first), stopping at the first that closes
   the proof;
3. hand the candidates that led to a new state to ``prior(state_text, candidates)``;
4. push each new state with ``path_score = parent score + log pi(a | s)``, where ``pi`` is
   the prior's weights normalized over those candidates.

Nodes come off the queue highest ``path_score`` first (the GPT-f / HTPS style best-first
search).

The prior contract is the one ``markov_lean.area_prior.make_prior`` implements. Each
candidate is a dict with ``tactic``, ``state_after`` (the goal state it leads to, ``no
goals`` when closed), ``logprob`` (LM log-probability), ``n_tokens`` and ``count``; the
prior returns one non-negative weight per candidate. The weights *are* the policy, so a
prior that wants the LM in the mix mixes it in itself (``AreaPrior`` does, with ``lam``).
The default, :func:`lm_prior`, is the softmax of the LM logprobs. If the prior has
``reset()`` it is called at the start of each theorem, and ``record(state_text,
progress)`` after each expansion, ``progress`` being the goals closed by its best child.

Every tactic tried is logged as one trace row. Trace rows carry all the columns of the
``steps`` table in docs/schema.md (``step`` is the depth of the state the tactic was run
on) plus search columns, so the rows with ``ok`` true can feed the Markov model as they
are. :func:`proof_steps` gives the ``steps`` rows of the proof found.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from markov_lean.search.lean_env import LeanError, LeanEnv, NO_GOALS, goals_to_text
from markov_lean.search.prover import Candidate

Prior = Callable[[str, list[dict]], Sequence[float]]

STEPS_COLUMNS = [
    "proof_id", "step", "source", "split", "theorem", "file_path",
    "tactic", "state_before", "state_after", "true_area",
]
TRACE_COLUMNS = STEPS_COLUMNS + [
    "node_id", "parent_id", "child_id", "ok", "error", "solved", "lm_logprob",
    "n_tokens", "count", "prior_weight", "log_policy", "path_score", "elapsed",
]


def make_lm_prior(temperature: float = 1.0, length_norm: bool = False) -> Prior:
    """Softmax of LM logprobs (divided by token count first when ``length_norm``)."""

    def prior(state_text: str, candidates: list[dict]) -> list[float]:
        if not candidates:
            return []
        lps = []
        for c in candidates:
            lp = float(c["logprob"])
            if length_norm:
                lp /= max(int(c.get("n_tokens") or 1), 1)
            lps.append(lp / temperature)
        m = max(lps)
        w = [math.exp(x - m) for x in lps]
        s = sum(w)
        return [x / s for x in w]

    return prior


lm_prior = make_lm_prior()


class Proposer(Protocol):
    def propose(
        self,
        statement: str,
        tactics: list[str],
        state_text: str,
        informal_prefix: str = "",
        header: str | None = None,
        n: int | None = None,
    ) -> list[Candidate]: ...


@dataclass
class SearchConfig:
    max_expansions: int = 600        # prover calls per theorem
    max_depth: int = 64              # tactics on one path
    wall_time: float = 600.0         # seconds per theorem
    n_candidates: int | None = None  # overrides the prover's n
    dedupe_states: bool = True       # skip states already in the tree

    @classmethod
    def from_dict(cls, d: dict | None) -> "SearchConfig":
        d = dict(d or {})
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__}
        if d:
            raise ValueError(f"unknown search options: {sorted(d)}")
        return cls(**known)


@dataclass
class Node:
    id: int
    parent: int | None
    tactics: list[str]
    goals: list[str]
    proof_state: int
    generation: int
    depth: int
    score: float

    @property
    def state_text(self) -> str:
        return goals_to_text(self.goals)


@dataclass
class Theorem:
    """A statement to prove. ``statement`` ends with ``:=`` or ``:= by``."""

    name: str
    statement: str
    source: str = "minif2f"
    split: str = "test"
    file_path: str = ""
    informal_prefix: str = ""
    header: str | None = None   # prompt header; None uses the prover's
    lean_preamble: str = ""     # non-import header lines (open, set_option) sent to Lean

    @property
    def proof_id(self) -> str:
        return f"{self.source}:{self.name}"

    @property
    def lean_statement(self) -> str:
        """What the REPL elaborates: the preamble, then the statement."""
        return (self.lean_preamble.rstrip() + "\n\n" if self.lean_preamble.strip() else "") + self.statement


@dataclass
class SearchResult:
    theorem: Theorem
    solved: bool
    stop_reason: str
    tactics: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    expansions: int = 0
    nodes: int = 0
    elapsed: float = 0.0

    @property
    def proof_text(self) -> str:
        """The found proof as Lean source, for an independent recheck."""
        body = "".join("  " + t.replace("\n", "\n  ") + "\n" for t in self.tactics)
        stmt = self.theorem.statement.rstrip()
        stmt = stmt if stmt.endswith("by") else stmt + " by"
        return stmt + "\n" + body

    def summary(self) -> dict:
        return {
            "proof_id": self.theorem.proof_id,
            "theorem": self.theorem.name,
            "source": self.theorem.source,
            "split": self.theorem.split,
            "solved": self.solved,
            "stop_reason": self.stop_reason,
            "n_tactics": len(self.tactics),
            "expansions": self.expansions,
            "nodes": self.nodes,
            "elapsed": round(self.elapsed, 3),
            "proof": self.proof_text if self.solved else "",
        }


def log_policy(weights: Sequence[float], n: int) -> list[float]:
    """Normalize prior weights to log-probabilities; ``-inf`` where a weight is 0."""
    w = [float(x) for x in weights]
    if len(w) != n:
        raise ValueError(f"prior returned {len(w)} weights for {n} candidates")
    if any(not math.isfinite(x) or x < 0 for x in w):
        raise ValueError(f"prior weights must be finite and >= 0, got {w}")
    s = sum(w)
    if s <= 0:
        return [-math.log(n)] * n if n else []
    return [math.log(x / s) if x > 0 else -math.inf for x in w]


def _state_key(goals: list[str]) -> str:
    return "\n\n".join(" ".join(g.split()) for g in goals)


class BestFirstSearch:
    def __init__(
        self,
        env: LeanEnv,
        prover: Proposer,
        prior: Prior = lm_prior,
        config: SearchConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.env = env
        self.prover = prover
        self.prior = prior
        self.cfg = config or SearchConfig()
        self.clock = clock

    def _refresh(self, thm: Theorem, node: Node) -> bool:
        """Make ``node.proof_state`` valid again after a REPL restart."""
        if node.generation == self.env.generation:
            return True
        try:
            res = self.env.replay(thm.lean_statement, node.tactics)
        except LeanError:
            return False
        if not res.ok or res.proof_state is None:
            return False
        node.proof_state, node.generation = res.proof_state, self.env.generation
        return True

    def search(self, thm: Theorem) -> SearchResult:
        cfg = self.cfg
        t0 = self.clock()
        reset = getattr(self.prior, "reset", None)
        if callable(reset):
            reset()
        record = getattr(self.prior, "record", None)
        try:
            root_state = self.env.init_theorem(thm.lean_statement)
        except LeanError as e:
            return SearchResult(thm, False, f"statement error: {e}", elapsed=self.clock() - t0)
        root = Node(0, None, [], root_state.goals, root_state.proof_state, root_state.generation, 0, 0.0)
        nodes: dict[int, Node] = {0: root}
        result = SearchResult(thm, False, "exhausted", nodes=1)
        heap: list[tuple[float, int]] = [(0.0, 0)]
        seen = {_state_key(root.goals)}
        while heap:
            if result.expansions >= cfg.max_expansions:
                result.stop_reason = "budget"
                break
            if self.clock() - t0 >= cfg.wall_time:
                result.stop_reason = "wall_time"
                break
            _, nid = heapq.heappop(heap)
            node = nodes[nid]
            if node.depth >= cfg.max_depth or not self._refresh(thm, node):
                continue
            cands = self.prover.propose(
                thm.statement, node.tactics, node.state_text,
                informal_prefix=thm.informal_prefix, header=thm.header, n=cfg.n_candidates,
            )
            result.expansions += 1

            # Run every candidate in Lean first: the prior scores where a tactic leads.
            new: list[tuple[Candidate, Any, dict]] = []   # (candidate, result, trace row)
            for c in sorted(cands, key=lambda c: c.logprob, reverse=True):
                if self.clock() - t0 >= cfg.wall_time or not self._refresh(thm, node):
                    break
                res = self.env.run_tactic(node.proof_state, c.tactic)
                row = self._row(thm, node, c, res.ok, res.state_text if res.ok else "",
                                res.error, res.solved, res.elapsed)
                result.trace.append(row)
                if res.solved:
                    row["child_id"] = len(nodes)
                    result.solved, result.stop_reason = True, "solved"
                    result.tactics = node.tactics + [c.tactic]
                    result.nodes = len(nodes) + 1
                    if callable(record):
                        record(node.state_text, float(len(node.goals)))
                    return self._finish(result, t0)
                if not res.ok:
                    continue
                key = _state_key(res.goals)
                if cfg.dedupe_states and key in seen:
                    row["error"] = "duplicate state"
                    continue
                seen.add(key)
                new.append((c, res, row))

            if callable(record):
                closed = max((len(node.goals) - len(r.goals) for _, r, _ in new), default=0)
                record(node.state_text, float(max(closed, 0)))
            if not new:
                continue
            prior_in = [
                {"tactic": c.tactic, "state_after": r.state_text, "logprob": c.logprob,
                 "n_tokens": c.n_tokens, "count": c.count}
                for c, r, _ in new
            ]
            weights = list(self.prior(node.state_text, prior_in))
            logpi = log_policy(weights, len(new))
            for (c, r, row), w, lp in zip(new, weights, logpi):
                row["prior_weight"], row["log_policy"] = float(w), lp
                row["path_score"] = node.score + lp
                if not math.isfinite(lp):
                    row["error"] = "zero prior weight"
                    continue
                child = Node(
                    len(nodes), node.id, node.tactics + [c.tactic], r.goals,
                    int(r.proof_state), r.generation, node.depth + 1, node.score + lp,
                )
                nodes[child.id] = child
                row["child_id"] = child.id
                heapq.heappush(heap, (-child.score, child.id))
        if result.stop_reason == "exhausted" and self.clock() - t0 >= cfg.wall_time:
            result.stop_reason = "wall_time"   # ran out mid-expansion, not out of states
        result.nodes = len(nodes)
        return self._finish(result, t0)

    @staticmethod
    def _row(
        thm: Theorem, node: Node, c: Candidate, ok: bool, state_after: str,
        error: str | None, solved: bool, elapsed: float,
    ) -> dict:
        return {
            "proof_id": thm.proof_id,
            "step": node.depth,
            "source": thm.source,
            "split": thm.split,
            "theorem": thm.name,
            "file_path": thm.file_path,
            "tactic": c.tactic,
            "state_before": node.state_text,
            "state_after": state_after,
            "true_area": None,
            "node_id": node.id,
            "parent_id": node.parent,
            "child_id": None,
            "ok": ok,
            "error": error,
            "solved": solved,
            "lm_logprob": c.logprob,
            "n_tokens": c.n_tokens,
            "count": c.count,
            "prior_weight": None,
            "log_policy": None,
            "path_score": None,
            "elapsed": elapsed,
        }

    def _finish(self, result: SearchResult, t0: float) -> SearchResult:
        result.elapsed = self.clock() - t0
        if result.solved:
            result.steps = proof_steps(result)
        return result


def proof_steps(result: SearchResult) -> list[dict]:
    """Rows of the ``steps`` table for the proof found, in order."""
    closing = [r for r in result.trace if r["solved"]]
    if not closing:
        return []
    by_child = {r["child_id"]: r for r in result.trace if r["child_id"] is not None}
    path = [closing[-1]]
    while path[-1]["node_id"] != 0:
        path.append(by_child[path[-1]["node_id"]])
    path.reverse()
    rows = [{k: r[k] for k in STEPS_COLUMNS} for r in path]
    for i, r in enumerate(rows):
        r["step"] = i
    rows[-1]["state_after"] = NO_GOALS
    return rows
