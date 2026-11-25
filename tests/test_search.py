"""Tests for the prover harness and search (src/markov_lean/search).

Nothing here talks to Lean, a GPU or the network: the REPL is a small Python script that
speaks the leanprover-community/repl JSON protocol over a toy proof world, and the vLLM
server is an ``httpx.MockTransport``.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from markov_lean.search import best_first, prover as prover_mod, runner
from markov_lean.search.best_first import (
    STEPS_COLUMNS,
    TRACE_COLUMNS,
    BestFirstSearch,
    SearchConfig,
    Theorem,
    log_policy,
    make_lm_prior,
)
from markov_lean.search.lean_env import LeanEnv, LeanError, as_sorry_command, goals_to_text
from markov_lean.search.prover import (
    Candidate,
    ProverClient,
    build_prompt,
    first_tactic,
    parse_choices,
    prefix_logprob,
    state_comment,
)

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------- fake REPL

FAKE_REPL = textwrap.dedent(
    r'''
    import json, sys, time

    # Toy proof world: goal -> tactic -> new goals.
    WORLD = {
        "⊢ start": {"a": ["⊢ mid"], "b": ["⊢ dead"], "c": ["⊢ start"], "split": ["⊢ L", "⊢ R"]},
        "⊢ mid": {"close": [], "b2": ["⊢ dead"]},
        "⊢ L": {"close": []},
        "⊢ R": {"close": []},
    }
    states = []

    def out(obj):
        # The real REPL pretty-prints over several lines, then a blank line.
        sys.stdout.write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n\n")
        sys.stdout.flush()

    def blocks():
        buf = []
        for line in sys.stdin:
            if line.strip():
                buf.append(line)
            elif buf:
                yield "".join(buf)
                buf = []
        if buf:
            yield "".join(buf)

    for block in blocks():
        req = json.loads(block)
        if "cmd" in req and "env" not in req:
            if "import Mathlib" not in req["cmd"]:
                out({"env": 0, "messages": [{"severity": "error", "pos": {"line": 1, "column": 0},
                                             "data": "no imports"}]})
            else:
                out({"env": 0})
        elif "cmd" in req:
            cmd = req["cmd"]
            if "bad_statement" in cmd:
                out({"env": 1, "messages": [{"severity": "error", "pos": {"line": 1, "column": 0},
                                             "data": "unknown identifier"}]})
                continue
            states.append(["⊢ start"])
            out({"env": 1, "sorries": [{"pos": {"line": 1, "column": 0}, "endPos": {"line": 1, "column": 5},
                                        "goal": "⊢ start", "proofState": len(states) - 1}],
                 "messages": [{"severity": "warning", "pos": {"line": 1, "column": 0},
                               "data": "declaration uses 'sorry'"}]})
        elif "tactic" in req:
            t, ps = req["tactic"], req["proofState"]
            if ps >= len(states):
                out({"message": "Unknown proof state."})
                continue
            goals = states[ps]
            if t == "sleep":
                time.sleep(60)
            if t == "msgerr":
                states.append(goals)
                out({"proofState": len(states) - 1, "goals": goals, "proofStatus": "Incomplete: open goals remain",
                     "messages": [{"severity": "error", "pos": {"line": 1, "column": 0}, "data": "type mismatch"}]})
                continue
            if t == "fake_close":
                states.append([])
                out({"proofState": len(states) - 1, "goals": [], "proofStatus": "Incomplete: contains sorry"})
                continue
            new = WORLD.get(goals[0], {}).get(t) if goals else None
            if new is None:
                out({"message": "Lean error:\n<input>:1:0: tactic '" + t + "' failed"})
                continue
            new = new + goals[1:]
            states.append(new)
            out({"proofState": len(states) - 1, "goals": new,
                 "proofStatus": "Completed" if not new else "Incomplete: open goals remain"})
    '''
)


@pytest.fixture
def repl_cmd(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_repl.py"
    script.write_text(FAKE_REPL, encoding="utf-8")
    return [sys.executable, "-I", str(script)]


@pytest.fixture
def env(repl_cmd):
    e = LeanEnv(repl_cmd, timeout=5.0, startup_timeout=10.0)
    e.start()
    yield e
    e.close()


# ---------------------------------------------------------------------- lean_env


def test_as_sorry_command_handles_both_statement_endings():
    assert as_sorry_command("theorem t : True :=") == "theorem t : True := by\n  sorry"
    assert as_sorry_command("theorem t : True := by\n") == "theorem t : True := by\n  sorry"
    with pytest.raises(ValueError):
        as_sorry_command("theorem t : True")


def test_goals_to_text():
    assert goals_to_text([]) == "no goals"
    assert goals_to_text(["⊢ a", "⊢ b"]) == "⊢ a\n\n⊢ b"


def test_init_and_run_tactics(env):
    root = env.init_theorem("theorem t : P :=")
    assert root.goals == ["⊢ start"] and root.state_text == "⊢ start"
    r = env.run_tactic(root.proof_state, "a")
    assert r.ok and r.goals == ["⊢ mid"] and not r.solved
    done = env.run_tactic(r.proof_state, "close")
    assert done.ok and done.solved and done.state_text == "no goals"
    # the earlier state is still usable (backtracking)
    again = env.run_tactic(root.proof_state, "b")
    assert again.ok and again.goals == ["⊢ dead"]


def test_tactic_failures_are_results_not_exceptions(env):
    root = env.init_theorem("theorem t : P :=")
    r = env.run_tactic(root.proof_state, "nonsense")
    assert not r.ok and "failed" in r.error
    r = env.run_tactic(root.proof_state, "msgerr")
    assert not r.ok and "type mismatch" in r.error
    r = env.run_tactic(root.proof_state, "fake_close")
    assert not r.ok and "contains sorry" in r.error
    for t in ("sorry", "exact sorry", "admit", "native_decide"):
        r = env.run_tactic(root.proof_state, t)
        assert not r.ok and r.error == "forbidden tactic"


def test_bad_statement_raises(env):
    with pytest.raises(LeanError, match="does not elaborate"):
        env.init_theorem("theorem bad_statement : P :=")


def test_header_errors_raise(repl_cmd):
    e = LeanEnv(repl_cmd, header="def x := 1", startup_timeout=10.0)
    with pytest.raises(LeanError, match="header"):
        e.start()


def test_missing_binary_raises():
    with pytest.raises(LeanError, match="could not start"):
        LeanEnv(["/nonexistent/repl"]).start()


def test_timeout_restarts_and_replay_recovers(repl_cmd):
    with LeanEnv(repl_cmd, timeout=0.5, startup_timeout=10.0) as e:
        root = e.init_theorem("theorem t : P :=")
        mid = e.run_tactic(root.proof_state, "a")
        gen = e.generation
        r = e.run_tactic(mid.proof_state, "sleep")
        assert not r.ok and r.error == "timeout"
        assert e.generation == gen + 1
        # old ids are gone; replaying the path rebuilds the state
        res = e.replay("theorem t : P :=", ["a"])
        assert res.ok and res.goals == ["⊢ mid"]
        assert e.run_tactic(res.proof_state, "close").solved


# ---------------------------------------------------------------------- prover


def test_prompt_matches_deepseek_layout():
    p = build_prompt("theorem t (n : ℕ) : n = n :=", ["intro h", "cases h with\n| a => rfl"],
                     "n : ℕ\n⊢ n = n", header="import Mathlib\n")
    assert p.startswith("Complete the following Lean 4 code:\n\n```lean4\nimport Mathlib\n")
    assert "theorem t (n : ℕ) : n = n := by\n  intro h\n  cases h with\n  | a => rfl\n" in p
    assert p.endswith("  /- tactic state:\n    n : ℕ\n    ⊢ n = n\n  -/\n")
    cot = build_prompt("theorem t : True := by", [], "⊢ True", header="", mode="cot")
    assert cot.startswith("Complete the following Lean 4 code with explanatory comments")


def test_state_comment_indents_every_line():
    assert state_comment("a\nb") == "  /- tactic state:\n    a\n    b\n  -/\n"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("  nlinarith [sq_nonneg (a - b)]\n  linarith\n", "nlinarith [sq_nonneg (a - b)]"),
        ("  -- try the obvious\n  simp\n", "simp"),
        ("\n  /- tactic state:\n    ⊢ x\n  -/\n  omega\n", "omega"),
        ("  rcases h with ⟨x, hx⟩ |\n      ⟨y, hy⟩\n  exact hx\n", "rcases h with ⟨x, hx⟩ |\n    ⟨y, hy⟩"),
        ("  norm_num -- closes it\n", "norm_num"),
        ("  decide\n```\nextra", "decide"),
        ("```", None),
        ("  -- only a comment\n", None),
        ("theorem other : True := by trivial", None),
    ],
)
def test_first_tactic(text, expected):
    got = first_tactic(text)
    assert (got[0] if got else None) == expected


def _logprobs(tokens, lps, offset0=0):
    offs, acc = [], offset0
    for t in tokens:
        offs.append(acc)
        acc += len(t)
    return {"tokens": tokens, "token_logprobs": lps, "text_offset": offs}


def test_prefix_logprob_counts_only_the_tactic_tokens():
    toks = ["  sim", "p", "\n", "  ring", "\n"]
    lp = [-0.5, -0.25, -0.1, -3.0, -0.1]
    text = "".join(toks)
    _, end = first_tactic(text)
    # offsets relative to the prompt (vLLM may count from there) give the same answer
    for off0 in (0, 1234):
        total, n = prefix_logprob(_logprobs(toks, lp, off0), end)
        assert n == 2 and total == pytest.approx(-0.75)
    # without text_offset, offsets are rebuilt from token lengths
    total, n = prefix_logprob({"tokens": toks, "token_logprobs": lp}, end)
    assert n == 2 and total == pytest.approx(-0.75)
    assert prefix_logprob(None, 10) == (0.0, 0)


def test_parse_choices_dedupes_with_logsumexp():
    def ch(text, lps):
        toks = [text]
        return {"text": text, "logprobs": _logprobs(toks, lps)}

    cands = parse_choices([ch("  simp\n", [-1.0]), ch("  ring\n", [-0.5]), ch("  simp\n", [-1.0]), ch("", [])])
    assert [c.tactic for c in cands] == ["simp", "ring"]
    simp = cands[0]
    assert simp.count == 2 and simp.logprob == pytest.approx(-1.0 + math.log(2))


def _mock_server(calls: list, fail_first: int = 0, status: int = 503):
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": prover_mod.DEFAULT_MODEL}]})
        body = json.loads(request.content)
        calls.append((dict(request.headers), body))
        state["n"] += 1
        if state["n"] <= fail_first:
            return httpx.Response(status, json={"error": "busy"})
        texts = ["  nlinarith\n", "  simp\n  ring\n", "  nlinarith\n"][: body["n"]]
        return httpx.Response(200, json={"choices": [
            {"index": i, "text": t, "logprobs": _logprobs([t], [-0.1 * (i + 1)])} for i, t in enumerate(texts)
        ]})

    return httpx.MockTransport(handler)


def _client(transport, **kw) -> ProverClient:
    return ProverClient(client=httpx.Client(transport=transport, base_url="http://prover"), retries=2, **kw)


def test_prover_request_and_candidates(monkeypatch):
    monkeypatch.setattr(prover_mod.time, "sleep", lambda s: None)
    calls: list = []
    p = _client(_mock_server(calls), n=3, max_tokens=128, seed=7)
    assert p.models() == [prover_mod.DEFAULT_MODEL]
    cands = p.propose("theorem t : 1 = 1 :=", [], "⊢ 1 = 1")
    _, body = calls[0]
    assert body["model"] == prover_mod.DEFAULT_MODEL
    assert body["n"] == 3 and body["logprobs"] == 1 and body["max_tokens"] == 128 and body["seed"] == 7
    assert "```" in body["stop"]
    assert body["prompt"].endswith("  /- tactic state:\n    ⊢ 1 = 1\n  -/\n")
    assert {c.tactic for c in cands} == {"nlinarith", "simp"}
    assert next(c for c in cands if c.tactic == "nlinarith").count == 2


def test_prover_retries_server_errors_but_not_client_errors(monkeypatch):
    monkeypatch.setattr(prover_mod.time, "sleep", lambda s: None)
    calls: list = []
    p = _client(_mock_server(calls, fail_first=2), n=1)
    assert [c.tactic for c in p.propose("theorem t : True :=", [], "⊢ True")] == ["nlinarith"]
    assert len(calls) == 3
    calls.clear()
    p = _client(_mock_server(calls, fail_first=10), n=1)
    with pytest.raises(httpx.HTTPStatusError):
        p.propose("theorem t : True :=", [], "⊢ True")
    assert len(calls) == 3  # 1 + 2 retries
    calls.clear()
    p = _client(_mock_server(calls, fail_first=10, status=400), n=1)
    with pytest.raises(httpx.HTTPStatusError):
        p.propose("theorem t : True :=", [], "⊢ True")
    assert len(calls) == 1


def test_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "k123")
    calls: list = []
    transport = _mock_server(calls)
    p = ProverClient(base_url="http://prover", n=1)
    p._client = httpx.Client(transport=transport, base_url="http://prover", headers=p._client.headers)
    p.propose("theorem t : True :=", [], "⊢ True")
    assert calls[0][0]["authorization"] == "Bearer k123"


# ---------------------------------------------------------------------- search


class FakeProver:
    """Deterministic proposals keyed by goal state."""

    def __init__(self, table: dict[str, list[tuple[str, float]]]):
        self.table = table
        self.calls: list[tuple[list[str], str]] = []

    def propose(self, statement, tactics, state_text, informal_prefix="", header=None, n=None):
        self.calls.append((list(tactics), state_text))
        return [Candidate(t, lp, n_tokens=3) for t, lp in self.table.get(state_text, [])]


# The LM prefers "b", which leads to a dead end; "a" leads to the proof.
TABLE = {
    "⊢ start": [("b", -0.1), ("a", -1.0), ("c", -2.0), ("boom", -0.05)],
    "⊢ mid": [("b2", -0.1), ("close", -0.5)],
    "⊢ dead": [("nothing", -0.1)],
}
THM = Theorem(name="toy", statement="theorem toy : P :=", source="minif2f", split="test")


def test_search_finds_proof_and_logs_schema_rows(env):
    fp = FakeProver(TABLE)
    res = BestFirstSearch(env, fp).search(THM)
    assert res.solved and res.stop_reason == "solved"
    assert res.tactics == ["a", "close"]
    assert res.expansions == 3  # root, dead (LM's favorite), mid
    assert res.proof_text == "theorem toy : P := by\n  a\n  close\n"

    steps = pd.DataFrame(res.steps)
    assert list(steps.columns) == STEPS_COLUMNS
    assert steps["step"].tolist() == [0, 1]
    assert steps["state_before"].tolist() == ["⊢ start", "⊢ mid"]
    assert steps["state_after"].tolist() == ["⊢ mid", "no goals"]
    assert set(steps["proof_id"]) == {"minif2f:toy"}

    trace = pd.DataFrame(res.trace)
    assert set(TRACE_COLUMNS) <= set(trace.columns)
    c_row = trace[trace.tactic == "c"].iloc[0]
    assert c_row.ok and c_row.error == "duplicate state"
    boom = trace[trace.tactic == "boom"].iloc[0]
    assert not boom.ok and "failed" in boom.error
    # policy over the two new states is normalized
    pushed = trace[trace.child_id.notna() & ~trace.solved]
    root_rows = pushed[pushed.node_id == 0]
    assert np.exp(root_rows.log_policy).sum() == pytest.approx(1.0)


def test_prior_changes_what_gets_expanded(env):
    seen = []

    def prefer_a(state_text, candidates):
        seen.append((state_text, candidates))
        return [10.0 if c["tactic"] == "a" else 1.0 for c in candidates]

    res = BestFirstSearch(env, FakeProver(TABLE), prior=prefer_a).search(THM)
    assert res.solved and res.expansions == 2
    state, cands = seen[0]
    assert state == "⊢ start"
    assert {c["tactic"]: c["state_after"] for c in cands} == {"b": "⊢ dead", "a": "⊢ mid"}
    assert all({"logprob", "n_tokens", "count"} <= set(c) for c in cands)


def test_zero_prior_weight_prunes(env):
    res = BestFirstSearch(env, FakeProver(TABLE), prior=lambda s, c: [0.0 if x["tactic"] == "a" else 1.0 for x in c]).search(THM)
    assert not res.solved and res.stop_reason == "exhausted"
    assert any(r["error"] == "zero prior weight" for r in res.trace)


def test_budget_and_depth_limits(env):
    res = BestFirstSearch(env, FakeProver(TABLE), config=SearchConfig(max_expansions=1)).search(THM)
    assert not res.solved and res.stop_reason == "budget" and res.expansions == 1
    res = BestFirstSearch(env, FakeProver(TABLE), config=SearchConfig(max_depth=1)).search(THM)
    assert not res.solved and res.stop_reason == "exhausted"


def test_wall_time_limit(env):
    ticks = iter(range(0, 10_000, 100))
    res = BestFirstSearch(env, FakeProver(TABLE), config=SearchConfig(wall_time=150),
                          clock=lambda: next(ticks)).search(THM)
    assert not res.solved and res.stop_reason == "wall_time"


def test_multi_goal_proof(env):
    table = {"⊢ start": [("split", -0.1)], "⊢ L\n\n⊢ R": [("close", -0.1)], "⊢ R": [("close", -0.1)]}
    res = BestFirstSearch(env, FakeProver(table)).search(THM)
    assert res.solved and res.tactics == ["split", "close", "close"]
    assert res.steps[0]["state_after"] == "⊢ L\n\n⊢ R"


def test_statement_error_is_reported(env):
    res = BestFirstSearch(env, FakeProver(TABLE)).search(Theorem("bad", "theorem bad_statement : P :="))
    assert not res.solved and res.stop_reason.startswith("statement error")


def test_timeout_mid_search_replays_the_node(repl_cmd):
    table = {"⊢ start": [("a", -0.1)], "⊢ mid": [("sleep", -0.1), ("close", -0.5)]}
    with LeanEnv(repl_cmd, timeout=0.5, startup_timeout=10.0) as e:
        res = BestFirstSearch(e, FakeProver(table)).search(THM)
    assert res.solved and res.tactics == ["a", "close"]
    assert any(r["error"] == "timeout" for r in res.trace)


def test_reset_and_record_hooks(env):
    class Recorder:
        def __init__(self):
            self.events = []

        def __call__(self, s, c):
            return [1.0] * len(c)

        def reset(self):
            self.events.append("reset")

        def record(self, state, progress):
            self.events.append((state, progress))

    rec = Recorder()
    BestFirstSearch(env, FakeProver(TABLE), prior=rec).search(THM)
    assert rec.events[0] == "reset"
    assert ("⊢ mid", 1.0) in rec.events  # the closing expansion closed one goal


def test_lm_prior_and_log_policy():
    prior = make_lm_prior()
    w = prior("⊢ x", [{"logprob": 0.0}, {"logprob": math.log(0.5)}])
    assert w == pytest.approx([2 / 3, 1 / 3])
    w = make_lm_prior(length_norm=True)("⊢ x", [{"logprob": -2.0, "n_tokens": 2}, {"logprob": -1.0, "n_tokens": 1}])
    assert w == pytest.approx([0.5, 0.5])
    assert log_policy([1, 3], 2) == pytest.approx([math.log(0.25), math.log(0.75)])
    assert log_policy([0, 0], 2) == pytest.approx([math.log(0.5)] * 2)
    with pytest.raises(ValueError):
        log_policy([1.0], 2)
    with pytest.raises(ValueError):
        log_policy([-1.0, 1.0], 2)


def test_area_prior_plugs_in(env):
    area_prior = pytest.importorskip("markov_lean.area_prior")

    def featurize(text: str):
        return [1.0 if "mid" in text else 0.0, 1.0 if "dead" in text else 0.0]

    prior = area_prior.make_prior(featurize, np.eye(2), np.eye(2), lam=0.5,
                                  coarse_T=np.array([[0.5, 0.5], [0.5, 0.5]]))
    res = BestFirstSearch(env, FakeProver(TABLE), prior=prior).search(THM)
    assert res.solved
    assert len(prior.history) == res.expansions  # record() was called per expansion
    pushed = [r for r in res.trace if r["prior_weight"] is not None and r["node_id"] == 0]
    assert sum(r["prior_weight"] for r in pushed) == pytest.approx(1.0)


# ---------------------------------------------------------------------- runner


def test_runner_end_to_end(tmp_path, repl_cmd, monkeypatch):
    bench = tmp_path / "bench.jsonl"
    rows = [
        {"name": "t1", "split": "valid", "formal_statement": "theorem t1 : P :=", "header": "import Mathlib\nopen Real\n",
         "informal_prefix": ""},
        {"name": "t2", "split": "test", "formal_statement": "theorem t2 : P := by\n", "header": "import Mathlib\n"},
        {"name": "t3", "split": "test", "formal_statement": "theorem t3 : P :=", "header": "import FooBar\n"},
    ]
    bench.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def handler(request):
        body = json.loads(request.content)
        state = body["prompt"].rsplit("/- tactic state:\n", 1)[1]
        text = {"⊢ start": "  a\n", "⊢ mid": "  close\n"}.get(state.split("\n")[0].strip(), "  nope\n")
        return httpx.Response(200, json={"choices": [{"text": text, "logprobs": _logprobs([text], [-0.1])}]})

    prover = ProverClient(client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://p"), n=1)
    env = LeanEnv(repl_cmd, timeout=5, startup_timeout=10)
    config = {"search": {"max_expansions": 10}, "prior": {"factory": "lm"}}
    out = tmp_path / "out"
    stats = runner.run(config, bench, "minif2f", out, env=env, prover=prover)
    assert stats == {"theorems": 2, "solved": 2}  # t3 skipped: import missing in the REPL header

    summary = [json.loads(l) for l in (out / "summary.jsonl").read_text().splitlines()]
    assert [s["proof_id"] for s in summary] == ["minif2f:t1", "minif2f:t2"]
    assert summary[0]["split"] == "val" and summary[0]["proof"] == "theorem t1 : P := by\n  a\n  close\n"
    steps = pd.read_parquet(out / "steps.parquet")
    assert list(steps.columns) == STEPS_COLUMNS and len(steps) == 4
    trace = pd.read_parquet(out / "trace.parquet")
    assert list(trace.columns) == TRACE_COLUMNS

    # resume skips what is done
    env2 = LeanEnv(repl_cmd, timeout=5, startup_timeout=10)
    stats = runner.run(config, bench, "minif2f", out, resume=True, env=env2, prover=prover)
    assert stats == {"theorems": 0, "solved": 0}
    assert len(pd.read_parquet(out / "steps.parquet")) == 4


def test_split_header_and_theorem_filters(tmp_path):
    assert runner.split_header("import Mathlib\nimport Aesop\n\nopen Real\n") == (
        ["import Mathlib", "import Aesop"], "open Real")
    bench = tmp_path / "b.jsonl"
    bench.write_text("".join(json.dumps({"name": f"t{i}", "split": s, "formal_statement": "theorem x : P :=",
                                         "header": "import Mathlib\nopen Nat\n"}) + "\n"
                             for i, s in enumerate(["valid", "test", "test"])))
    thms = list(runner.load_theorems(bench, "minif2f", splits=["test"], limit=1))
    assert [t.name for t in thms] == ["t1"]
    assert thms[0].lean_statement == "open Nat\n\ntheorem x : P :="


def test_load_prior():
    assert runner.load_prior(None)("s", [{"logprob": 0.0}]) == [1.0]
    p = runner.load_prior({"factory": "markov_lean.search.best_first:make_lm_prior", "kwargs": {"temperature": 2.0}})
    assert p("s", [{"logprob": 0.0}, {"logprob": -2.0}]) == pytest.approx([1 / (1 + math.exp(-1)), math.exp(-1) / (1 + math.exp(-1))])
    with pytest.raises(ValueError):
        runner.load_prior({"factory": "no_colon"})


def test_search_config_rejects_unknown_keys():
    assert SearchConfig.from_dict({"max_expansions": 5}).max_expansions == 5
    with pytest.raises(ValueError):
        SearchConfig.from_dict({"max_expansion": 5})


def test_search_yaml_loads():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "search.yaml").read_text())
    SearchConfig.from_dict(cfg["search"])
    ProverClient(**cfg["prover"]).close()
    runner.load_prior(cfg["prior"])


# ---------------------------------------------------------------------- scripts


@pytest.mark.parametrize("name", ["setup_ec2.sh", "serve_prover.sh"])
def test_ec2_scripts_parse_and_never_touch_instances(name):
    path = ROOT / "scripts" / name
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    for banned in ("aws ec2", "run-instances", "start-instances", "boto3", "create-instance"):
        assert banned not in text
    if name == "serve_prover.sh":
        assert "--dtype bfloat16" in text and "MAX_MODEL_LEN:-4096" in text
