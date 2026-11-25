"""Tests for the local llama.cpp prover backend (src/markov_lean/search/local_prover.py).

No model and no ``llama_cpp``: :class:`TrieModel` is a character-level stand-in that
continues a prompt with scripted text, keyed by the goal state in the prompt's
``/- tactic state`` comment, with chosen probabilities at each branch point. Set
``MARKOV_LEAN_TEST_GGUF`` to a GGUF file to also run the real llama.cpp wrapper.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from markov_lean import cli, search_eval as se
from markov_lean.search import local_prover as lp
from markov_lean.search.lean_env import LeanEnv, LeanError
from markov_lean.search.prover import ProverClient, make_proposer

from test_search import FAKE_REPL

ROOT = Path(__file__).resolve().parents[1]
BOS, EOS, VOCAB = 1, 2, 70_000
PEAK = 50.0


class TrieModel:
    """Character tokens (``ord(c) + 3``). After the prompt it continues with one of the
    scripted texts for the prompt's goal state: at each position the next character's
    probability is the total weight of the texts that agree with what came so far; once
    a text is finished (or nothing matches) it emits EOS."""

    def __init__(self, table: dict[str, list[tuple[str, float]]], n_ctx: int = 4096) -> None:
        self.table, self.n_ctx, self.eos = table, n_ctx, EOS
        self.cached: list[int] = []
        self.evaluated = 0      # tokens run through the "model", after prefix reuse
        self.closed = False

    def tokenize(self, text: str) -> list[int]:
        return [BOS] + [ord(c) + 3 for c in text]

    def detokenize(self, tokens, prev=()) -> str:
        return "".join(chr(t - 3) for t in tokens if t > 2)

    def feed(self, tokens) -> np.ndarray:
        tokens = list(tokens)
        keep = 0
        for a, b in zip(self.cached, tokens):
            if a != b:
                break
            keep += 1
        keep = min(keep, len(tokens) - 1)
        self.evaluated += len(tokens) - keep
        self.cached = tokens
        text = self.detokenize(tokens)
        head, _, gen = text.rpartition("-/\n")
        state = head.rpartition("/- tactic state:")[2]
        options = next((v for k, v in self.table.items() if k in state), [])
        logits = np.zeros(VOCAB)
        weights: dict[int, float] = {}
        for cont, w in options:
            if cont.startswith(gen) and len(cont) > len(gen):
                t = ord(cont[len(gen)]) + 3
                weights[t] = weights.get(t, 0.0) + w
        if not weights:
            logits[EOS] = PEAK
            return logits
        total = sum(weights.values())
        for t, w in weights.items():
            logits[t] = PEAK + math.log(w / total)
        return logits

    def close(self) -> None:
        self.closed = True


START = [("  a\n  close\n", 0.7),
         ("  nlinarith [sq_nonneg (a - b),\n    sq_nonneg a]\n  done\n", 0.3)]


def _prover(table=None, **kw) -> lp.LlamaCppProver:
    model = TrieModel(table or {"⊢ start": START}, n_ctx=kw.pop("n_ctx", 4096))
    return lp.LlamaCppProver("unused.gguf", local_model=model, **kw)


# ---------------------------------------------------------------------- sampling


def test_sample_token_greedy_and_raw_logprob():
    logits = np.log(np.array([0.6, 0.3, 0.1]))
    tok, lpr = lp.sample_token(logits, 0.0, 0.95, np.random.default_rng(0))
    assert tok == 0 and lpr == pytest.approx(math.log(0.6))
    # the log-probability is of the raw distribution, whatever the temperature
    rng = np.random.default_rng(1)
    for _ in range(50):
        tok, lpr = lp.sample_token(logits, 2.0, 1.0, rng)
        assert lpr == pytest.approx(math.log([0.6, 0.3, 0.1][tok]))


def test_sample_token_top_p_keeps_the_nucleus_only():
    logits = np.log(np.array([0.6, 0.3, 0.1]))
    rng = np.random.default_rng(0)
    assert {lp.sample_token(logits, 1.0, 0.5, rng)[0] for _ in range(200)} == {0}
    got = [lp.sample_token(logits, 1.0, 0.85, rng)[0] for _ in range(400)]
    assert set(got) == {0, 1} and 0.55 < got.count(0) / len(got) < 0.78   # 0.6 / 0.9 renormalized
    # a flat distribution falls back to the full sort and still samples correctly
    flat = np.zeros(100_000)
    assert 0 <= lp.sample_token(flat, 1.0, 0.95, rng)[0] < 100_000


def test_sample_token_is_reproducible_from_the_seed():
    logits = np.random.default_rng(3).standard_normal(1000) * 3
    a = [lp.sample_token(logits, 1.0, 0.95, np.random.default_rng([7, i]))[0] for i in range(20)]
    b = [lp.sample_token(logits, 1.0, 0.95, np.random.default_rng([7, i]))[0] for i in range(20)]
    assert a == b and len(set(a)) > 1


@pytest.mark.parametrize("text,done", [
    ("  nlinar", False),
    ("  omega\n", False),                       # a deeper continuation line could follow
    ("  omega\n  ", False),
    ("  omega\n  simp", True),                  # the next tactic has started
    ("  omega\n\n", True),                      # blank line closes the block
    ("  nlinarith [sq_nonneg a,\n", False),
    ("  nlinarith [sq_nonneg a,\n ", False),      # a line of spaces may still go deeper
    ("  nlinarith [sq_nonneg a,\n    sq_nonneg b]\n  linarith", True),
    ("  simp```", True),
    ("  -- a comment first\n", False),
    ("", False),
])
def test_tactic_complete(text, done):
    assert lp.tactic_complete(text) is done


# ---------------------------------------------------------------------- the prover


def test_proposes_deduplicated_first_tactics_with_raw_logprobs():
    p = _prover(n=16, seed=0)
    cands = p.propose("theorem t (a b : ℕ) : a + b = b + a :=", [], "⊢ start")
    by = {c.tactic: c for c in cands}
    assert set(by) == {"a", "nlinarith [sq_nonneg (a - b),\n  sq_nonneg a]"}
    assert sum(c.count for c in cands) == 16
    for c in cands:   # duplicates merge with log-sum-exp, so logprob = log(count * p)
        p_one = 0.7 if c.tactic == "a" else 0.3
        assert c.logprob == pytest.approx(math.log(c.count * p_one), abs=1e-6)
    # generation stopped at the end of the first tactic, not the end of the scripted text
    assert all("close" not in r and "done" not in r for c in cands for r in c.raw)
    assert p.stats["samples"] == 16 and p.stats["requests"] == 1


def test_n_tokens_counts_the_tactic_only():
    p = _prover({"⊢ start": [("  simp\n  done\n", 1.0)]}, n=1, seed=0)
    (c,) = p.propose("theorem t : P :=", [], "⊢ start")
    assert c.tactic == "simp" and c.n_tokens == len("  simp")


def test_stop_strings_and_eos_end_a_sample():
    p = _prover({"⊢ start": [("  simp```\nmore", 0.5), ("  ring", 0.5)]}, n=8, seed=0)
    cands = p.propose("theorem t : P :=", [], "⊢ start")
    assert {c.tactic for c in cands} == {"simp", "ring"}
    assert all("more" not in r and "```" not in r for c in cands for r in c.raw)


def test_same_seed_same_samples_and_prefix_reuse():
    a, b = _prover(n=12, seed=5), _prover(n=12, seed=5)
    ca = a.propose("theorem t : P :=", ["x"], "⊢ start")
    cb = b.propose("theorem t : P :=", ["x"], "⊢ start")
    assert [(c.tactic, c.count) for c in ca] == [(c.tactic, c.count) for c in cb]
    # the second call at the same state re-evaluates one prompt token, not the prompt
    model = a._model
    before = model.evaluated
    a.propose("theorem t : P :=", ["x"], "⊢ start", n=1)
    prompt_len = len(model.tokenize(lp.build_prompt("theorem t : P :=", ["x"], "⊢ start")))
    assert model.evaluated - before < 20 < prompt_len


def test_prompt_longer_than_the_context_gives_no_candidates(caplog):
    p = _prover(n=4, seed=0, n_ctx=50)
    assert p.propose("theorem t : P :=", [], "⊢ start") == []
    assert p.stats["too_long"] == 1 and "does not fit" in caplog.text


def test_close_closes_the_model():
    p = _prover(n=1, seed=0)
    p.propose("theorem t : P :=", [], "⊢ start")
    p.close()
    assert p._model.closed and p.models()[0].endswith("(unused.gguf, llama.cpp)")


# ---------------------------------------------------------------------- choosing a backend


def test_make_proposer_picks_the_backend(monkeypatch):
    made = {}

    def fake_load(path, **kw):
        made.update(path=path, **kw)
        return TrieModel({})

    monkeypatch.setattr(lp, "load_model", fake_load)
    cfg = {"base_url": "http://x:8000", "timeout": 5, "retries": 1, "n": 4, "max_tokens": 64,
           "model": "m", "model_path": "~/m.gguf", "n_ctx": 2048}
    vllm = make_proposer(cfg)
    assert isinstance(vllm, ProverClient) and vllm.n == 4
    vllm.close()
    local = make_proposer({**cfg, "backend": "llama_cpp"}, seed=3)
    assert isinstance(local, lp.LlamaCppProver) and local.seed == 3 and local.max_tokens == 64
    assert made["path"] == "~/m.gguf" and made["n_ctx"] == 2048
    with pytest.raises(ValueError, match="model_path"):
        make_proposer({"backend": "llama_cpp"})
    with pytest.raises(ValueError, match="unknown prover backend"):
        make_proposer({"backend": "mlx"})


def test_vllm_context_overflow_gives_no_candidates():
    def handler(request):
        return httpx.Response(400, json={"object": "error", "message": (
            "This model's maximum context length is 4096 tokens. However, you requested 4300 tokens.")})

    p = ProverClient(client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://p"))
    assert p.propose("theorem t : P :=", [], "⊢ start") == []

    def other(request):
        return httpx.Response(400, json={"message": "bad request"})

    p = ProverClient(client=httpx.Client(transport=httpx.MockTransport(other), base_url="http://p"))
    with pytest.raises(httpx.HTTPStatusError):
        p.propose("theorem t : P :=", [], "⊢ start")


def test_lean_env_start_failure_includes_stderr(tmp_path):
    cmd = [sys.executable, "-c", "import sys; sys.stderr.write('error: unknown package Mathlib'); sys.exit(1)"]
    with pytest.raises(LeanError, match="unknown package Mathlib"):
        LeanEnv(cmd, cwd=tmp_path, startup_timeout=10).start()


# ---------------------------------------------------------------------- end to end


WORLD = {"⊢ start": [("  a\n", 0.6), ("  b\n", 0.4)], "⊢ mid": [("  close\n", 1.0)]}


@pytest.fixture
def eval_setup(tmp_path, monkeypatch):
    models = []

    def fake_load(path, **kw):
        models.append(TrieModel(WORLD))
        return models[-1]

    monkeypatch.setattr(lp, "load_model", fake_load)
    script = tmp_path / "fake_repl.py"
    script.write_text(FAKE_REPL, encoding="utf-8")
    bench = tmp_path / "minif2f.jsonl"
    bench.write_text("".join(
        json.dumps({"name": f"mathd_algebra_{i}", "split": "test",
                    "formal_statement": f"theorem mathd_algebra_{i} : P :=", "header": "import Mathlib\n"}) + "\n"
        for i in range(4)))
    raw = yaml.safe_load((ROOT / "configs" / "search_eval_mac.yaml").read_text())
    raw["base_config"] = str(ROOT / "configs" / "search.yaml")
    raw["lean"] = {"command": [sys.executable, "-I", str(script)], "timeout": 5, "startup_timeout": 10}
    raw["benchmarks"]["minif2f"]["path"] = str(bench)
    raw["msm_dir"] = None
    cfg = tmp_path / "mac.yaml"
    cfg.write_text(yaml.safe_dump(raw, allow_unicode=True))
    return tmp_path, cfg, models


def test_search_eval_runs_on_the_local_backend(eval_setup):
    tmp, cfg, models = eval_setup
    out = tmp / "out"
    args = ["search-eval", "--config", str(cfg), "--out", str(out), "--yes",
            "--condition", "lm", "--condition", "random_l0.3",
            "--model-path", "/Users/someone/models/x.gguf"]
    assert cli.main(args) == 0
    rows = [r for p in sorted(out.glob("minif2f/*/seed0/summary.jsonl")) for r in se.read_summaries(p)]
    assert len(rows) == 8 and all(r["solved"] for r in rows)
    assert {r["proof"].split("by\n", 1)[1] for r in rows} == {"  a\n  close\n"}
    run = json.loads((out / "runs.jsonl").read_text().splitlines()[0])
    assert run["prover"]["backend"] == "llama_cpp" and run["prover"]["model_path"] == "x.gguf"
    assert run["budget"] == {"max_expansions": 12, "wall_time": 180, "max_depth": 16}
    assert models and all(m.closed for m in models)
    # the report reads it, with the baseline and one other condition
    assert cli.main(["search-report", "--runs", str(out), "--n-boot", "20", "--no-figures"]) == 0
    assert (out / "report" / "report.md").exists()


def test_runner_check_and_single_run_use_the_backend_flag(eval_setup):
    from markov_lean.search import __main__ as search_main

    tmp, _, models = eval_setup
    script = tmp / "fake_repl.py"
    cfg = tmp / "search.yaml"
    cfg.write_text(yaml.safe_dump({
        "prover": {"n": 4, "max_tokens": 32},
        "lean": {"command": [sys.executable, "-I", str(script)], "timeout": 5, "startup_timeout": 10},
        "search": {"max_expansions": 6}, "prior": {"factory": "lm"}}))
    out = tmp / "single"
    rc = search_main.main(["--config", str(cfg), "--benchmark", str(tmp / "minif2f.jsonl"), "--out", str(out),
                           "--limit", "2", "--backend", "llama_cpp", "--model-path", "x.gguf"])
    assert rc == 0 and len((out / "summary.jsonl").read_text().splitlines()) == 2
    assert models[-1].closed
    # --check runs the prover and the REPL; the toy REPL can't do omega, so it reports 1
    assert search_main.main(["--config", str(cfg), "--check", "--backend", "llama_cpp",
                             "--model-path", "x.gguf"]) == 1


def test_mac_config_matches_the_ec2_conditions():
    mac = yaml.safe_load((ROOT / "configs" / "search_eval_mac.yaml").read_text())
    ec2 = yaml.safe_load((ROOT / "configs" / "search_eval.yaml").read_text())
    for name, spec in mac["conditions"].items():
        assert spec == ec2["conditions"][name]
    assert mac["subset_seed"] == ec2["subset_seed"] and mac["workers"] == 1
    assert mac["benchmarks"]["minif2f"] == ec2["benchmarks"]["minif2f"]
    cfg = se.EvalConfig.load(ROOT / "configs" / "search_eval_mac.yaml")
    assert cfg.search_config["prover"]["backend"] == "llama_cpp"


def test_mac_smoke_doc_uses_real_flags():
    text = (ROOT / "MAC_SMOKE.md").read_text()
    known = {a for act in se.build_parser()._actions for a in act.option_strings}
    for line in text.splitlines():
        if "search-eval" in line and line.lstrip().startswith(("markov-lean", ".venv", "$E", "E=")):
            for tok in line.split():
                if tok.startswith("--"):
                    assert tok.split("=")[0] in known, tok


@pytest.mark.skipif(not os.environ.get("MARKOV_LEAN_TEST_GGUF"), reason="set MARKOV_LEAN_TEST_GGUF to a GGUF file")
def test_real_llama_cpp_model():
    pytest.importorskip("llama_cpp")
    path = os.environ["MARKOV_LEAN_TEST_GGUF"]
    model = lp.LlamaCppModel(path, n_ctx=1024)
    toks = model.tokenize("theorem t (a b : ℕ) : a + b = b + a := by\n")
    first = model.feed(toks)
    again = model.feed(toks)
    assert first.shape == (model.n_vocab,) and np.allclose(first, again, atol=1e-3)
    p = lp.LlamaCppProver(path, n=3, max_tokens=16, seed=0, local_model=model)
    a = p.complete("Complete the following Lean 4 code:\n\n```lean4\n", 3)
    b = p.complete("Complete the following Lean 4 code:\n\n```lean4\n", 3)
    assert [c["text"] for c in a] == [c["text"] for c in b]
    assert all(len(c["logprobs"]["token_logprobs"]) == len(c["logprobs"]["text_offset"]) for c in a)
    p.close()
