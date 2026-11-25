"""Tests for the local Ollama judge. Ollama is a mock HTTP server on localhost: no model is needed."""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from markov_lean import cli, judge
from markov_lean import ollama_judge as oj

FIX = Path(__file__).parent / "fixtures" / "judge"
MODEL = "qwen2.5:7b-instruct"


# --------------------------------------------------------------------------- mock server


def good_answer(user: str, code: str = "11") -> dict:
    steps = [int(s) for s in re.findall(r"^\[step (\d+)\]", user, flags=re.M)]
    return {"steps": [{"step": s, "msc": code, "confidence": 0.8} for s in steps],
            "proof_msc": code, "proof_confidence": 0.9}


class MockOllama:
    """A fake Ollama: /api/version, /api/tags and /api/chat. ``script`` is a list of callables or
    (status, body) pairs used for the next chat calls in order; after it runs out, every chat call
    gets a valid answer that labels each shown step."""

    def __init__(self, version="0.6.2", models=(MODEL,)):
        self.version = version
        self.models = list(models)
        self.script: list = []
        self.chats: list[dict] = []
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, status, body):
                data = (body if isinstance(body, str) else json.dumps(body)).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/version":
                    self._send(200, {"version": mock.version})
                elif self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m, "model": m} for m in mock.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path != "/api/chat":
                    return self._send(404, {"error": "not found"})
                mock.chats.append(body)
                if body["model"] not in mock.models:
                    return self._send(404, {"error": f"model '{body['model']}' not found"})
                user = body["messages"][-1]["content"]
                if mock.script:
                    item = mock.script.pop(0)
                    status, reply = item(body) if callable(item) else item
                    if status != 200:
                        return self._send(status, reply)
                else:
                    reply = mock.reply(json.dumps(good_answer(user)))
                self._send(200, reply)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @staticmethod
    def reply(content: str, done_reason: str = "stop", prompt_tokens: int = 900, eval_tokens: int = 120) -> dict:
        return {"model": MODEL, "message": {"role": "assistant", "content": content}, "done": True,
                "done_reason": done_reason, "prompt_eval_count": prompt_tokens, "eval_count": eval_tokens}

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def ollama():
    with MockOllama() as m:
        yield m


@pytest.fixture
def cfg():
    return judge.load_config(judge.DEFAULT_CONFIG)


@pytest.fixture
def steps():
    return pd.read_json(FIX / "steps.jsonl", lines=True)


@pytest.fixture
def proofs(steps, cfg):
    return judge.proofs_from_steps(steps, cfg.max_steps_per_proof, cfg.max_state_chars)


def client(m: MockOllama) -> oj.OllamaClient:
    return oj.OllamaClient(m.url, timeout=10)


def no_sleep(_):
    pass


# --------------------------------------------------------------------------- JSON repair


@pytest.mark.parametrize("text", [
    '{"steps": [], "proof_msc": "11", "proof_confidence": 0.9}',
    '```json\n{"steps": [], "proof_msc": "11", "proof_confidence": 0.9}\n```',
    'Here is the labelling:\n{"steps": [], "proof_msc": "11", "proof_confidence": 0.9}\nHope this helps.',
    '{"steps": [{"step": 0, "msc": "11", "confidence": 0.5},], "proof_msc": "11", "proof_confidence": 0.9,}',
])
def test_parse_json_repairs_common_mistakes(text):
    assert oj.parse_json(text)["proof_msc"] == "11"


def test_parse_json_keeps_braces_inside_strings():
    data = oj.parse_json('note {x} -> {"steps": [], "proof_msc": "26", "why": "a } brace", "proof_confidence": 1}')
    assert data is None or data.get("proof_msc") == "26"
    assert oj.parse_json('{"steps": [], "proof_msc": "26", "why": "a } brace", "proof_confidence": 1}')["why"] == \
        "a } brace"


@pytest.mark.parametrize("text", [None, "", "no json here", '{"steps": [', "[1, 2]"])
def test_parse_json_gives_up_cleanly(text):
    assert oj.parse_json(text) is None


# --------------------------------------------------------------------------- one request


def test_request_uses_the_api_judges_prompt_schema_and_labels(proofs):
    s = oj.LocalSettings()
    body = oj.chat_body(proofs[0], s)
    assert body["model"] == oj.DEFAULT_MODEL == "qwen2.5:7b-instruct"
    assert body["messages"][0] == {"role": "system", "content": judge.SYSTEM_PROMPT}
    assert body["messages"][1] == {"role": "user", "content": judge.render_user(proofs[0])}
    assert body["format"] == judge.output_schema()
    assert body["format"]["properties"]["proof_msc"]["enum"] == list(judge.msc.CODES)
    assert body["options"]["temperature"] == 0 and body["options"]["num_ctx"] == 32768
    assert oj.chat_body(proofs[0], oj.LocalSettings(schema=False))["format"] == "json"
    # a retry is reseeded and slightly warmer, so it does not repeat the same failure
    retry = oj.chat_body(proofs[0], s, attempt=1)["options"]
    assert retry["seed"] == 1 and retry["temperature"] > 0


def test_label_proof_writes_schema_rows(ollama, proofs):
    proof = next(p for p in proofs if p.steps)
    res = oj.label_proof(client(ollama), proof, oj.LocalSettings())
    assert res.status == "ok" and res.attempts == 1
    assert {r["step"] for r in res.rows} == {s.step for s in proof.steps} | {judge.PROOF_LEVEL_STEP}
    assert {r["judge"] for r in res.rows} == {"ollama/qwen2.5:7b-instruct"}
    assert set(res.rows[0]) == set(judge.LABEL_COLUMNS)
    assert res.prompt_tokens == 900 and res.output_tokens == 120


def test_server_errors_are_retried_with_backoff(ollama, proofs):
    ollama.script = [(500, {"error": "loading"}), (503, {"error": "busy"})]
    waits = []
    res = oj.label_proof(client(ollama), proofs[0], oj.LocalSettings(), sleep=waits.append)
    assert res.status == "ok" and res.attempts == 3
    assert waits == [2.0, 4.0]


def test_server_that_never_recovers_is_reported(ollama, proofs):
    ollama.script = [(500, {"error": "x"})] * 4
    res = oj.label_proof(client(ollama), proofs[0], oj.LocalSettings(retries=3), sleep=no_sleep)
    assert res.status == "server_error" and res.rows == [] and res.attempts == 4


def test_bad_json_is_repaired_or_retried(ollama, proofs):
    proof = next(p for p in proofs if p.steps)
    user = judge.render_user(proof)
    fenced = "```json\n" + json.dumps(good_answer(user)) + "\n```"
    ollama.script = [(200, MockOllama.reply(fenced))]
    assert oj.label_proof(client(ollama), proof, oj.LocalSettings()).status == "ok"
    ollama.script = [(200, MockOllama.reply("I think it is number theory."))]
    res = oj.label_proof(client(ollama), proof, oj.LocalSettings(), sleep=no_sleep)
    assert res.status == "ok" and res.attempts == 2


def test_cut_off_answer_is_retried_with_more_room(ollama, proofs):
    proof = next(p for p in proofs if p.steps)
    ollama.script = [(200, MockOllama.reply('{"steps": [{"step": 0, "ms', done_reason="length"))]
    res = oj.label_proof(client(ollama), proof, oj.LocalSettings(), sleep=no_sleep)
    assert res.status == "ok"
    first, second = (c["options"]["num_predict"] for c in ollama.chats[-2:])
    assert second == 2 * first


def test_partial_answers_keep_the_most_complete_one(ollama, proofs):
    proof = max(proofs, key=lambda p: len(p.steps))
    assert len(proof.steps) >= 2
    user = judge.render_user(proof)
    full = good_answer(user)
    one = {**full, "steps": full["steps"][:1]}
    two = {**full, "steps": full["steps"][:2] + [{"step": 999, "msc": "11", "confidence": 1}]}
    ollama.script = [(200, MockOllama.reply(json.dumps(one))), (200, MockOllama.reply(json.dumps(two))),
                     (200, MockOllama.reply(json.dumps(one)))]
    res = oj.label_proof(client(ollama), proof, oj.LocalSettings(retries=2), sleep=no_sleep)
    assert res.status == "partial" and len(res.rows) == 3  # proof label + 2 valid steps; step 999 dropped
    assert "invalid" in res.issue["detail"]


def test_string_step_numbers_are_accepted(ollama, proofs):
    proof = next(p for p in proofs if p.steps)
    ans = good_answer(judge.render_user(proof))
    for s in ans["steps"]:
        s["step"] = str(s["step"])
    ollama.script = [(200, MockOllama.reply(json.dumps(ans)))]
    assert oj.label_proof(client(ollama), proof, oj.LocalSettings()).status == "ok"


def test_too_long_prompts_are_skipped_without_a_call(ollama, proofs):
    res = oj.label_proof(client(ollama), proofs[0], oj.LocalSettings(num_ctx=512))
    assert res.status == "too_long" and not ollama.chats


def test_full_context_means_a_cut_prompt(ollama, proofs):
    ollama.script = [(200, MockOllama.reply("{}", prompt_tokens=32767))]
    res = oj.label_proof(client(ollama), proofs[0], oj.LocalSettings())
    assert res.status == "too_long"


# --------------------------------------------------------------------------- stratified sample


def _fake_proofs(sizes_by_stratum: dict[str, list[int]]):
    proofs, strata = [], {}
    for h, sizes in sizes_by_stratum.items():
        for i, n in enumerate(sizes):
            pid = f"{h}:{i}"
            proofs.append(judge.Proof(pid, pid, "s", [judge.Step(k, "t", "a", "b") for k in range(n)], n))
            strata[pid] = h
    return proofs, strata


def test_stratified_sample_hits_the_budget_and_covers_strata():
    proofs, strata = _fake_proofs({"A": [10] * 100, "B": [10] * 30, "C": [5] * 10, "D": [20] * 2})
    sample = oj.stratified_sample(proofs, strata, 500, seed=0)
    n = sum(len(p.steps) for p in sample)
    assert 500 <= n < 500 + 20
    got = {h: sum(len(p.steps) for p in sample if strata[p.proof_id] == h) for h in "ABCD"}
    assert all(got[h] > 0 for h in "ABCD")
    total = {"A": 1000, "B": 300, "C": 50, "D": 40}
    for h in "AB":  # proportional to each stratum's steps, within a proof or two
        assert abs(got[h] - 500 * total[h] / 1390) <= 25
    # any prefix is close to stratified: the first half already has every stratum
    assert {strata[p.proof_id] for p in sample[: len(sample) // 2]} == set("ABCD")
    assert [p.proof_id for p in sample] == [p.proof_id for p in oj.stratified_sample(proofs, strata, 500, seed=0)]
    assert len({p.proof_id for p in sample}) == len(sample)


def test_sample_of_none_keeps_every_proof():
    proofs, strata = _fake_proofs({"A": [3] * 7, "B": [4] * 5})
    assert len(oj.stratified_sample(proofs, strata, None)) == 12


def test_strata_from_api_labels_and_folders(steps):
    folders = oj.proof_strata(steps)
    assert folders["mathlib:Real.add_pow_le_pow_mul_pow_of_sq_le_sq"] == "Analysis"
    assert folders["minif2f:mathd_numbertheory_3"] == "source:minif2f"
    labels = pd.DataFrame({"proof_id": ["minif2f:mathd_numbertheory_3"] * 2, "step": [-1, 0],
                           "judge": ["claude-haiku-5-5"] * 2, "msc": ["11", "05"], "confidence": [0.9, 0.9]})
    by_label = oj.proof_strata(steps, labels)
    assert by_label["minif2f:mathd_numbertheory_3"] == "11"
    assert by_label["mathlib:Real.add_pow_le_pow_mul_pow_of_sq_le_sq"] == "unlabelled"


def test_plan_keeps_to_the_proofs_the_api_judges_labelled(steps, cfg):
    some = sorted(steps["proof_id"].astype(str).unique())[:2]
    labels = pd.DataFrame({"proof_id": some, "step": [-1, -1], "judge": ["h", "h"], "msc": ["11", "26"],
                           "confidence": [1.0, 1.0]})
    plan = oj.plan_sample(steps, cfg, labels, "auto", None, 0)
    assert sorted(p.proof_id for p in plan) == some


# --------------------------------------------------------------------------- whole runs


def test_run_labels_resumes_and_exports(ollama, steps, cfg, tmp_path):
    out = tmp_path / "local"
    lines = []
    oj.run_label(steps, out, oj.LocalSettings(), client=client(ollama), cfg=cfg, n_steps=None,
                 limit_proofs=2, echo=lines.append, sleep=no_sleep)
    assert len(ollama.chats) == 2
    labels = pd.read_parquet(out / "labels.parquet")
    assert list(labels.columns) == judge.LABEL_COLUMNS and labels["proof_id"].nunique() == 2

    summary = oj.run_label(steps, out, oj.LocalSettings(), client=client(ollama), cfg=cfg, n_steps=None,
                           echo=lines.append, sleep=no_sleep)
    n_proofs = steps["proof_id"].nunique()
    assert len(ollama.chats) == n_proofs  # the first two were not asked again
    assert summary["proofs_done"] == n_proofs and summary["status_counts"] == {"ok": n_proofs}
    labels = pd.read_parquet(out / "labels.parquet")
    expected = sum(len(p.steps) for p in judge.proofs_from_steps(steps)) + n_proofs
    assert len(labels) == expected
    assert labels.astype({"step": "int64"}).dtypes.equals(judge.labels_frame([]).dtypes)
    assert any("already done" in l for l in lines)

    # a third run asks nothing; export rebuilds the same table without a server
    oj.run_label(steps, out, oj.LocalSettings(), client=client(ollama), cfg=cfg, n_steps=None,
                 echo=lines.append, sleep=no_sleep)
    assert len(ollama.chats) == n_proofs
    (out / "labels.parquet").unlink()
    oj.export(out, echo=lines.append)
    assert len(pd.read_parquet(out / "labels.parquet")) == expected
    issues = json.loads((out / "issues.json").read_text())
    assert issues == []


def test_server_errors_are_redone_on_resume(ollama, steps, cfg, tmp_path):
    ollama.script = [(500, {"error": "x"})] * 4
    oj.run_label(steps, tmp_path, oj.LocalSettings(), client=client(ollama), cfg=cfg, n_steps=None,
                 echo=lambda *_: None, sleep=no_sleep)
    first = oj.read_progress(tmp_path / oj.PROGRESS)
    assert len(first) == 1 and next(iter(first.values()))["status"] == "server_error"  # it stopped
    summary = oj.run_label(steps, tmp_path, oj.LocalSettings(), client=client(ollama), cfg=cfg, n_steps=None,
                           echo=lambda *_: None, sleep=no_sleep)
    assert summary["status_counts"] == {"ok": steps["proof_id"].nunique()}


def test_missing_model_and_old_server(steps, cfg, tmp_path):
    with MockOllama(models=["llama3.1:8b"]) as m:
        with pytest.raises(oj.OllamaError, match="ollama pull qwen2.5:7b-instruct"):
            oj.run_label(steps, tmp_path, oj.LocalSettings(), client=client(m), cfg=cfg, echo=lambda *_: None)
    with MockOllama(version="0.4.7") as m:
        oj.run_label(steps, tmp_path, oj.LocalSettings(), client=client(m), cfg=cfg, n_steps=None,
                     limit_proofs=1, echo=lambda *_: None)
        assert m.chats[0]["format"] == "json"


def test_a_saved_run_for_another_model_is_not_mixed(ollama, steps, cfg, tmp_path):
    oj.run_label(steps, tmp_path, oj.LocalSettings(), client=client(ollama), cfg=cfg, limit_proofs=1,
                 echo=lambda *_: None)
    ollama.models.append("llama3.1:8b-instruct-q4_K_M")
    with pytest.raises(oj.OllamaError, match="holds labels from"):
        oj.run_label(steps, tmp_path, oj.LocalSettings(model="llama3.1:8b-instruct-q4_K_M"), client=client(ollama),
                     cfg=cfg, echo=lambda *_: None)


def test_unreachable_server_is_explained(cfg, capsys):
    c = oj.OllamaClient("http://127.0.0.1:9", timeout=2)
    assert oj.check(c, oj.LocalSettings(), None, cfg) == 1
    assert "Start the Ollama app" in capsys.readouterr().out


def test_cli_check_and_label(ollama, tmp_path, capsys):
    steps_path = str(FIX / "steps.jsonl")
    assert cli.main(["local-judge", "check", "--host", ollama.url, "--steps", steps_path]) == 0
    out = capsys.readouterr().out
    assert "is pulled" in out and "Test proof" in out
    assert cli.main(["local-judge", "label", "--host", ollama.url, "--steps", steps_path,
                     "--sample-steps", "5", "--out", str(tmp_path / "o")]) == 0
    labels = pd.read_parquet(tmp_path / "o" / "labels.parquet")
    assert (labels["step"] >= 0).sum() >= 5
    assert cli.main(["local-judge", "label", "--host", ollama.url, "--steps", steps_path,
                     "--model", "missing:1b", "--out", str(tmp_path / "p")]) == 1
    assert "ollama pull missing:1b" in capsys.readouterr().err


def test_cli_label_defaults_to_the_run_folder(ollama, steps, tmp_path):
    run = tmp_path / "run"
    (run / "judge").mkdir(parents=True)
    steps.to_parquet(run / "judge" / "sample_steps.parquet", index=False)
    assert oj.main(["label", "--host", ollama.url, "--run", str(run), "--all"]) == 0
    assert (run / "judge" / "local" / "qwen2.5_7b-instruct" / "labels.parquet").exists()
