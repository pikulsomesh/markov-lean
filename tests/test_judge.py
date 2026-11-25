"""Tests for the MSC judges. The Anthropic client is always a fake: no test calls the API."""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from markov_lean import judge, msc

FIX = Path(__file__).parent / "fixtures" / "judge"
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    # make sure nothing can reach the real API even if a key is set in the shell
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def cfg():
    return judge.load_config(judge.DEFAULT_CONFIG)


@pytest.fixture
def steps():
    return pd.read_json(FIX / "steps.jsonl", lines=True)


@pytest.fixture
def proofs(steps, cfg):
    return judge.proofs_from_steps(steps, cfg.max_steps_per_proof, cfg.max_state_chars)


# ----------------------------------------------------------------------------- fake client


class FakeBatches:
    def __init__(self, results_by_custom_id=None, statuses=None):
        self.created: list[list[dict]] = []
        self.results_by_custom_id = results_by_custom_id or {}
        self.statuses = statuses or {}
        self.retrieved = 0

    def create(self, requests):
        self.created.append(requests)
        return NS(id=f"msgbatch_{len(self.created)}", processing_status="in_progress")

    def retrieve(self, batch_id):
        self.retrieved += 1
        seq = self.statuses.get(batch_id, ["ended"])
        status = seq.pop(0) if len(seq) > 1 else seq[0]
        return NS(id=batch_id, processing_status=status, request_counts=NS(processing=0))

    def results(self, batch_id):
        idx = int(batch_id.split("_")[1]) - 1
        reqs = self.created[idx] if idx < len(self.created) else []
        out = [self.results_by_custom_id[r["custom_id"]] for r in reqs if r["custom_id"] in self.results_by_custom_id]
        return list(reversed(out))  # results arrive in any order


class FakeClient:
    def __init__(self, batches):
        self.messages = NS(batches=batches)


def ok(cid, payload, stop="end_turn"):
    content = [NS(type="thinking", thinking="", signature="sig"), NS(type="text", text=json.dumps(payload))]
    return NS(custom_id=cid, result=NS(type="succeeded",
                                       message=NS(stop_reason=stop, stop_details=None, content=content)))


# ----------------------------------------------------------------------------- msc


def test_msc_has_63_top_level_codes():
    assert len(msc.MSC2020_TOP) == 63
    assert all(re.fullmatch(r"\d\d", c) for c in msc.CODES)
    assert list(msc.CODES) == sorted(msc.CODES)
    for gone in ("02", "04", "09", "10", "21", "27", "36", "48", "50", "56", "69", "71", "73", "79", "84"):
        assert gone not in msc.MSC2020_TOP
    assert set(msc.LEAN_COMMON) <= set(msc.CODES)


def test_msc_titles_from_official_pdf():
    assert msc.title("00") == "General and overarching topics; collections"
    assert msc.title("13") == "Commutative algebra"
    assert msc.title("18") == "Category theory; homological algebra"
    assert msc.title("20") == "Group theory and generalizations"


@pytest.mark.skipif(not os.environ.get("MSC2020_CSV"), reason="set MSC2020_CSV to the official msc2020.org CSV")
def test_msc_titles_against_official_csv():
    official = {}
    with open(os.environ["MSC2020_CSV"], encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if row and re.fullmatch(r"\d\d-XX", row[0].strip()):
                official[row[0][:2]] = re.sub(r"\s*\[.*$|\s*\{.*$", "", row[1]).strip()
    assert set(official) == set(msc.MSC2020_TOP)
    assert official == msc.MSC2020_TOP


@pytest.mark.parametrize("raw,expected", [
    ("26", "26"), ("26-XX", "26"), ("26A42", "26"), (26, "26"), (5, "05"), ("5", "05"),
    ("36", None), ("abc", None), (None, None), ("", None),
])
def test_normalize(raw, expected):
    assert msc.normalize(raw) == expected


# ----------------------------------------------------------------------------- requests


def test_proofs_grouped_and_ordered(proofs):
    by_id = {p.proof_id: p for p in proofs}
    p = by_id["mathlib:Real.add_pow_le_pow_mul_pow_of_sq_le_sq"]
    assert [s.step for s in p.steps] == [0, 1, 2]
    assert p.statement.startswith("a b c : ℝ")


def test_long_proofs_are_sampled_and_states_cut():
    rows = [{"proof_id": "x", "step": i, "theorem": "T", "tactic": "simp",
             "state_before": "y" * 5000, "state_after": "z"} for i in range(100)]
    [p] = judge.proofs_from_steps(pd.DataFrame(rows), max_steps=10, max_state_chars=200)
    assert len(p.steps) == 10 and p.n_steps_total == 100
    assert p.steps[0].step == 0 and p.steps[-1].step == 99
    assert len(p.steps[0].state_before) < 260
    assert "10 of them are shown" in judge.render_user(p)


def test_build_requests_one_per_judge(proofs, cfg):
    reqs, man = judge.build_requests(proofs, cfg)
    assert len(reqs) == 2 * len(proofs)
    for r in reqs:
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", r["custom_id"])
        p = r["params"]
        assert p["model"] in (HAIKU, SONNET)
        assert p["output_config"]["effort"] == "low"
        assert p["output_config"]["format"]["type"] == "json_schema"
        assert p["messages"][-1]["role"] == "user"  # no prefill
        assert "thinking" not in p and "temperature" not in p and "fallbacks" not in p
    assert {m["judge"] for m in man.values()} == {HAIKU, SONNET}
    assert len({r["custom_id"] for r in reqs}) == len(reqs)


def test_schema_enumerates_codes():
    s = judge.output_schema()
    assert s["properties"]["proof_msc"]["enum"] == list(msc.CODES)
    assert s["additionalProperties"] is False
    assert s["properties"]["steps"]["items"]["additionalProperties"] is False


# ----------------------------------------------------------------------------- cost and guard


def test_estimate_is_positive_and_batch_discounted(proofs, cfg):
    reqs, man = judge.build_requests(proofs, cfg)
    est = judge.estimate_cost(reqs, man, cfg)
    assert set(est.per_judge) == {HAIKU, SONNET}
    assert est.per_judge[SONNET]["dollars"] > est.per_judge[HAIKU]["dollars"] > 0
    cfg.batch_discount = 1.0
    full = judge.estimate_cost(reqs, man, cfg)
    assert full.total == pytest.approx(2 * est.total)


def test_budget_guard_blocks_before_any_call(proofs, cfg, tmp_path):
    reqs, man = judge.build_requests(proofs, cfg)
    batches = FakeBatches()
    with pytest.raises(judge.BudgetExceeded):
        judge.submit(reqs, man, cfg, tmp_path, client=FakeClient(batches), max_dollars=1e-9, echo=lambda s: None)
    assert batches.created == []
    assert not (tmp_path / "batches.json").exists()


def test_cli_cap_cannot_raise_config_cap(proofs, cfg):
    over = cfg.max_dollars + 10.0  # just over the shipped config cap
    est = judge.CostEstimate({HAIKU: {"requests": 1, "input_tokens": 1, "output_tokens": 1, "dollars": over}}, 0.5)
    with pytest.raises(judge.BudgetExceeded):
        judge.check_budget(est, cfg, max_dollars=cfg.max_dollars * 10)


def test_dry_run_writes_requests_and_calls_nothing(proofs, cfg, tmp_path):
    reqs, man = judge.build_requests(proofs, cfg)
    batches = FakeBatches()
    ids = judge.submit(reqs, man, cfg, tmp_path, client=FakeClient(batches), dry_run=True, echo=lambda s: None)
    assert ids == [] and batches.created == []
    assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == len(reqs)
    assert json.loads((tmp_path / "manifest.json").read_text()).keys() == man.keys()


def test_submit_chunks(proofs, cfg, tmp_path):
    cfg.chunk_size = 3
    reqs, man = judge.build_requests(proofs, cfg)
    batches = FakeBatches()
    ids = judge.submit(reqs, man, cfg, tmp_path, client=FakeClient(batches), echo=lambda s: None)
    assert len(ids) == len(batches.created) == 2  # 6 requests in chunks of 3
    assert json.loads((tmp_path / "batches.json").read_text())["batch_ids"] == ids


def test_nothing_in_the_run_dir_mentions_a_key(proofs, cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-secret")
    reqs, man = judge.build_requests(proofs, cfg)
    judge.submit(reqs, man, cfg, tmp_path, client=FakeClient(FakeBatches()), echo=lambda s: None)
    for f in tmp_path.iterdir():
        assert "sk-ant-test-secret" not in f.read_text()


def test_estimate_for_1000_proofs_fits_the_credit(cfg):
    proofs = judge.synthetic_proofs(1000, mean_steps=10, mean_state_chars=600)
    reqs, man = judge.build_requests(proofs, cfg)
    est = judge.estimate_cost(reqs, man, cfg)
    assert 0 < est.total < cfg.max_dollars


# ----------------------------------------------------------------------------- collection


def _results_for(reqs, man, *, haiku_code="26", sonnet_code="26"):
    out = {}
    for r in reqs:
        entry = man[r["custom_id"]]
        code = haiku_code if entry["judge"] == HAIKU else sonnet_code
        out[r["custom_id"]] = ok(r["custom_id"], {
            "proof_msc": code, "proof_confidence": 0.9,
            "steps": [{"step": s, "msc": code, "confidence": 0.8} for s in entry["steps"]],
        })
    return out


def test_collect_writes_schema_rows(proofs, cfg, tmp_path):
    reqs, man = judge.build_requests(proofs, cfg)
    batches = FakeBatches(_results_for(reqs, man, sonnet_code="11"))
    ids = judge.submit(reqs, man, cfg, tmp_path, client=FakeClient(batches), echo=lambda s: None)
    labels, issues = judge.collect(FakeClient(batches), ids, man)
    assert list(labels.columns) == judge.LABEL_COLUMNS
    assert issues == []
    n_steps = sum(len(p.steps) for p in proofs)
    assert len(labels) == 2 * (n_steps + len(proofs))
    assert set(labels[labels.judge == SONNET].msc) == {"11"}
    assert (labels[labels.step == -1].confidence == 0.9).all()


def test_parse_result_handles_stop_reasons_and_errors():
    entry = {"proof_id": "p", "judge": HAIKU, "steps": [0, 1]}
    refusal = NS(custom_id="c", result=NS(type="succeeded", message=NS(
        stop_reason="refusal", stop_details=NS(type="refusal", category="cyber"), content=[])))
    rows, issue = judge.parse_result(refusal, entry)
    assert rows == [] and issue["issue"] == "refusal" and issue["detail"] == "cyber"

    cut = ok("c", {}, stop="max_tokens")
    rows, issue = judge.parse_result(cut, entry)
    assert rows == [] and issue["issue"] == "max_tokens"

    for rtype in ("errored", "expired", "canceled"):
        res = NS(custom_id="c", result=NS(type=rtype, error=NS(type="api_error")))
        rows, issue = judge.parse_result(res, entry)
        assert rows == [] and issue["issue"] == rtype

    bad = NS(custom_id="c", result=NS(type="succeeded", message=NS(
        stop_reason="end_turn", stop_details=None, content=[NS(type="text", text="not json")])))
    assert judge.parse_result(bad, entry)[1]["issue"] == "bad_json"


def test_parse_result_drops_invalid_and_unknown_steps():
    entry = {"proof_id": "p", "judge": HAIKU, "steps": [0, 1]}
    res = ok("c", {"proof_msc": "26", "proof_confidence": 1.7, "steps": [
        {"step": 0, "msc": "36", "confidence": 0.5},   # not an MSC 2020 code
        {"step": 1, "msc": "26A42", "confidence": -2},  # cut to 26, confidence clipped
        {"step": 7, "msc": "26", "confidence": 0.5},   # not a step we showed
    ]})
    rows, issue = judge.parse_result(res, entry)
    assert [(r["step"], r["msc"], r["confidence"]) for r in rows] == [(-1, "26", 1.0), (1, "26", 0.0)]
    assert issue["issue"] == "partial" and "1 steps missing" in issue["detail"]


def test_wait_polls_until_ended():
    batches = FakeBatches(statuses={"msgbatch_1": ["in_progress", "in_progress", "ended"]})
    sleeps = []
    judge.wait(FakeClient(batches), ["msgbatch_1"], 5, 1, sleep=sleeps.append, echo=lambda s: None)
    assert sleeps == [5, 5] and batches.retrieved == 3


# ----------------------------------------------------------------------------- agreement


def _labels(rows):
    return judge.labels_frame([dict(zip(judge.LABEL_COLUMNS, r)) for r in rows])


def test_agreement_and_kappa():
    rows = []
    for i, (a, b) in enumerate([("26", "26"), ("11", "11"), ("26", "11"), ("20", "20")]):
        rows += [(f"p{i}", 0, HAIKU, a, 0.8), (f"p{i}", 0, SONNET, b, 0.6),
                 (f"p{i}", -1, HAIKU, a, 0.9), (f"p{i}", -1, SONNET, a, 0.9)]
    stats = judge.agreement(_labels(rows), HAIKU, SONNET)
    assert stats["step"]["n"] == 4 and stats["step"]["agreement"] == 0.75
    assert stats["step"]["top_disagreements"] == [{"a": "26", "b": "11", "n": 1}]
    assert stats["proof"]["agreement"] == 1.0 and stats["proof"]["kappa"] == 1.0
    # kappa by hand: po = .75, pe = (2/4*1/4)+(1/4*2/4)+(1/4*1/4) = 5/16
    assert stats["step"]["kappa"] == pytest.approx((0.75 - 5 / 16) / (1 - 5 / 16))


def test_consensus_keeps_agreed_rows_only():
    labels = _labels([("p", 0, HAIKU, "26", 0.8), ("p", 0, SONNET, "26", 0.6),
                      ("p", 1, HAIKU, "26", 0.8), ("p", 1, SONNET, "11", 0.6)])
    c = judge.consensus(labels, HAIKU, SONNET)
    assert c[["step", "judge", "msc"]].values.tolist() == [[0, "consensus", "26"]]
    assert c.confidence.iloc[0] == pytest.approx(0.7)


# ----------------------------------------------------------------------------- validation


def test_score_known_msc180_style():
    known = pd.read_csv(FIX / "known.csv", dtype=str)
    labels = _labels([("k1", -1, HAIKU, "11", 0.9), ("k2", -1, HAIKU, "54", 0.6),
                      ("k3", -1, HAIKU, "20", 0.9), ("k1", -1, SONNET, "11", 0.9)])
    rep = judge.score_known(labels, known)
    assert rep[HAIKU]["top1_accuracy"] == pytest.approx(2 / 3)
    assert rep[HAIKU]["any_match_accuracy"] == 1.0  # 54 is k2's secondary code
    assert rep[SONNET]["coverage"] == pytest.approx(1 / 3)
    assert rep[HAIKU]["per_area_top1"]["26"] == {"accuracy": 0.0, "n": 1}


def test_known_statements_become_stepless_requests(cfg):
    known = pd.read_csv(FIX / "known.csv", dtype=str)
    proofs = judge.proofs_from_known(known)
    reqs, man = judge.build_requests(proofs, cfg)
    assert all(m["steps"] == [] for m in man.values())
    assert "No proof steps are given" in reqs[0]["params"]["messages"][0]["content"]


def test_score_synthetic(steps):
    labels = _labels([("synthetic:g0", 0, HAIKU, "20", 0.9), ("synthetic:g0", 1, HAIKU, "08", 0.4)])
    assert judge.score_synthetic(labels, steps) == {HAIKU: {"n": 2, "accuracy": 0.5}}


# ----------------------------------------------------------------------------- command line


def test_cli_estimate_and_dry_run(tmp_path, capsys):
    assert judge.main(["estimate", "--steps", str(FIX / "steps.jsonl")]) == 0
    assert "total: $" in capsys.readouterr().out
    client = FakeClient(FakeBatches())
    assert judge.main(["submit", "--steps", str(FIX / "steps.jsonl"), "--out", str(tmp_path),
                       "--dry-run"], client=client) == 0
    assert client.messages.batches.created == []
    assert judge.main(["submit", "--steps", str(FIX / "steps.jsonl"), "--out", str(tmp_path),
                       "--max-dollars", "0.000001"], client=client) == 2
    assert client.messages.batches.created == []
