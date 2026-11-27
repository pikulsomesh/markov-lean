"""Tests for the judge-context control. Ollama is the mock server of test_ollama_judge: no model needed."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from markov_lean import context_control as cc
from markov_lean import judge
from markov_lean import ollama_judge as oj
from test_ollama_judge import MODEL, MockOllama

ROOT = Path(__file__).resolve().parents[1]
STEP_RE = re.compile(r"^\[step (\d+)\] tactic: (\S+)", re.M)
AREAS = ["11", "26", "05"]


def make_steps(n_proofs: int = 6, n_steps: int = 6) -> pd.DataFrame:
    """Proofs whose tactics name their area, ``t<area>_<i>``: two runs of areas per proof."""
    rows = []
    for p in range(n_proofs):
        a, b = AREAS[p % 3], AREAS[(p + 1) % 3]
        for i in range(n_steps):
            area = a if i < n_steps // 2 else b
            rows.append({"proof_id": f"mathlib:P{p}", "step": i, "source": "leandojo", "split": "train",
                         "theorem": f"P{p}", "file_path": f"Mathlib/Folder{p % 2}/X.lean",
                         "tactic": f"t{area}_{i}", "state_before": f"goal {p}.{i}",
                         "state_after": f"goal {p}.{i + 1}" if i < n_steps - 1 else "no goals"})
    return pd.DataFrame(rows)


def content_judge(body: dict) -> tuple[int, dict]:
    """Labels each step by its tactic: what a judge that reads only the step itself would do."""
    user = body["messages"][-1]["content"]
    steps = [{"step": int(s), "msc": t[1:3], "confidence": 0.9} for s, t in STEP_RE.findall(user)]
    return 200, MockOllama.reply(json.dumps({"steps": steps, "proof_msc": steps[0]["msc"], "proof_confidence": 0.9}))


def position_judge(body: dict) -> tuple[int, dict]:
    """Labels the first half of the steps as shown 11 and the rest 26: a judge that only smooths
    along the order it reads."""
    user = body["messages"][-1]["content"]
    ids = [int(s) for s, _ in STEP_RE.findall(user)]
    steps = [{"step": s, "msc": "11" if k < len(ids) // 2 else "26", "confidence": 0.9} for k, s in enumerate(ids)]
    return 200, MockOllama.reply(json.dumps({"steps": steps, "proof_msc": "11", "proof_confidence": 0.9}))


@pytest.fixture
def cfg():
    return judge.load_config(judge.DEFAULT_CONFIG)


def _run(m, steps, out, cfg, judge_fn, **kw):
    m.script = [judge_fn] * 1000
    s = oj.LocalSettings(model=MODEL)
    return cc.run_label(steps, out, s, client=oj.OllamaClient(m.url, timeout=10), cfg=cfg,
                        sleep=lambda _: None, echo=lambda _: None, **kw)


def test_shuffled_view_hides_order_and_keeps_content():
    steps = make_steps(1, 8)
    (proof,) = judge.proofs_from_steps(steps)
    view, back, order = cc.shuffled_view(proof, seed=0)
    assert order != [s.step for s in proof.steps] and sorted(order) == [s.step for s in proof.steps]
    assert all(cc.ID_RANGE[0] <= s.step < cc.ID_RANGE[1] for s in view.steps)
    assert sorted(back.values()) == [s.step for s in proof.steps]
    assert [back[s.step] for s in view.steps] == order
    original = {s.step: s for s in proof.steps}
    for s in view.steps:  # each shown step keeps its own tactic and goal
        assert (s.tactic, s.state_before) == (original[back[s.step]].tactic, original[back[s.step]].state_before)
    # the prompt still ends with the true final state, and shows the same theorem and statement
    text = judge.render_user(view)
    assert text.rstrip().endswith("no goals") and proof.statement in text
    assert cc.shuffled_view(proof, seed=0)[2] == order  # deterministic per proof and seed
    assert cc.shuffled_view(proof, seed=1)[2] != order or len(proof.steps) < 3


def test_content_judge_gives_the_same_labels_in_both_conditions(tmp_path, cfg):
    steps = make_steps()
    with MockOllama() as m:
        rc = _run(m, steps, tmp_path, cfg, content_judge, n_proofs=6)
        assert rc == 0 and len(m.chats) == 12
    lab = pd.read_parquet(tmp_path / "labels.parquet")
    lab = lab[lab.step >= 0]  # this mock names the proof after the first step it reads
    o = lab[lab.judge == cc.condition_judge(MODEL, "v1", "ordered")]
    s = lab[lab.judge == cc.condition_judge(MODEL, "v1", "shuffled")]
    key = ["proof_id", "step", "msc"]
    assert o[key].sort_values(key).reset_index(drop=True).equals(s[key].sort_values(key).reset_index(drop=True))
    res = cc.compare(tmp_path, echo=lambda _: None, n_boot=200)
    assert res["proofs_compared"] == 6 and res["steps_compared"] == 36
    assert res["step_agreement_ordered_vs_shuffled"]["raw_agreement"] == 1.0
    assert res["ratio_difference"] == pytest.approx(0.0)
    # one switch in five moves per proof; the shuffle expectation of 3+3 steps is 18/30
    assert res["conditions"]["ordered"]["switch_rate"] == pytest.approx(0.2)
    assert res["conditions"]["ordered"]["ratio"] == pytest.approx(0.2 / 0.6)
    assert (tmp_path / "REPORT.md").read_text().startswith("# Judge-context control")
    assert cc.complete(tmp_path)


def test_smoothing_judge_loses_its_coherence_when_shuffled(tmp_path, cfg):
    steps = make_steps(n_proofs=12, n_steps=8)
    with MockOllama() as m:
        assert _run(m, steps, tmp_path, cfg, position_judge, n_proofs=12) == 0
    res = cc.compare(tmp_path, echo=lambda _: None, n_boot=200)
    o, s = res["conditions"]["ordered"], res["conditions"]["shuffled"]
    assert o["switch_rate"] == pytest.approx(1 / 7)  # one switch, at the middle of what it read
    assert s["switch_rate"] > o["switch_rate"] and s["ratio"] > o["ratio"]
    assert res["ratio_difference"] > 0 and res["ratio_difference_ci"][0] > 0


def test_resumes_without_repeating_and_api_comparison(tmp_path, cfg):
    steps = make_steps()
    api = []
    for r in steps.itertuples(index=False):
        for j in ("claude-haiku", "claude-sonnet"):
            api.append({"proof_id": r.proof_id, "step": r.step, "judge": j, "msc": r.tactic[1:3], "confidence": 0.9})
    api_labels = judge.labels_frame(api)
    with MockOllama() as m:
        assert _run(m, steps, tmp_path, cfg, content_judge, labels=api_labels, n_proofs=6, limit_proofs=2) == 1
        assert len(m.chats) == 4 and not cc.complete(tmp_path)
        assert _run(m, steps, tmp_path, cfg, content_judge, labels=api_labels, n_proofs=6) == 0
        assert len(m.chats) == 12
        assert _run(m, steps, tmp_path, cfg, content_judge, labels=api_labels, n_proofs=6) == 0
        assert len(m.chats) == 12  # nothing left to do
    api_labels.to_parquet(tmp_path / "api.parquet", index=False)
    res = cc.compare(tmp_path, tmp_path / "api.parquet", echo=lambda _: None, n_boot=100)
    assert res["api"]["agreement_with_shuffled"]["raw_agreement"] == 1.0
    assert res["api"]["switching_same_proofs"]["proofs"] == 6


def test_server_error_stops_and_retries_on_resume(tmp_path, cfg):
    steps = make_steps(3, 4)
    with MockOllama() as m:
        rc = _run(m, steps, tmp_path, cfg, lambda b: (500, {"error": "boom"}), n_proofs=3)
        assert rc == 1
        assert _run(m, steps, tmp_path, cfg, content_judge, n_proofs=3) == 0
    assert json.loads((tmp_path / "summary.json").read_text())["proofs_ok_in_both"] == 3


def test_cli_compare_refuses_an_unfinished_run(tmp_path, cfg):
    steps = make_steps(3, 4)
    with MockOllama() as m:
        _run(m, steps, tmp_path, cfg, content_judge, n_proofs=3, limit_proofs=1)
    assert cc.main(["compare", "--out", str(tmp_path), "--run", str(tmp_path)]) == 1
    assert cc.main(["status", "--out", str(tmp_path), "--check"]) == 1
    assert cc.main(["compare", "--out", str(tmp_path), "--run", str(tmp_path), "--partial", "--n-boot", "50"]) == 0


def test_queue_runs_the_control_last():
    from markov_lean import mac_queue as mq

    q = mq.Queue.load(ROOT / "configs/mac_queue.yaml", root=ROOT)
    s = q.stages[-1]
    assert s.name == "qwen_context_control" and q.stages[-2].name == "minif2f_b64"
    assert "scripts/mac_context_control.sh" in s.run and s.done_when
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts/mac_context_control.sh")]).returncode == 0
    out = cc.default_out(oj.DEFAULT_MODEL, s.env.get("PROMPT", "v1"))
    assert str(out / "comparison.json") in [str(d) for d in s.done_when]
