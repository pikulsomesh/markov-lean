"""Tests for the judge prompt versions (prompts.py) and the gold set's dev/test split.

No model is called: the local judge talks to the mock Ollama server from test_ollama_judge.py.
"""
from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from markov_lean import cli, judge, msc
from markov_lean import gold as G
from markov_lean import judge_accuracy as A
from markov_lean import ollama_judge as oj
from markov_lean import prompts as P

# ---------------------------------------------------------------- v1 stays the API judges' prompt


def test_v1_is_the_api_prompt_and_the_default():
    v1 = P.get("v1")
    assert P.DEFAULT_VERSION == "v1"
    assert v1.system is judge.SYSTEM_PROMPT
    assert v1.render_user is judge.render_user and v1.schema is judge.output_schema
    proof = judge.Proof("p", "T", "⊢ 1 = 1", [judge.Step(0, "rfl", "⊢ 1 = 1", "no goals")], 1)
    body = oj.chat_body(proof, oj.LocalSettings())
    assert body["messages"][0]["content"] == judge.SYSTEM_PROMPT
    assert body["messages"][1]["content"] == judge.render_user(proof)
    assert body["format"] == judge.output_schema()
    assert oj.judge_name("qwen2.5:7b-instruct") == "ollama/qwen2.5:7b-instruct"


def test_api_requests_do_not_use_v2():
    cfg = judge.load_config(judge.DEFAULT_CONFIG)
    proof = judge.Proof("p", "T", "⊢ 1 = 1", [], 0)
    requests, _ = judge.build_requests([proof], cfg)
    assert all(r["params"]["system"] == judge.SYSTEM_PROMPT for r in requests)
    assert all("topic" not in json.dumps(r["params"]["output_config"]) for r in requests)


def test_unknown_version():
    with pytest.raises(ValueError, match="unknown prompt version"):
        P.get("v9")


# ---------------------------------------------------------------- v2


def test_v2_schema_order_and_parsing():
    sch = P.output_schema_v2()
    assert list(sch["properties"]) == ["topic", "proof_msc", "proof_confidence", "steps"]
    assert sch["properties"]["proof_msc"]["enum"] == list(msc.CODES)
    entry = {"proof_id": "p", "judge": "ollama/m@v2", "steps": [0]}
    out = {"topic": "finite group of order 6", "proof_msc": "20", "proof_confidence": 0.8,
           "steps": [{"step": 0, "msc": "20", "confidence": 0.7}]}
    rows, issue = judge.rows_from_output(out, entry)
    assert issue is None and {r["msc"] for r in rows} == {"20"} and {r["step"] for r in rows} == {-1, 0}


def test_v2_prompt_names_every_gold_class_and_every_code():
    text = P.SYSTEM_PROMPT_V2
    gold_classes = set(G.read_gold().query("status == 'strict'").msc)
    for c in gold_classes | {"13", "16", "28", "46", "51", "52", "55"}:
        assert re.search(rf"^- {c} ", text, re.M) or f", {c} " in text, c
    for c in msc.CODES:  # the full list is there, so any code can be answered
        assert f"\n{c} {msc.MSC2020_TOP[c]}" in text
    # 03 is described last in the guide, and only for statements about logic or set theory itself
    guide = text.split("Area guide")[1].split("When two codes")[0]
    assert guide.rfind("- 03 ") > max(guide.rfind(f"- {c} ") for c in ("05", "20", "26", "40", "54"))


def test_v2_examples_are_valid_and_not_gold_problems():
    """No example is a gold problem: neither the whole example nor its conclusion (after ⊢)
    appears in any gold statement, strict, family or excluded, ignoring spaces and symbols.
    (Word-overlap scores are no use here: short Lean statements share most of their tokens.)"""
    assert len(P.V2_EXAMPLES) >= 10
    for _, code, _ in P.V2_EXAMPLES:
        assert msc.normalize(code) == code

    def squash(s: str) -> str:
        return re.sub(r"\W+", "", s).lower()

    gold_sq = [squash(s) for s in G.read_gold().statement if s]
    for ex, _, _ in P.V2_EXAMPLES:
        whole, goal = squash(ex), squash(ex.split("⊢")[-1])
        assert not any(whole in g or g in whole for g in gold_sq), ex
        if len(goal) >= 8:  # a short conclusion such as "b ^ 7 = 1" is too common to compare alone
            assert not any(goal in g for g in gold_sq), ex


def test_v2_user_message_for_a_sorry_step_and_for_a_real_proof():
    stmt = "G : Type\ninst : Group G\n⊢ CommGroup G"
    sorry = judge.Proof("g", "ex", stmt, [judge.Step(0, "sorry", stmt, "")], 1)
    u = P.render_user_v2(sorry)
    assert u.count("CommGroup G") == 1  # the goal is not printed twice
    assert re.findall(r"^\[step (\d+)\]", u, re.M) == ["0"]
    assert "No proof is given" in u
    real = judge.Proof("r", "ex", stmt, [judge.Step(0, "intro a b", stmt, "a b : G\n⊢ a * b = b * a"),
                                        judge.Step(1, "simp", "a b : G\n⊢ a * b = b * a", "no goals")], 2)
    u = P.render_user_v2(real)
    assert u.count("CommGroup G") == 1 and "(the statement above)" in u
    assert re.findall(r"^\[step (\d+)\]", u, re.M) == ["0", "1"]


def test_v2_chat_body_and_judge_name():
    proof = judge.Proof("p", "T", "⊢ 1 = 1", [judge.Step(0, "sorry", "⊢ 1 = 1", "")], 1)
    s = oj.LocalSettings(prompt="v2")
    body = oj.chat_body(proof, s)
    assert body["messages"][0]["content"] == P.SYSTEM_PROMPT_V2
    assert body["format"] == P.output_schema_v2()
    assert body["options"]["num_predict"] == oj.LocalSettings().num_predict(1) + 48
    assert oj.judge_name("qwen2.5:7b-instruct", "v2") == "ollama/qwen2.5:7b-instruct@v2"
    # the v2 system prompt is a shared prefix for every problem, and fits the default context easily
    assert oj.estimated_prompt_tokens(proof, s) < 8000


def test_local_judge_v2_end_to_end_and_no_mixing(tmp_path):
    from test_ollama_judge import MockOllama

    steps_path = tmp_path / "gold_steps.parquet"
    assert cli.main(["gold-set", "steps", "--out", str(steps_path), "--strict-only", "--per-class", "2",
                     "--eval-split", "test"]) == 0
    out = tmp_path / "v2"
    with MockOllama() as m:
        assert cli.main(["local-judge", "label", "--host", m.url, "--steps", str(steps_path), "--all",
                         "--stratify-by", "none", "--prompt", "v2", "--out", str(out)]) == 0
        assert all(c["messages"][0]["content"] == P.SYSTEM_PROMPT_V2 for c in m.chats)
        # resuming the same folder with another prompt is refused
        assert cli.main(["local-judge", "label", "--host", m.url, "--steps", str(steps_path), "--all",
                         "--stratify-by", "none", "--out", str(out)]) == 1
    labels = pd.read_parquet(out / "labels.parquet")
    assert set(labels.judge) == {"ollama/qwen2.5:7b-instruct@v2"}
    assert json.loads((out / "summary.json").read_text())["prompt"] == "v2"
    assert cli.main(["judge-accuracy", "--labels", str(out / "labels.parquet"), "--eval-split", "test",
                     "--out", str(tmp_path / "acc"), "--n-boot", "20", "--no-figures"]) == 0
    rep = json.loads((tmp_path / "acc" / "judge_accuracy.json").read_text())
    r = rep["judges"]["ollama/qwen2.5:7b-instruct@v2"]
    assert r["strict"]["n"] == len(pd.read_parquet(steps_path)) and r["n_from_step_label"] == 0
    assert rep["settings"]["eval_split"] == "test"
    assert "test half of the fixed dev/test split" in (tmp_path / "acc" / "judge_accuracy.md").read_text()


def test_default_out_folder_has_the_prompt_suffix(tmp_path, monkeypatch):
    from test_ollama_judge import MockOllama

    steps_path = tmp_path / "s.parquet"
    G.to_steps(G.read_gold().head(40), statuses=("strict",)).to_parquet(steps_path)
    monkeypatch.chdir(tmp_path)
    with MockOllama() as m:
        assert cli.main(["local-judge", "label", "--host", m.url, "--steps", str(steps_path), "--all",
                         "--stratify-by", "none", "--prompt", "v2"]) == 0
    assert (tmp_path / "results" / "local_judge" / "qwen2.5_7b-instruct_v2" / "labels.parquet").exists()


# ---------------------------------------------------------------- dev/test split


def test_committed_split_matches_the_function():
    gold = G.read_gold()
    sp = G.read_eval_split()
    assert list(sp) == list(gold.id)
    assert (gold.id.map(sp) == G.eval_split(gold)).all()


def test_split_is_stratified_and_order_free():
    gold = G.read_gold()
    sp = G.eval_split(gold)
    assert set(sp) == {"dev", "test"}
    seen = gold.id.isin(G.SEEN_BEFORE_SPLIT)
    assert seen.sum() == len(G.SEEN_BEFORE_SPLIT) == 28
    assert (sp[seen] == "dev").all()  # no statement read during prompt design is in test
    strict = gold[gold.status == "strict"].assign(sp=sp, seen=seen)
    for c, g in strict.groupby("msc"):
        # before the seen rows move, each class splits floor(n/2) dev / rest test
        n_dev_hash = int(((g.sp == "dev") & ~g.seen).sum()) + int(g.seen.sum())
        assert n_dev_hash >= len(g) // 2, c
    assert set(strict[strict.sp == "test"].msc) == set(strict.msc) - {"03"}
    assert int((strict.sp == "test").sum()) == 261
    shuffled = gold.sample(frac=1, random_state=3)
    assert (G.eval_split(shuffled).loc[gold.index] == sp).all()
    assert not (G.eval_split(gold, seed=1) == sp).all()


def test_select_split():
    gold = G.read_gold()
    dev, test = G.select_split(gold, "dev"), G.select_split(gold, "test")
    assert set(dev.id).isdisjoint(test.id) and len(dev) + len(test) == len(gold)
    assert G.select_split(gold, "all") is gold
    with pytest.raises(ValueError, match="no dev/test assignment"):
        G.select_split(pd.concat([gold, gold.head(1).assign(id="new:x")]), "test")
    with pytest.raises(ValueError, match="unknown split"):
        G.select_split(gold, "train")


def test_gold_steps_and_accuracy_on_one_half(tmp_path, capsys):
    gold = G.read_gold()
    sp = G.read_eval_split()
    out = tmp_path / "test_steps.parquet"
    assert cli.main(["gold-set", "steps", "--out", str(out), "--strict-only", "--eval-split", "test"]) == 0
    steps = pd.read_parquet(out)
    assert {sp[i] for i in steps.proof_id} == {"test"}
    assert len(steps) == int(((gold.status == "strict") & (gold.id.map(sp) == "test")).sum())
    lab = pd.DataFrame({"proof_id": gold.id, "step": -1, "judge": "j",
                        "msc": [m or "11" for m in gold.msc], "confidence": 0.9})
    rep_all = A.report(lab, gold, n_boot=20)
    rep_test = A.report(lab, gold, n_boot=20, eval_split="test")
    assert rep_test["judges"]["j"]["n_gold"] < rep_all["judges"]["j"]["n_gold"]
    assert rep_test["judges"]["j"]["strict"]["accuracy"] == 1.0
    assert rep_all["settings"]["eval_split"] == "all"
    assert "dev/test split" not in A.markdown(rep_all)


def test_cli_eval_split_writes_the_file(tmp_path, capsys):
    out = tmp_path / "split.csv"
    assert cli.main(["gold-set", "eval-split", "--out", str(out)]) == 0
    df = pd.read_csv(out, dtype=str)
    assert dict(zip(df.id, df.eval_split)) == G.read_eval_split()
