"""Tests for the search evaluation runner (src/markov_lean/search_eval.py).

No GPU, Lean or network: Lean is the fake REPL from test_search.py and the prover is
:class:`TableProver` below, plugged in through ``prover.factory``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from markov_lean import cli, search_eval as se
from markov_lean.area_prior import make_prior
from markov_lean.search.best_first import STEPS_COLUMNS, TRACE_COLUMNS
from markov_lean.search.prover import Candidate

from test_search import FAKE_REPL

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "data"

TABLE = {
    "⊢ start": [("b", -0.1), ("a", -1.0), ("c", -2.0)],
    "⊢ mid": [("b2", -0.1), ("close", -0.5)],
    "⊢ dead": [("nothing", -0.1)],
}


class TableProver:
    """Deterministic proposals keyed by goal state; ``fail`` makes every call raise."""

    def __init__(self, seed: int = 0, fail: bool = False, **_: object) -> None:
        self.seed, self.fail = seed, fail

    def propose(self, statement, tactics, state_text, informal_prefix="", header=None, n=None):
        if self.fail:
            raise ConnectionError("prover down")
        return [Candidate(t, lp, n_tokens=2) for t, lp in TABLE.get(state_text, [])]


@pytest.fixture
def setup(tmp_path):
    script = tmp_path / "fake_repl.py"
    script.write_text(FAKE_REPL, encoding="utf-8")
    bench = tmp_path / "bench.jsonl"
    rows = [{"name": n, "split": s, "formal_statement": f"theorem {n} : P :=", "header": "import Mathlib\n"}
            for n, s in [("mathd_algebra_1", "test"), ("amc12_2", "test"), ("imo_3", "test"),
                         ("aime_4", "test"), ("val_5", "valid"), ("bad_statement_6", "test")]]
    bench.write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = {
        "lean": {"command": [sys.executable, "-I", str(script)], "timeout": 5, "startup_timeout": 10},
        "prover": {"factory": "test_search_eval:TableProver"},
        "budget": {"max_expansions": 10, "wall_time": 60},
        "seeds": [0, 1],
        "benchmarks": {"minif2f": {"path": str(bench), "format": "jsonl", "splits": ["test"]}},
        "conditions": {"lm": {"factory": "lm"},
                       "random_l0.5": {"factory": "markov_lean.search_eval:RandomPrior", "kwargs": {"lam": 0.5}}},
    }
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True))
    return tmp_path, cfg_path, config


def _summaries(out: Path) -> pd.DataFrame:
    rows = [r for p in sorted(out.glob("*/*/seed*/summary.jsonl")) for r in se.read_summaries(p)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- end to end


def test_run_writes_resumable_logs(setup):
    tmp, cfg_path, _ = setup
    out = tmp / "out"
    assert cli.main(["search-eval", "--config", str(cfg_path), "--out", str(out), "--yes", "--workers", "1"]) == 0
    df = _summaries(out)
    # 5 test problems x 2 conditions x 2 seeds; the valid-split problem is not loaded
    assert len(df) == 20 and set(df["condition"]) == {"lm", "random_l0.5"} and set(df["seed"]) == {0, 1}
    for col in ("benchmark", "max_expansions", "wall_time", "root_area", "name_area", "order", "proof_id"):
        assert col in df.columns
    assert (df["max_expansions"] == 10).all() and (df["wall_time"] == 60).all()
    ok = df[~df["stop_reason"].str.startswith("statement error")]
    assert ok["solved"].all() and len(ok) == 16
    assert set(df["name_area"].dropna()) == {"algebra", "amc", "imo", "aime"}
    d = se.run_dir(out, "minif2f", "lm", 0)
    assert list(pd.read_parquet(d / "trace.parquet").columns) == TRACE_COLUMNS
    assert list(pd.read_parquet(d / "steps.parquet").columns) == STEPS_COLUMNS
    assert json.loads((out / "runs.jsonl").read_text().splitlines()[0])["benchmarks"] == {"minif2f": 5}

    # rerunning finds nothing to do and writes nothing
    assert cli.main(["search-eval", "--config", str(cfg_path), "--out", str(out), "--yes"]) == 0
    assert len(_summaries(out)) == 20

    # and the report reads what the runner wrote
    assert cli.main(["search-report", "--runs", str(out), "--n-boot", "50", "--no-figures"]) == 0
    table = pd.read_csv(out / "report" / "pass_at_k.csv")
    assert set(table["condition"]) == {"lm", "random_l0.5"} and (table["n_problems"] == 5).all()
    assert table["pass@1"].tolist() == pytest.approx([0.8, 0.8])   # 4 of 5; one statement error
    assert (out / "report" / "paired.csv").exists() and (out / "report" / "by_area.csv").exists()


def test_limit_takes_a_prefix_of_the_fixed_order_and_extends_it(setup):
    tmp, cfg_path, _ = setup
    out = tmp / "out"
    args = ["search-eval", "--config", str(cfg_path), "--out", str(out), "--yes", "--condition", "lm", "--seed", "0"]
    cli.main(args + ["--limit", "2"])
    first = set(_summaries(out)["theorem"])
    cli.main(args + ["--limit", "4"])
    df = _summaries(out)
    assert len(first) == 2 and first < set(df["theorem"]) and len(df) == 4
    assert sorted(df["order"]) == [0, 1, 2, 3]


def test_parallel_workers_share_the_queue(setup):
    tmp, cfg_path, config = setup
    cfg = se.EvalConfig.load(cfg_path)
    problems = {"minif2f": se.order_problems(se.load_benchmark("minif2f", config["benchmarks"]["minif2f"]))}
    tasks, done = se.plan_tasks(problems, ["lm", "random_l0.5"], [0, 1], tmp / "out")
    assert done == 0 and len(tasks) == 20
    stats = se.run_tasks(cfg, problems, tasks, tmp / "out", workers=2)
    assert stats["done"] == 20 and stats["errors"] == 0
    df = _summaries(tmp / "out")
    assert len(df) == 20 and not df.duplicated(["condition", "seed", "proof_id"]).any()
    assert set(df["worker"]) <= {0, 1}
    for p in (tmp / "out").glob("*/*/seed*/trace.jsonl"):   # every line whole
        for line in p.read_text().splitlines():
            json.loads(line)


def test_prover_errors_stop_the_worker_and_leave_problems_to_retry(setup):
    tmp, cfg_path, config = setup
    config["prover"]["fail"] = True
    cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True))
    out = tmp / "out"
    code = cli.main(["search-eval", "--config", str(cfg_path), "--out", str(out), "--yes", "--name", "imo_3",
                     "--name", "aime_4"])
    assert code == 1
    assert _summaries(out).empty   # nothing recorded, so a rerun retries them


def test_time_limit_stops_before_new_problems(setup):
    tmp, cfg_path, _ = setup
    out = tmp / "out"
    assert cli.main(["search-eval", "--config", str(cfg_path), "--out", str(out), "--yes", "--max-hours", "0"]) == 0
    assert _summaries(out).empty


def test_estimate_only_and_no_tty_needs_yes(setup, capsys, monkeypatch):
    tmp, cfg_path, _ = setup
    out = tmp / "out"
    assert se.main(["--config", str(cfg_path), "--out", str(out), "--estimate"]) == 0
    text = capsys.readouterr().out
    assert "expected" in text and "worst" in text and "20 problem runs left" in text
    monkeypatch.setattr(sys, "stdin", open("/dev/null"))
    assert se.main(["--config", str(cfg_path), "--out", str(out)]) == 2
    assert not (out / "minif2f").exists()


def test_unknown_condition_is_rejected(setup):
    _, cfg_path, _ = setup
    with pytest.raises(SystemExit):
        se.main(["--config", str(cfg_path), "--condition", "nope", "--estimate"])


# ---------------------------------------------------------------------- pieces


def test_plan_is_problem_major_and_skips_done(tmp_path):
    from markov_lean.search.best_first import Theorem

    thms = [Theorem(name=f"t{i}", statement="theorem t : P :=") for i in range(3)]
    problems = {"minif2f": thms, "putnam": thms[:1]}
    tasks, _ = se.plan_tasks(problems, ["lm", "x"], [0], tmp_path)
    assert [(t.index, t.benchmark, t.condition) for t in tasks[:5]] == [
        (0, "minif2f", "lm"), (0, "minif2f", "x"), (0, "putnam", "lm"), (0, "putnam", "x"), (1, "minif2f", "lm")]
    se.append_result(se.run_dir(tmp_path, "minif2f", "lm", 0), {"proof_id": "minif2f:t0"}, [])
    tasks, done = se.plan_tasks(problems, ["lm", "x"], [0], tmp_path)
    assert done == 1 and len(tasks) == 7


def test_order_is_stable_and_seeded():
    from markov_lean.search.best_first import Theorem

    thms = [Theorem(name=f"t{i}", statement="s :=") for i in range(50)]
    a = [t.name for t in se.order_problems(thms, 0)]
    assert a == [t.name for t in se.order_problems(list(reversed(thms)), 0)]
    assert a != [t.name for t in se.order_problems(thms, 1)] and a != sorted(a)


def test_random_prior_is_reproducible_and_mixes_like_the_area_prior():
    cands = [{"tactic": "a", "logprob": -0.1}, {"tactic": "b", "logprob": -1.0}, {"tactic": "c", "logprob": -3.0}]
    p = se.RandomPrior(lam=0.5, seed=3)
    w = p("⊢ x", cands)
    assert sum(w) == pytest.approx(1.0) and w == p("⊢ x", list(cands))
    assert w != se.RandomPrior(lam=0.5, seed=4)("⊢ x", cands)
    lm = np.exp([-0.1, -1.0, -3.0])
    assert se.RandomPrior(lam=0.0)("⊢ x", cands) == pytest.approx(list(lm / lm.sum()))
    assert p("⊢ x", []) == []
    with pytest.raises(ValueError):
        se.RandomPrior(lam=1.5)


def test_prior_spec_fills_msm_dir_and_seed():
    cfg = se.EvalConfig.from_dict({
        "conditions": {"a": {"factory": se.AREA_FACTORY, "kwargs": {"lam": 0.3}},
                       "r": {"factory": "markov_lean.search_eval:RandomPrior", "kwargs": {"lam": 0.3}}},
        "msm_dir": "results/x/msm"})
    assert cfg.prior_spec("a", 0)["kwargs"] == {"lam": 0.3, "msm_dir": "results/x/msm"}
    assert cfg.prior_spec("r", 2)["kwargs"]["seed"] == 2
    cfg.msm_dir = None
    with pytest.raises(ValueError, match="msm_dir"):
        cfg.prior_spec("a", 0)
    with pytest.raises(ValueError):
        se.EvalConfig.from_dict({"conditions": {"bad/name": {"factory": "lm"}}})


def test_shipped_config_loads_and_area_kwargs_fit_the_prior():
    cfg = se.EvalConfig.load(ROOT / "configs" / "search_eval.yaml")
    assert cfg.budget == {"max_expansions": 64, "wall_time": 300.0, "max_depth": 32}
    assert cfg.search_config["prover"]["max_tokens"] == 128
    assert cfg.search_config["prover"]["n"] == 16          # inherited from search.yaml
    assert {"lm", "random_l0.3", "diversity", "stuck_escape", "full"} <= set(cfg.conditions)
    rng = np.random.default_rng(0)
    centers, memb = rng.normal(size=(4, 3)), rng.dirichlet(np.ones(2), size=4)
    for name in cfg.conditions:
        spec = cfg.prior_spec(name, 0)
        if spec["factory"] != se.AREA_FACTORY:
            continue
        kw = {k: v for k, v in spec["kwargs"].items() if k not in ("msm_dir", "use_coarse_T")}
        prior = make_prior(lambda s: rng.normal(size=3), centers, memb, coarse_T=np.eye(2), **kw)
        w = prior("⊢ x", [{"tactic": "t", "state_after": "⊢ y", "logprob": -1.0}])
        assert sum(w) == pytest.approx(1.0)


def test_name_area():
    assert se.name_area("mathd_numbertheory_12") == "number_theory"
    assert se.name_area("amc12a_2002_p6") == "amc"
    assert se.name_area("putnam_1962_a1") is None


PUTNAM = """import Mathlib

open Real

-- note
abbrev putnam_2000_a1_solution : Set ℝ := sorry
-- Set.Ioo 0 (A ^ 2)
/--
Let $A$ be a positive real number. What are the possible values?
-/
theorem putnam_2000_a1
(A : ℝ) (hA : A > 0) :
{S | S > 0} = putnam_2000_a1_solution :=
sorry
"""


def test_parse_putnam_fills_the_solution_and_keeps_the_problem_text():
    p = se.parse_putnam_file(PUTNAM, "putnam_2000_a1")
    assert p["name"] == "putnam_2000_a1"
    assert p["statement"].startswith("theorem putnam_2000_a1\n(A : ℝ)") and p["statement"].endswith("putnam_2000_a1_solution :=")
    assert "abbrev putnam_2000_a1_solution : Set ℝ := Set.Ioo 0 (A ^ 2)" in p["preamble"]
    assert "open Real" in p["preamble"] and "import" not in p["preamble"] and "/--" not in p["preamble"]
    assert p["informal"].startswith("/--") and "possible values" in p["informal"]
    with pytest.raises(ValueError, match="sorry"):
        se.parse_putnam_file(PUTNAM.replace("-- Set.Ioo 0 (A ^ 2)\n", ""), "putnam_2000_a1")
    assert se.parse_putnam_file("import Mathlib\n", "x") is None


def test_load_putnam_fixture_and_minif2f_fixture():
    thms = se.load_benchmark("putnam", {"path": str(FIXTURES / "PutnamBench"), "format": "putnam"})
    assert [t.name for t in thms] == ["putnam_1962_a1"]
    t = thms[0]
    assert t.proof_id == "putnam:putnam_1962_a1" and t.lean_preamble == "open MeasureTheory"
    assert t.lean_statement.startswith("open MeasureTheory\n\ntheorem putnam_1962_a1")
    assert t.header.startswith("import Mathlib\n\nopen MeasureTheory") and t.informal_prefix.startswith("/--")
    mini = se.load_benchmark("minif2f", {"path": str(FIXTURES / "deepseek_benchmarks/minif2f.jsonl"),
                                         "format": "jsonl", "splits": ["val"]})
    assert [x.name for x in mini] == ["amc12a_2002_p6"] and mini[0].split == "val"
    with pytest.raises(FileNotFoundError):
        se.load_benchmark("x", {"path": "/nonexistent.jsonl"})


def test_estimate_uses_guess_then_observations():
    budget = {"max_expansions": 64, "wall_time": 300.0}
    cost = {"hourly_usd": 1.0, "ebs_usd_per_hour": 0.0, "startup_hours": 0.0, "sec_per_expansion": 2.0,
            "expected_fraction": {"minif2f": 0.5}}
    est = se.estimate({("minif2f", "lm"): (100, 0)}, budget, workers=1, cost=cost)
    assert est.rows[0]["basis"] == "guess"
    assert est.hours("expected_s") == pytest.approx(100 * 128 * 0.5 / 3600)
    assert est.hours("worst_s") == pytest.approx(100 * 300 / 3600)
    obs = {("minif2f", "lm"): [10.0] * 5}
    est = se.estimate({("minif2f", "lm"): (100, 5)}, budget, workers=4, cost=cost, observed=obs)
    assert est.rows[0]["basis"] == "observed5"
    assert est.usd("expected_s") == pytest.approx(100 * 10 / (4 * 0.8) / 3600)
    assert "expected" in est.table()


def test_append_ignores_a_cut_off_line(tmp_path):
    d = tmp_path / "run"
    se.append_result(d, {"proof_id": "a"}, [{"x": 1}])
    with open(d / "summary.jsonl", "a") as f:
        f.write('{"proof_id": "b"')   # crash mid-write
    assert se.done_ids(d / "summary.jsonl") == {"a"}


def test_gpu_and_mac_b64_configs_match_the_mac_run():
    """The AWS run and the bigger Mac run differ from each other only in the weights and the sampler,
    and from the first Mac run only in the budget, so their budget curves line up."""
    load = lambda n: yaml.safe_load((ROOT / "configs" / f"{n}.yaml").read_text())   # noqa: E731
    mac, b64, gpu = load("search_eval_mac_overnight"), load("search_eval_mac_b64"), load("search_eval_gpu")
    assert b64["budget"]["max_expansions"] == gpu["budget"]["max_expansions"] == 64
    assert b64["budget"]["max_depth"] == gpu["budget"]["max_depth"]
    assert mac["budget"]["max_expansions"] == 12
    for name in ("lm", "area_l0.3", "random_l0.3"):
        assert mac["conditions"][name] == b64["conditions"][name] == gpu["conditions"][name]
    for key in ("seeds", "subset_seed", "msm_dir", "benchmarks"):
        assert mac[key] == b64[key] == gpu[key], key
    assert {k: v for k, v in b64["prover"].items()} == mac["prover"]
    gcfg = se.EvalConfig.load(ROOT / "configs" / "search_eval_gpu.yaml")
    assert gcfg.search_config["prover"]["n"] == mac["prover"]["n"] == 8
    assert gcfg.search_config["prover"]["max_tokens"] == mac["prover"]["max_tokens"]
    assert gcfg.search_config["prover"].get("backend", "vllm") != "llama_cpp"
    assert gcfg.budget["max_expansions"] == 64
    # the new results folders sort after mac_overnight, so the paper keeps reading it
    assert sorted(["mac_overnight", "mac_overnight_b64", "search_gpu"])[0] == "mac_overnight"
