"""The Mac overnight search run: scripts/mac_overnight.sh, its config, and what
search-eval and search-report do for local runs. No model, Lean or network: the prover is
the fixed table from test_search_eval.py, Lean the fake REPL, llama_cpp a stub module."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from markov_lean import search_eval as se
from markov_lean import search_report as sr

from test_search import FAKE_REPL

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mac_overnight.sh"
CONFIG = ROOT / "configs" / "search_eval_mac_overnight.yaml"
TESTS = Path(__file__).resolve().parent

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.fixture
def local(tmp_path):
    """A config shaped like the overnight one, with the table prover and the fake REPL."""
    repl = tmp_path / "fake_repl.py"
    repl.write_text(FAKE_REPL, encoding="utf-8")
    bench = tmp_path / "minif2f.jsonl"
    rows = [{"name": n, "split": "test", "formal_statement": f"theorem {n} : P :=", "header": "import Mathlib\n"}
            for n in ("mathd_algebra_1", "amc12_2", "imo_3", "aime_4")]
    bench.write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = {
        "lean": {"command": [sys.executable, "-I", str(repl)], "timeout": 5, "startup_timeout": 10},
        "prover": {"factory": "test_search_eval:TableProver", "backend": "llama_cpp",
                   "model_path": "~/models/deepseek-prover-v1.5-rl_q4_k_m.gguf"},
        "budget": {"max_expansions": 12, "wall_time": 180},
        "seeds": [0],
        "benchmarks": {"minif2f": {"path": str(bench), "format": "jsonl", "splits": ["test"]}},
        "conditions": {"lm": {"factory": "lm"},
                       "random_l0.3": {"factory": "markov_lean.search_eval:RandomPrior", "kwargs": {"lam": 0.3}}},
        "cost": {"hourly_usd": 0.0, "ebs_usd_per_hour": 0.0},
    }
    cfg_path = tmp_path / "overnight.yaml"
    cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True))
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "llama_cpp.py").write_text("def llama_supports_gpu_offload():\n    return True\n")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"")
    return tmp_path, cfg_path, config, stub, model


def _script(args, env_extra, tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(env_extra)
    env["PATH"] = os.environ.get("PATH", "")
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env,
                          cwd=tmp_path, timeout=300)


def test_overnight_config_is_local_and_sized_for_the_mac():
    cfg = se.EvalConfig.load(CONFIG)
    prover = cfg.search_config["prover"]
    assert prover["backend"] == "llama_cpp" and "base_url" not in yaml.safe_load(CONFIG.read_text())["prover"]
    assert cfg.workers == 1 and cfg.seeds == [0]
    assert cfg.budget["max_expansions"] == 12 and cfg.budget["wall_time"] == 180
    assert {"lm", "area_l0.3", "random_l0.3"} <= set(cfg.conditions)
    assert cfg.conditions["random_l0.3"]["kwargs"]["lam"] == cfg.conditions["area_l0.3"]["kwargs"]["lam"]
    assert cfg.cost["hourly_usd"] == 0 and cfg.cost["ebs_usd_per_hour"] == 0
    assert cfg.benchmarks["minif2f"]["splits"] == ["test"]


def test_overnight_putnam_reads_the_pinned_gold_copy():
    """BENCHMARK=putnam uses the PutnamBench checkout scripts/fetch_gold_sources.sh pins, so the Mac
    needs no second clone and the run uses the same commit as the gold set."""
    spec = yaml.safe_load(CONFIG.read_text())["benchmarks"]["putnam"]
    assert spec["format"] == "putnam" and spec["glob"] == "lean4/src/*.lean"
    fetch = (ROOT / "scripts" / "fetch_gold_sources.sh").read_text()
    assert "RAW=data/raw/gold" in fetch and "clone_at PutnamBench" in fetch
    assert spec["path"] == "data/raw/gold/PutnamBench"
    for name in ("search_eval", "search_eval_mac_overnight"):   # the same budget both ways
        assert yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())["benchmarks"]["putnam"]["format"] == "putnam"


@pytest.mark.skipif((ROOT / "data" / "raw" / "gold" / "PutnamBench").exists(),
                    reason="PutnamBench is fetched here, so the script would start the search")
def test_missing_putnam_points_to_the_gold_fetch(tmp_path, local):
    _, _, _, stub, model = local
    res = _script([], {"PY": sys.executable, "MODEL": str(model), "CONFIG": str(CONFIG), "BENCHMARK": "putnam",
                       "CONDITIONS": "lm", "PYTHONPATH": str(stub)}, tmp_path)
    assert res.returncode == 1 and "scripts/fetch_gold_sources.sh" in res.stderr


def test_dry_run_is_local_resumable_and_credential_free(tmp_path):
    res = _script(["--dry-run"], {"HOURS": "7", "LIMIT": "30"}, tmp_path)
    assert res.returncode == 0, res.stderr
    run = next(l for l in res.stdout.splitlines() if l.startswith("run:"))
    for part in ("markov_lean.mac_guard --hours 7 --hours-flag=--max-hours", "--resume-free-gb 4",
                 "--backend llama_cpp", "--limit 30", "--workers 1", "--yes", "--no-parquet",
                 "--condition lm", "--condition area_l0.3", "--condition random_l0.3",
                 "-u ANTHROPIC_API_KEY", "-u AWS_ACCESS_KEY_ID", "-u AWS_SECRET_ACCESS_KEY"):
        assert part in run
    assert "search_report" in res.stdout and "--no-runner-yield" not in run
    report = next(l for l in res.stdout.splitlines() if l.startswith("report:"))
    assert "--runs results/mac_overnight --out results/mac_overnight/report" in report   # every benchmark
    putnam = _script(["--dry-run"], {"BENCHMARK": "putnam", "LIMIT": "150"}, tmp_path).stdout
    run = next(l for l in putnam.splitlines() if l.startswith("run:"))
    assert "--benchmark putnam" in run and "--limit 150" in run and "--out results/mac_overnight " in run
    assert "--runs results/mac_overnight --out" in putnam   # the miniF2F report is extended, not replaced
    for plan, hours in (("full", "48"), ("night", "12")):
        res = _script(["--dry-run"], {"PLAN": plan}, tmp_path)
        assert f"--hours {hours} " in res.stdout
    assert "--no-runner-yield" in _script(["--dry-run"], {"YIELD_TO_RUNNER": "0"}, tmp_path).stdout
    assert _script(["--dry-run"], {"PLAN": "forever"}, tmp_path).returncode == 2
    plist = _script(["--agent-plist"], {}, tmp_path)
    assert plist.returncode == 0 and "<key>RunAtLoad</key><true/>" in plist.stdout
    assert "SuccessfulExit" in plist.stdout and str(ROOT) in plist.stdout
    text = SCRIPT.read_text()
    commands = [l.split("#")[0] for l in text.splitlines() if not l.lstrip().startswith("#")]
    assert not any(w in " ".join(commands).split() for w in ("aws", "curl", "wget", "ssh"))


def test_checks_stop_before_any_compute(tmp_path, local):
    _, cfg_path, _, stub, model = local
    py = sys.executable
    res = _script([], {"PY": py, "MODEL": str(tmp_path / "missing.gguf"), "CONFIG": str(cfg_path)}, tmp_path)
    assert res.returncode == 1 and "model file" in res.stderr
    res = _script([], {"PY": py, "MODEL": str(model), "CONFIG": str(CONFIG), "MSM_DIR": str(tmp_path / "nope")},
                  tmp_path)
    assert res.returncode == 1 and "featurizer.pkl" in res.stderr
    res = _script([], {"PY": py, "MODEL": str(model), "CONFIG": str(cfg_path), "CONDITIONS": "lm",
                       "PYTHONPATH": str(tmp_path)}, tmp_path)   # no llama_cpp importable
    assert res.returncode == 1 and "Metal" in res.stderr


def test_script_runs_interleaved_resumes_and_reports(tmp_path, local):
    _, cfg_path, _, stub, model = local
    out = tmp_path / "out"
    env = {"PY": sys.executable, "MODEL": str(model), "CONFIG": str(cfg_path), "OUT": str(out),
           "CONDITIONS": "lm random_l0.3", "HOURS": "1", "ANTHROPIC_API_KEY": "should-not-reach-the-run",
           "YIELD_TO_RUNNER": "0", "RESUME_FREE_GB": "0", "MIN_FREE_GB": "0", "MAX_SWAP_GROWTH_GB": "1000",
           "POLL": "0.2", "HOME": str(tmp_path),
           "PYTHONPATH": os.pathsep.join([str(stub), str(TESTS), str(ROOT / "src")])}
    res = _script([], env, tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    for cond in ("lm", "random_l0.3"):
        lines = se.read_summaries(se.run_dir(out, "minif2f", cond, 0) / "summary.jsonl")
        assert len(lines) == 4
        assert {l["backend"] for l in lines} == {"llama_cpp"}
        assert {l["model"] for l in lines} == {"model.gguf"}   # MODEL= wins over the config
    logs = list((out / "logs").glob("*.log"))
    assert len(logs) == 1
    log = logs[0].read_text()
    assert "[1/8]" in log and "[8/8]" in log   # one progress line per problem, as it finishes
    report = (out / "report" / "report.md").read_text()
    assert "Local runs" in report and "not comparable" in report and "Detectable effect" in report
    assert (out / "report" / "pass_at_k.tex").read_text().startswith("% Local runs")
    assert (out / "report" / "LOCAL_MODEL.txt").exists()
    run = json.loads((out / "runs.jsonl").read_text().splitlines()[0])
    assert run["prover"]["backend"] == "llama_cpp"

    assert (se.run_dir(out, "minif2f", "lm", 0) / "trace.parquet").exists()   # rebuilt once, at the end
    assert json.loads((out / "guard_state.json").read_text())["status"] == "done"
    status = _script(["--status"], env, tmp_path)
    assert status.returncode == 0 and status.stdout.startswith("done") and "4 problems done" in status.stdout

    again = _script([], env, tmp_path)   # the next night: nothing left, nothing rerun
    assert again.returncode == 0 and "Nothing left to run." in again.stdout
    assert len(se.read_summaries(se.run_dir(out, "minif2f", "lm", 0) / "summary.jsonl")) == 4


def test_single_worker_logs_progress_as_problems_finish(local, caplog, monkeypatch):
    tmp, cfg_path, config, _, _ = local
    monkeypatch.syspath_prepend(str(TESTS))
    cfg = se.EvalConfig.load(cfg_path)
    problems = {"minif2f": se.order_problems(se.load_benchmark("minif2f", config["benchmarks"]["minif2f"]))}
    tasks, _ = se.plan_tasks(problems, ["lm", "random_l0.3"], [0], tmp / "out")
    seen = []
    real = se.append_result

    def spy(directory, summary, trace):
        seen.append(sum("problem runs" not in r.message and r.message.startswith("[") for r in caplog.records))
        real(directory, summary, trace)

    monkeypatch.setattr(se, "append_result", spy)
    with caplog.at_level(logging.INFO, logger="markov_lean.search_eval"):
        stats = se.run_tasks(cfg, problems, tasks, tmp / "out", workers=1)
    assert stats["done"] == 8
    assert seen == list(range(8))   # problem i's line was out before problem i + 1 was saved


def test_a_prior_that_does_not_load_stops_the_run_before_any_search(local, tmp_path):
    _, cfg_path, config, _, _ = local
    config["conditions"]["area_l0.3"] = {"factory": se.AREA_FACTORY, "kwargs": {"lam": 0.3}}
    config["msm_dir"] = str(tmp_path / "no_model")
    cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True))
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="area_l0.3"):
        se.main(["--config", str(cfg_path), "--out", str(out), "--yes"])
    assert not (out / "minif2f").exists()


# ---------------------------------------------------------------------- report pieces


def test_mcnemar_power_and_detectable_gain():
    assert 0.0 < sr.mcnemar_power(244, 0.10, 0.0) <= 0.05   # the exact test keeps its size
    assert sr.mcnemar_power(244, 0.10, 0.03) < sr.mcnemar_power(244, 0.10, 0.06) < sr.mcnemar_power(244, 0.10, 0.09)
    assert sr.mcnemar_power(244, 0.10, 0.06) == pytest.approx(0.8, abs=0.05)
    # the numbers in docs/search_eval.md
    assert sr.mcnemar_mde(244, 0.10) == pytest.approx(0.06, abs=0.003)
    assert sr.mcnemar_mde(110, 0.10) == pytest.approx(0.09, abs=0.006)
    assert sr.mcnemar_mde(100, 0.06) != sr.mcnemar_mde(100, 0.06)   # NaN: too few disagree
    paired = pd.DataFrame([{"benchmark": "minif2f", "condition": "area_l0.3", "n_common": 200,
                            "only_cond": 14, "only_base": 6}])
    det = sr.detectable_effects(paired)
    assert det.loc[0, "discordance"] == pytest.approx(0.10) and 0.05 < det.loc[0, "mde"] < 0.08


def test_local_note_only_for_llama_cpp_runs(tmp_path):
    def write(root, backend):
        d = root / "minif2f" / "lm" / "seed0"
        d.mkdir(parents=True)
        line = {"proof_id": "p0", "solved": True, "stop_reason": "solved", "max_expansions": 12, "wall_time": 180}
        if backend:
            line["backend"] = backend
        (d / "summary.jsonl").write_text(json.dumps(line) + "\n")
        return root

    vllm = write(tmp_path / "ec2", "vllm")
    assert sr.local_model_note(sr.load_summaries(vllm), vllm) is None
    mac = write(tmp_path / "mac", "llama_cpp")
    note = sr.local_model_note(sr.load_summaries(mac), mac)
    assert "llama.cpp" in note and "12 expansions, 180 s" in note
    old = write(tmp_path / "old", None)   # lines from before the backend field: runs.jsonl says it
    (old / "runs.jsonl").write_text(json.dumps({"prover": {"backend": "llama_cpp",
                                                           "model_path": "x_q4_k_m.gguf"}}) + "\n")
    note = sr.local_model_note(sr.load_summaries(old), old / "minif2f")
    assert note and "x_q4_k_m.gguf" in note


def test_local_judge_wrapper_reports_status(tmp_path):
    env = {k: v for k, v in os.environ.items()}
    env.update({"PY": sys.executable, "STATE": str(tmp_path / "judge_guard.json")})
    res = subprocess.run(["bash", str(ROOT / "scripts" / "mac_local_judge.sh"), "--status"], capture_output=True,
                         text=True, env=env, timeout=60)
    assert res.returncode == 0 and "no guard state" in res.stdout
    text = (ROOT / "scripts" / "mac_local_judge.sh").read_text()
    assert "markov_lean.mac_guard" in text and "ollama stop" in text and "-u ANTHROPIC_API_KEY" in text
