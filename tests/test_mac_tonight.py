"""scripts/mac_tonight.sh, the one command for tonight on the Mac. Every heavy tool is a
fake on PATH (git, lake, curl, ollama, pgrep) or passed in (PY and the four scripts it
calls), each logging one line per call, so this checks the order, the skipping on rerun,
the stop on failure, the y/N before the only paid step, and that the key never shows."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mac_tonight.sh"
SECRET = "sk-ant-test-0123456789abcdef"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("tee") is None,
                                reason="needs bash")

LOG = 'echo "$(basename "$0") $* key=$([ -n "${ANTHROPIC_API_KEY+x}" ] && echo set || echo unset)" >> "$CALLS"\n'
FAIL = 'case "$(basename "$0") $*" in *"${FAKE_FAIL:-@@none@@}"*) exit 1 ;; esac\n'

FAKES = {
    "bin/git": LOG + FAIL + """
case "$1" in
  ls-remote) case "$*" in
      *mathlib4*) printf 'a\\trefs/tags/v4.20.0\\nb\\trefs/tags/v4.21.0\\nc\\trefs/tags/v4.22.0\\nd\\trefs/tags/v4.22.0-rc1\\n' ;;
      *) printf 'a\\trefs/tags/v4.20.0\\nb\\trefs/tags/v4.21.0\\n' ;; esac ;;
  clone) for a; do last=$a; done; mkdir -p "$last/.git" ;;
esac
""",
    "bin/lake": LOG + FAIL + """
case "$1" in
  build) mkdir -p .lake/build/bin ;;
  env) cat >/dev/null; echo "${FAKE_REPL_REPLY:-{\\"env\\": 0\\}}" ;;
esac
""",
    "bin/curl": LOG + FAIL + """
while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done
printf '01234567890123456789' > "$out"
""",
    "bin/ollama": LOG + '[ "$1" = list ] && echo "qwen2.5:7b-instruct  abc  4.7 GB"\nexit 0\n',
    "bin/pgrep": 'exit "${FAKE_LAKE_RUNNING:-1}"\n',
    "py": LOG + FAIL + """
[ "$1" = -c ] && exit 0
args="$*"
case "$args" in
  "-m markov_lean.mac_guard"*) while [ "$1" != -- ]; do shift; done; shift; exec "$@" ;;
  *"markov_lean.search --check"*) echo "sample tactics (4): omega"; echo "lean ok: ${FAKE_LEAN_OK:-True}" ;;
  *"fetch minif2f"*) mkdir -p data/raw/deepseek_benchmarks; touch data/raw/deepseek_benchmarks/minif2f.jsonl ;;
  *"gold-set steps"*|*"judge submit"*)
    while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done
    case "$args" in
      *"judge submit"*) mkdir -p "$out"; echo '{}' > "$out/batches.json" ;;
      *) mkdir -p "$(dirname "$out")"; touch "$out" ;;
    esac ;;
  *"judge estimate"*) echo "estimate: \\$3.85" ;;
esac
exit 0
""",
    "setup_lean.sh": LOG + FAIL + """
mkdir -p lean
printf 'rev = "%s"\\n' "$MATHLIB_TAG" > lean/lakefile.toml
echo "leanprover/lean4:$MATHLIB_TAG" > lean/lean-toolchain
echo '{}' > lean/lake-manifest.json
echo "MATHLIB_TAG=$MATHLIB_TAG" >> "$CALLS"
""",
    "after_judges.sh": LOG + FAIL,
    "mac_local_judge.sh": LOG + FAIL,
    "mac_overnight.sh": LOG + FAIL + 'echo "overnight MODEL=$MODEL"\n',
}


@pytest.fixture
def mac(tmp_path):
    work = tmp_path / "work"
    (work / "lean").mkdir(parents=True)
    (work / "lean" / "lakefile.toml").write_text('name = "MarkovLean"\nrev = "PINNED_BY_SETUP"\n')
    fakes = tmp_path / "fakes"
    for rel, body in FAKES.items():
        p = fakes / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#!/usr/bin/env bash\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    home = tmp_path / "home"
    home.mkdir()
    calls = tmp_path / "calls.txt"
    env = {
        "PATH": f"{fakes / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(home), "CALLS": str(calls), "WORKDIR": str(work), "PY": str(fakes / "py"),
        "SETUP_LEAN": str(fakes / "setup_lean.sh"), "AFTER_JUDGES": str(fakes / "after_judges.sh"),
        "LOCAL_JUDGE": str(fakes / "mac_local_judge.sh"), "OVERNIGHT": str(fakes / "mac_overnight.sh"),
        "MODEL_MIN_BYTES": "10", "ANTHROPIC_API_KEY": SECRET,
    }

    def run(*args, stdin="", **extra):
        e = dict(env)
        e.update(extra)
        e = {k: v for k, v in e.items() if v is not None}
        res = subprocess.run(["bash", str(SCRIPT), *args], input=stdin, capture_output=True, text=True,
                             env=e, cwd=work, timeout=120)
        out = calls.read_text().splitlines() if calls.exists() else []
        calls.write_text("")
        return res, out

    return work, home, run


def _index(calls, needle):
    return next(i for i, c in enumerate(calls) if needle in c)


def test_one_command_runs_every_stage_in_order_then_resumes(mac):
    work, home, run = mac
    res, calls = run()
    assert res.returncode == 0, res.stdout + res.stderr
    text = res.stdout + res.stderr
    for n in range(1, 9):
        assert f"stage {n} of 8" in text
    assert "MATHLIB_TAG=v4.21.0" in calls   # newest v4.N.0 that both Mathlib and the REPL have
    assert (work / "results/mac/mathlib_tag.txt").read_text().strip() == "v4.21.0"
    order = ["setup_lean.sh", "checkout -q v4.21.0", "lake build", "lake env", "curl -fL --retry 4 -C -",
             "fetch minif2f", "markov_lean.search --check", "search-eval", "search-report", "after_judges.sh",
             "gold-set steps --strict-only --eval-split test", "mac_local_judge.sh --steps", "--prompt v2",
             "ollama stop qwen2.5:7b-instruct", "judge-accuracy --eval-split test", "gold-set steps --out",
             "judge estimate", "mac_overnight.sh"]
    idx = [_index(calls, o) for o in order]
    assert idx == sorted(idx), calls
    assert calls[-1].startswith("mac_overnight.sh")
    # not at a terminal: the paid step is never taken
    assert not any("judge submit" in c for c in calls)
    assert "not at a terminal" in text
    # credentials: the search and the judges run without the key; after_judges gets the shell's
    smoke = [c for c in calls if "search-eval" in c and "--yes" in c]
    assert smoke and all(c.endswith("key=unset") for c in smoke)
    assert any("mac_guard" in c and "search-eval" in c for c in calls)
    assert next(c for c in calls if c.startswith("after_judges.sh")).endswith("key=set")
    assert all(c.endswith("key=unset") for c in calls if "markov_lean.search --check" in c or "judge estimate" in c)
    assert "--prompt v2 --out results/gold/local/qwen2.5_7b-instruct_v2" in next(c for c in calls if "--prompt v2" in c)
    log = next((work / "results/mac/tonight/logs").glob("*.log")).read_text()
    assert SECRET not in text and SECRET not in log and "stage 8 of 8" in log
    assert f"overnight MODEL={home}/models/deepseek-prover-v1.5-rl_q4_k_m.gguf" in text

    res, calls = run()   # rerun: every finished stage is skipped, the search resumes
    assert res.returncode == 0, res.stdout + res.stderr
    assert [c.split()[0] for c in calls if not c.startswith("py -c")] == ["mac_overnight.sh"]
    assert res.stdout.count("already done") == 7

    res, _ = run("--status")
    assert res.returncode == 0
    assert "v4.21.0" in res.stdout and "declined" in res.stdout and "started" in res.stdout
    assert "overnight" in res.stdout   # the search's own --status


def test_reuses_the_tag_already_pinned_in_lean(mac):
    work, _, run = mac
    (work / "lean/lakefile.toml").write_text('rev = "v4.19.0"\n')
    (work / "lean/lean-toolchain").write_text("leanprover/lean4:v4.19.0\n")
    (work / "lean/lake-manifest.json").write_text("{}")
    res, calls = run("--only", "1")
    assert res.returncode == 0, res.stdout + res.stderr
    assert not any("ls-remote" in c or c.startswith("setup_lean.sh") for c in calls)
    assert [c.split(" key=")[0] for c in calls if not c.startswith("py -c")] == ["lake exe cache get", "lake build"]
    assert (work / "results/mac/mathlib_tag.txt").read_text().strip() == "v4.19.0"
    res, calls = run("--only", "2")
    assert res.returncode == 0 and any("checkout -q v4.19.0" in c for c in calls)


def test_waits_for_a_lean_setup_that_is_still_running(mac):
    _, _, run = mac
    res, calls = run(FAKE_LAKE_RUNNING="0")
    assert res.returncode == 1 and "still running" in (res.stdout + res.stderr)
    assert not any(c.startswith(("setup_lean.sh", "lake", "mac_overnight.sh")) for c in calls)


def test_a_failure_stops_with_a_plain_message_and_from_skips_ahead(mac):
    _, _, run = mac
    res, calls = run(FAKE_FAIL="search-report")
    assert res.returncode == 1
    assert "STOPPED at stage 4" in (res.stdout + res.stderr) and "--from 5" in (res.stdout + res.stderr)
    assert not any(c.startswith(("after_judges.sh", "mac_overnight.sh")) for c in calls)
    res, calls = run("--from", "5")
    assert res.returncode == 1   # 5 to 7 ran, but the search still needs stage 4
    assert any(c.startswith("after_judges.sh") for c in calls) and any("judge estimate" in c for c in calls)
    assert "stage 4 (smoke search (10 problems)) has not finished" in (res.stdout + res.stderr)
    assert not any(c.startswith("mac_overnight.sh") for c in calls)


def test_repl_check_with_an_error_stops(mac):
    _, _, run = mac
    run("--only", "1")
    res, _ = run("--only", "2", FAKE_REPL_REPLY='{"messages": [{"severity": "error"}], "env": 0}')
    assert res.returncode == 1 and "REPL check" in (res.stdout + res.stderr)
    res, _ = run("--only", "4", FAKE_LEAN_OK="False")
    assert res.returncode == 1 and "lean ok: True" in (res.stdout + res.stderr)


def test_the_api_submit_needs_a_typed_yes_and_never_repeats(mac):
    work, _, run = mac
    res, calls = run("--only", "7", stdin="\n", MAC_TONIGHT_TTY="1")
    assert res.returncode == 0 and not any("judge submit" in c for c in calls)
    assert "[y/N]" in res.stdout and "estimate: $3.85" in res.stdout
    res, calls = run("--only", "7", stdin=None, MAC_TONIGHT_TTY="1", ASK_TIMEOUT="1")   # nobody answers
    assert res.returncode == 0 and not any("judge submit" in c for c in calls) and "not submitted" in res.stdout
    res, calls = run("--only", "7", stdin="y\n", MAC_TONIGHT_TTY="1", ANTHROPIC_API_KEY=None)
    assert not any("judge submit" in c for c in calls) and "not set" in res.stdout
    res, calls = run("--only", "7", stdin="y\n", MAC_TONIGHT_TTY="1")
    assert res.returncode == 0
    sub = [c for c in calls if "judge submit" in c]
    assert len(sub) == 1 and "--steps results/gold/steps_all.parquet --out results/gold/api --max-dollars 10" in sub[0]
    assert sub[0].endswith("key=set") and SECRET not in res.stdout + res.stderr
    assert (work / "results/gold/api/batches.json").exists()
    res, calls = run("--only", "7", stdin="y\n", MAC_TONIGHT_TTY="1")
    assert "already submitted" in res.stdout and not any("judge submit" in c for c in calls)


def test_script_spends_and_launches_nothing_by_itself():
    text = SCRIPT.read_text()
    code = " ".join(l.split("#")[0] for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert "aws" not in code.split() and "ssh" not in code.split() and "ec2 " not in code
    assert "$ANTHROPIC_API_KEY" not in code and "${ANTHROPIC_API_KEY}" not in code   # only ever tested for being set
    assert code.count("markov_lean.judge submit") == 1 and "--max-dollars" in code
    for tool in ("pyproject", "pip install", "launchctl", "defaults write", "sudo"):
        assert tool not in code
