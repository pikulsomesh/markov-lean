# Search evaluation runbook (EC2 g6.xlarge)

`markov-lean search-eval` runs best-first proof search on miniF2F (test, 244 problems) and
PutnamBench under named prior conditions with fixed budgets, and `markov-lean
search-report` turns the logs into pass@k tables, paired tests, per-area tables and figures.
Everything here runs on one g6.xlarge (NVIDIA L4 24 GB, 4 vCPU, 16 GB RAM) that you launch
and stop yourself. Nothing in this repo calls the AWS API.

## What one run is

* A **condition** is a prior from `configs/search_eval.yaml`:

  | condition | what it tests |
  |---|---|
  | `lm` | no prior: the prover's own ranking (baseline) |
  | `area_l0.1`, `area_l0.3`, `area_l0.6` | area prior steering by the coarse transition matrix, at three mixing weights |
  | `diversity` | diversity bonus only (favour areas visited less) |
  | `stuck_escape` | stuck escape only (push away from areas with no progress) |
  | `full` | transition target, diversity and stuck escape together |
  | `random_l0.3` | control: random scores mixed in with the same rule and weight |

* **Budgets** are the same for every problem and condition: 64 expansions (prover calls),
  300 s wall time, depth 32 (`budget` in the config). Each summary line records them.
* **Seeds** go to vLLM as the sampling seed, so at a given goal state every condition sees
  the same tactic samples. One seed gives pass@1; more seeds give pass@k over seeds.
* **Order**: problems run in a fixed random order (`subset_seed`), all conditions of a
  problem before the next one. `--limit N` takes the first N of that order; raising N later
  extends the same subset, and stopping early still leaves a paired sample.
* **Resume**: rerun the same command. Finished problems are skipped; a problem whose search
  crashed (prover down, REPL dead) is retried. A worker stops after 3 errors in a row.

Logs go to `<out>/<benchmark>/<condition>/seed<k>/` as `summary.jsonl` and `trace.jsonl`
(schema in `docs/schema.md`), plus `trace.parquet` and `steps.parquet` at the end of a run.
`<out>/runs.jsonl` records each invocation (commit, budgets, conditions, prover settings).

## Cost plans

`--estimate` prints the plan before anything runs. The numbers below are its first guesses
for 3 workers at $0.83/h (g6.xlarge on-demand in us-east-1 plus a 200 GB disk); the guess
assumes about 6 s per expansion. After the smoke run the estimate switches to measured times
per problem, so rerun `--estimate` before each step.

| step | command flags | expected | worst case |
|---|---|---|---|
| 1 smoke | `--benchmark minif2f --limit 5 --condition lm --condition area_l0.3` | 0.6 h, $0.50 | $0.60 |
| 2 core miniF2F | `--benchmark minif2f --condition lm --condition area_l0.3 --condition full --condition random_l0.3` | 19 h, $16 | $29 |
| 3 ablations miniF2F | `--benchmark minif2f --condition area_l0.1 --condition area_l0.6 --condition diversity --condition stuck_escape` | 19 h, $16 | $29 |
| 4 PutnamBench subset | `--benchmark putnam --limit 150 --condition lm --condition full --condition random_l0.3` | 15 h, $13 | $13 |

Steps 1, 2 and 4 come to about $30 expected; all four to about $46 of the $100 credit.
PutnamBench problems mostly fail, so they use their full wall time and cost about $0.10
each per condition: grow `--limit` rather than adding conditions there. Use `--max-usd`
on every long run as a hard stop (it stops taking new problems once that much instance
time has passed; the problems in flight finish).

## One-time setup

From your laptop, after launching the instance (Ubuntu Deep Learning AMI, 200 GB gp3,
security group with SSH from your IP only):

```bash
ssh ubuntu@<ip>
git clone https://github.com/pikulsomesh/markov-lean && cd markov-lean
bash scripts/setup_ec2.sh            # vLLM, this package, Lean + Mathlib, the REPL
.venv/bin/markov-lean fetch minif2f putnam
```

The area conditions need the saved Markov model (written by `markov-lean pilot` after PR
#13). Copy it from the Mac Mini; it holds a pickle, so only copy a directory you produced:

```bash
# on the Mac Mini, from the repo root
ssh ubuntu@<ip> mkdir -p markov-lean/results/pilot3
scp -r results/pilot3/msm ubuntu@<ip>:markov-lean/results/pilot3/
```

If it lives elsewhere, pass `--msm-dir <path>` or edit `msm_dir` in the config.

## Each session

```bash
ssh ubuntu@<ip>
cd markov-lean && git pull
tmux new -s eval                      # keeps the run alive if SSH drops; reattach: tmux attach -t eval
bash scripts/serve_prover.sh          # waits until vLLM answers
.venv/bin/python -m markov_lean.search --check --config configs/search.yaml

E=".venv/bin/markov-lean search-eval --config configs/search_eval.yaml --out results/search_eval"

# 1. smoke: checks the whole path and calibrates the estimate
$E --benchmark minif2f --limit 5 --condition lm --condition area_l0.3 --estimate
$E --benchmark minif2f --limit 5 --condition lm --condition area_l0.3 --yes

# 2. core miniF2F (look at the estimate first)
$E --benchmark minif2f --condition lm --condition area_l0.3 --condition full --condition random_l0.3 --estimate
$E --benchmark minif2f --condition lm --condition area_l0.3 --condition full --condition random_l0.3 --yes --max-usd 20

# 3. ablations, 4. PutnamBench subset: same pattern with the flags from the table above
```

Watch memory in a second tmux pane with `free -h` and `nvidia-smi`. Each worker runs its own
Lean REPL with Mathlib loaded (Mathlib's files are memory-mapped and shared, so this is
mostly fine). If memory gets tight, use `--workers 2`; the run resumes where it stopped.

To stop a run: Ctrl-C. Finished problems are saved; rerun the same command to resume.

## Report

```bash
.venv/bin/markov-lean search-report --runs results/search_eval
# or several roots, or a glob of benchmark folders; the report goes to their common parent
.venv/bin/markov-lean search-report --runs results/search_eval/* --out results/search_report \
    --label 'lm=LM only ($\lambda = 0$)' --label 'area_l0.3=Area prior ($\lambda = 0.3$)'
```

`--runs` takes runs roots, benchmark, condition or seed folders, or `summary.jsonl` files.
The output folder (default `<runs>/report`) gets:

| file | what |
|---|---|
| `pass_at_k.csv`, `pass_at_k.tex` | pass@k per benchmark and condition, bootstrap 95% CIs over problems; the `.tex` is pass@1 with CIs, best per column in bold |
| `pass_at_k_all.tex` | pass@k for every k reported (when there are several seeds) |
| `paired.csv`, `paired.tex` | each condition against `lm` on the problems both finished: paired bootstrap of the pass@1 gain, exact McNemar with Holm correction |
| `by_area.csv` | pass@1 by the Markov model's area of each problem's first goal state (`root_area`), or by miniF2F's problem group |
| `by_area_gain.csv`, `by_area_gain.tex` | each condition's paired gain over `lm` within each area |
| `budget.csv`, `budget_gain.csv` | share solved within each expansion budget, and the paired gain over `lm` at each budget, with CIs |
| `search_stats.csv`, `search_stats.tex` | proof length (tactics), expansions to solve, nodes per run, stop reasons |
| `effort.csv` | on runs both solved (same problem and seed): geometric-mean expansion ratio against `lm`, proof-length difference, Wilcoxon test |
| `report.md` | all tables in one page |
| figures | `fig_pass_at_1.pdf`, `fig_budget.pdf`, `fig_budget_gain.pdf`, `fig_effort.pdf`, and per benchmark `fig_by_area_<b>.pdf`, `fig_area_gain_<b>.pdf` |

The `.tex` files are bare `tabular`s to `\input` inside a `table`. `--label NAME=LATEX` and
`--bench-label NAME=LATEX` set row and column names (benchmarks default to miniF2F and
PutnamBench). To try it without a GPU: `markov-lean search-report --runs tests/fixtures/search_runs/*`
(synthetic runs, see `tests/search_fixture.py`).

Copy results back to the Mac Mini before stopping the instance:

```bash
# on the Mac Mini
rsync -av ubuntu@<ip>:markov-lean/results/search_eval/ results/search_eval/
```

## Stop the instance

The instance bills by the second while it runs, whether or not a search is running.

```bash
kill $(pgrep -f "vllm serve")   # optional; stopping the instance ends it anyway
sudo shutdown -h now            # from the instance; stops it if its shutdown behaviour is "stop" (the default)
```

Or stop it in the EC2 console (Instances, select it, Instance state, Stop instance), or from
your laptop with `aws ec2 stop-instances --instance-ids <id>`. Check in the console that it
shows **Stopped**. A stopped instance still pays for its disk (about $16 a month for
200 GB); terminate it when the project no longer needs it.

To end a long run and stop the instance in one go:

```bash
$E <flags> --yes --max-usd 20; sudo shutdown -h now
```

## On the Mac Mini (no GPU)

When there is no GPU, the core comparison runs on the Mac Mini (M2, 16 GB) with the local
prover from `MAC_SMOKE.md`: a 4-bit GGUF of DeepSeek-Prover-V1.5-RL run by llama.cpp on Metal.
Do the smoke test (`MAC_SMOKE.md`, steps 1 to 6) first.

```bash
scripts/mac_overnight.sh --estimate     # the plan in hours; runs nothing
scripts/mac_overnight.sh                # the run; rerun the same command to resume
scripts/mac_overnight.sh --status       # from another terminal or over SSH
scripts/mac_overnight.sh --report       # a report from what has finished so far
```

It runs miniF2F test under `lm`, `area_l0.3` and `random_l0.3`, interleaved per problem in the
fixed order, with `configs/search_eval_mac_overnight.yaml`: 12 expansions of 8 samples and 180 s
per problem, one worker. It keeps the Mac awake with `caffeinate`, logs one line per finished
problem to `results/mac_overnight/logs/`, and writes `results/mac_overnight/report/` at the end.
It removes API and AWS credentials from the run's environment and never calls either.

The area conditions need a saved Markov model: `results/pilot3/msm` by default (from
`markov-lean pilot` on the Mac), or `MSM_DIR=<a msm folder you produced>`. The script stops
before any compute if it is missing, and so does `search-eval` for any prior that does not load.

### Plans and what they can detect

| plan | command | problems | time | detectable gain |
|---|---|---|---|---|
| one night | `PLAN=night scripts/mac_overnight.sh` | about 110 (80 to 115) | 12 h | about 9 points (9 to 11) |
| all of miniF2F test | `scripts/mac_overnight.sh` (`PLAN=full`, the default) | 244 | about 26 h expected, 37 h at most, plus pauses | about 6 points (4.5 to 7) |

Time: about 15 s an expansion on the M2 (8 samples at roughly 15 to 30 tokens a second, plus Lean)
and 3 conditions a problem. A problem that is solved early takes a minute or less; one that fails
uses its 180 s. The estimate switches to measured times once problems finish. `PLAN=full` stops
after `HOURS=48` in all; a stopped run is a prefix of the fixed order, so `PLAN=night` tonight and
the same command again tomorrow is the same run as `PLAN=full`.

Detectable gain: the smallest true gain of the area prior over `lm` in solve rate that an exact
two-sided McNemar test at alpha 0.05 finds with 80% power, assuming 10% of problems are solved by
one condition and not the other (the range is for 6% to 15%). Every condition sees the same tactic
samples at a goal state, so only problems where reordering changes the outcome disagree. With 6%
disagreeing, one night is too small for any gain to reach 80% power. `report.md` recomputes this
from the disagreement it observes, in its "Detectable effect" table, so after the first night you
know which plan you need. The `random_l0.3` comparison uses the same test with Holm correction.

Other conditions (`area_l0.1`, `area_l0.6`) are in the config: add them with
`CONDITIONS="lm area_l0.3 random_l0.3 area_l0.6"` once the core comparison is done; finished
problems are skipped.

### PutnamBench on the Mac

`BENCHMARK=putnam` runs the same three conditions and budget on PutnamBench. It reads the pinned
copy that `scripts/fetch_gold_sources.sh` puts in `data/raw/gold/PutnamBench` (672 problems, the
commit the gold set uses; about 11 MB). Nearly every problem fails and uses its full 180 s under
every condition, so a problem takes about 9 minutes and gives almost no disagreeing problems: the
run shows whether the area prior changes anything on harder statements, not a powered test. Run a
prefix of the fixed order with `LIMIT`:

```bash
scripts/fetch_gold_sources.sh           # skips what is already there
BENCHMARK=putnam LIMIT=150 scripts/mac_overnight.sh --estimate
BENCHMARK=putnam LIMIT=150 scripts/mac_overnight.sh
```

150 problems take about 23 hours plus pauses (`PLAN=full` stops after 48 h; rerun to go on). Keep
the default `OUT=results/mac_overnight`: the runs go to `results/mac_overnight/putnam/`, next to
miniF2F, `report/` then covers both benchmarks, and `markov-lean paper` fills the Putnam search
macros from the same folder. PutnamBench at that commit targets Mathlib v4.33.1, so with an older
Mathlib in `lean/` some statements fail to elaborate; they are counted as statement errors
(`resSearchStatementErrorsPutnam`) and fail under every condition.

These numbers are for this model and budget only. Pass rates from the Mac are not comparable with
the vLLM runs on EC2 or with published numbers; `search-report` puts that warning at the top of
`report.md`, in every `.tex` file and in `LOCAL_MODEL.txt` whenever a run used llama.cpp.

### Sharing the Mac with GitHub runner jobs

The Mac also hosts self-hosted GitHub Actions runners. The search holds about 9 GB, so the script
runs it under `markov_lean.mac_guard`, which gives that memory back on its own:

* it stops the search cleanly (as Ctrl-C would) when a runner job starts (a `Runner.Worker`
  process; the always-on `Runner.Listener` doesn't count), when macOS memory pressure reaches
  *warn*, when available memory falls under `MIN_FREE_GB` (1) or when swap grows by more than
  `MAX_SWAP_GROWTH_GB` (1.5) since the search started. Stopping, not pausing, is what frees the
  memory. The problem in flight is lost and rerun; finished problems are kept.
* it starts the search again once no runner job has run for `COOLDOWN` seconds (120), pressure is
  normal and `RESUME_FREE_GB` (4) is available. Until then it waits and says why, in the log and in
  `--status`.
* it gives up after 3 failed starts in a row (an error in the search itself), and stops after
  `HOURS` in all, pauses included.
* guarded jobs take turns through one lock file, so the search and the local judge
  (`scripts/mac_local_judge.sh`, about 7 GB) never run side by side.

It changes no system setting and never touches the runner. `YIELD_TO_RUNNER=0` keeps only the
memory checks. `python -m markov_lean.mac_guard --check` prints what it reads right now.

To have the run start again after a restart or a logout, an optional launchd agent:

```bash
scripts/mac_overnight.sh --agent-plist > ~/Library/LaunchAgents/com.markov-lean.search.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.markov-lean.search.plist
# remove: launchctl bootout gui/$(id -u)/com.markov-lean.search && rm ~/Library/LaunchAgents/com.markov-lean.search.plist
```

It starts the script at login and again if it crashes, not after it finishes.

## Tests

`tests/test_search_eval.py`, `tests/test_search_report.py`, `tests/test_mac_overnight.py` and
`tests/test_mac_guard.py` run in CI with no GPU, Lean or network: Lean is a fake REPL, the prover a
fixed table and the guard's machine readings are faked.
