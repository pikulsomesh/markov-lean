# Search evaluation runbook

`markov-lean search-eval` runs best-first proof search on miniF2F (test, 244 problems) and
PutnamBench under named prior conditions with fixed budgets, and `markov-lean
search-report` turns the logs into pass@k tables, paired tests, per-area tables and figures.
It runs on the Mac Mini (M2, 16 GB) with a 4-bit local prover, or on one EC2 g6.xlarge
(NVIDIA L4 24 GB, 4 vCPU, 16 GB RAM) with the unquantized prover behind vLLM, which you
launch and stop yourself. Nothing in this repo calls the AWS API.

## The plan: Mac first, one AWS run

The AWS credit is a one-time $100, so the Mac does everything it can and AWS does only what the
Mac can't: run the unquantized model.

| run | where | config and folder | search time | cost |
|---|---|---|---|---|
| miniF2F, 12 expansions, 3 conditions | Mac | `search_eval_mac_overnight.yaml`, `results/mac_overnight` | done | none |
| PutnamBench 150, 12 expansions | Mac | same | about 23 h | none |
| miniF2F, 64 expansions, 3 conditions | Mac | `search_eval_mac_b64.yaml`, `results/mac_overnight_b64` | about 90 h plus pauses, about 1 GB of disk | none |
| miniF2F, 64 expansions, 3 conditions, unquantized | AWS | `search_eval_gpu.yaml`, `results/search_gpu` | about 13 h | about $12 expected, capped at $20 |

The two 64-expansion runs share the budget, sampling (8 samples per expansion), conditions and
problem order, so their difference is the weights and the sampler. `budget.csv` and
`budget_gain.csv` give each run's solve rate and paired gain at every budget up to 64, so the
12-expansion point of the AWS run also checks the first Mac run.

Not planned: PutnamBench at 64 expansions (150 problems x 3 conditions is about 120 h on the Mac
or about $27 on AWS, for a handful of solved problems at most), and more conditions on AWS
(each one adds about a third to the time and cost). Add them on the Mac if the core runs leave
room: `CONDITIONS="lm area_l0.3 random_l0.3 area_l0.6"`; finished problems are skipped.

Folder names matter: `markov-lean paper` reads `results/search_eval` if it holds runs, else the
first folder under `results/` in name order that does. `mac_overnight_b64` and `search_gpu` sort
after `mac_overnight`, so the paper keeps its current numbers until you point it at the new runs.

## What one run is

* A **condition** is a prior from the config (`configs/search_eval.yaml` lists all of them):

  | condition | what it tests |
  |---|---|
  | `lm` | no prior: the prover's own ranking (baseline) |
  | `area_l0.1`, `area_l0.3`, `area_l0.6` | area prior steering by the coarse transition matrix, at three mixing weights |
  | `diversity` | diversity bonus only (favour areas visited less) |
  | `stuck_escape` | stuck escape only (push away from areas with no progress) |
  | `full` | transition target, diversity and stuck escape together |
  | `random_l0.3` | control: random scores mixed in with the same rule and weight |

* **Budgets** are the same for every problem and condition within a run: expansions (prover
  calls), a wall-time safety stop and a depth limit (`budget` in the config). Each summary
  line records them.
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

## The AWS run

### Costs

g6.xlarge on-demand is $0.8048 an hour in us-east-2 (Ohio), plus about $0.016 an hour for the
100 GB disk and the public IPv4 address while it exists. Billing runs from launch until you stop
the instance, whether or not a search is running. A stopped instance still pays for its disk
(about $8 a month); terminate it once the results are on the Mac.

| step | hours | expected |
|---|---|---|
| setup (vLLM, Lean and Mathlib, the REPL, the model download) | about 1.5 | $1.25 |
| smoke test, 5 problems x 3 conditions | about 0.5 | $0.40 |
| main run, 244 problems x 3 conditions, 3 workers | about 13 | $11 |
| a later start to copy the results back, then terminate | about 0.2 | $0.15 |

About $12 to $13 in all. The search time is a guess of 5 s per expansion until the smoke test
measures it; rerun `--estimate` after the smoke test before starting the main run. `--max-usd 20`
on the main run is a hard stop on new problems (problems in flight finish, about 30 minutes at
most), so the whole session stays under about $22 even if the guess is badly off. A stopped run
is a prefix of the fixed problem order and is still a paired comparison. With `--workers 2`
(if memory is tight) the run takes about half as long again.

### Launch

In the EC2 console, region **US East (Ohio)** (the GPU quota is there): Deep Learning Base AMI
with Single CUDA (Ubuntu 24.04), instance type g6.xlarge, your key pair, SSH from your IP only,
100 GiB gp3. The account's "Running On-Demand G and VT instances" quota must be at least 4.

### Setup, once per instance

On the Mac Mini, note the Mathlib tag the Mac runs use, so both machines check proofs alike:

```bash
cat results/mac/mathlib_tag.txt     # or: sed -n 's/^rev = "\(.*\)"/\1/p' lean/lakefile.toml
```

On the instance (`ssh -i ~/.ssh/<key>.pem ubuntu@<public-ip>`):

```bash
tmux new -s setup                   # reattach after a dropped connection: tmux attach -t setup
git clone https://github.com/pikulsomesh/markov-lean && cd markov-lean
MATHLIB_TAG=<tag> bash scripts/setup_ec2.sh
.venv/bin/markov-lean fetch minif2f
bash scripts/fetch_gold_sources.sh  # only for PutnamBench
```

The area condition needs the saved Markov model. From the Mac Mini, in the repo (it holds a
pickle, so only copy a folder you produced):

```bash
ssh -i ~/.ssh/<key>.pem ubuntu@<public-ip> mkdir -p markov-lean/results/pilot3
scp -i ~/.ssh/<key>.pem -r results/pilot3/msm ubuntu@<public-ip>:markov-lean/results/pilot3/
```

### Run

```bash
tmux new -s eval                    # reattach: tmux attach -t eval
cd ~/markov-lean
bash scripts/serve_prover.sh        # waits until vLLM answers; the first start downloads ~14 GB
.venv/bin/python -m markov_lean.search --check --config configs/search.yaml

E=".venv/bin/markov-lean search-eval --config configs/search_eval_gpu.yaml --out results/search_gpu --benchmark minif2f --condition lm --condition area_l0.3 --condition random_l0.3"

# smoke test: 5 problems; they count toward the main run
$E --limit 5 --yes
free -h; nvidia-smi                 # memory headroom; if "available" is under 2 GB, add --workers 2 below

# main run: check the estimate, then run, write the report, pack the small files and stop the instance
$E --estimate
$E --yes --max-usd 20; .venv/bin/markov-lean search-report --runs results/search_gpu; \
  tar czf ~/search_gpu_small.tgz results/search_gpu/runs.jsonl results/search_gpu/report \
      $(find results/search_gpu -name summary.jsonl); sudo shutdown -h now
```

Leave tmux with Ctrl-B then D; the run goes on. Status, from the instance or from the Mac:

```bash
ssh -i ~/.ssh/<key>.pem ubuntu@<public-ip> 'wc -l markov-lean/results/search_gpu/minif2f/*/seed0/summary.jsonl'
```

Each file reaches 244 lines when its condition is done. To stop early: Ctrl-C in tmux, then
`sudo shutdown -h now`; rerunning the same `$E --yes` command resumes.

### Results back to the Mac, then terminate

The instance stops itself after the main run. Start it again in the console (its public IP
changes), then from the Mac, in the repo:

```bash
scp -i ~/.ssh/<key>.pem ubuntu@<new-ip>:search_gpu_small.tgz . && tar xzf search_gpu_small.tgz && rm search_gpu_small.tgz
```

That is the summaries, `runs.jsonl` and the report: a few MB. The traces (`trace.jsonl`, about a
GB) stay on the instance; copy them too only if a later analysis needs them. Then stop the
instance, check that it shows **Stopped**, and terminate it (Instance state, Terminate) so the
disk stops billing.

## Report

```bash
.venv/bin/markov-lean search-report --runs results/search_gpu
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

### Bigger budget on the Mac

`configs/search_eval_mac_b64.yaml` is the same run at 64 expansions (wall time 1200 s, depth 32).
Start it after the PutnamBench run has finished, into its own folder:

```bash
CONFIG=configs/search_eval_mac_b64.yaml OUT=results/mac_overnight_b64 HOURS=168 scripts/mac_overnight.sh --estimate
CONFIG=configs/search_eval_mac_b64.yaml OUT=results/mac_overnight_b64 HOURS=168 scripts/mac_overnight.sh
CONFIG=configs/search_eval_mac_b64.yaml OUT=results/mac_overnight_b64 scripts/mac_overnight.sh --status
```

About 90 h of search (a failing problem now takes about 16 minutes per condition), so about four
days with pauses for runner jobs; `HOURS=168` stops it after a week in all, and rerunning resumes.
It needs about 1 GB of disk for the traces. The report at the end gives the budget curve from 12
to 64 expansions.

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
  *critical*, when available memory falls under `MIN_FREE_GB` (1) or when swap grows by more than
  `MAX_SWAP_GROWTH_GB` (1.5) since the search started. Stopping, not pausing, is what frees the
  memory. The problem in flight is lost and rerun; finished problems are kept. Pressure *warn*
  alone does not stop it: with the prover loaded the Mac sits at *warn* with about 2 GB available
  and no swap, and stopping there made the search unload, restart and stop again without progress.
* it starts the search again once no runner job has run for `COOLDOWN` seconds (120), pressure is
  normal and `RESUME_FREE_GB` (3) is available. Until then it waits and says why, in the log and in
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
