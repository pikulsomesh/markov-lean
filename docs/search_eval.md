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

## Tests

`tests/test_search_eval.py` and `tests/test_search_report.py` run in CI with no GPU, Lean or
network: Lean is a fake REPL and the prover a fixed table.
