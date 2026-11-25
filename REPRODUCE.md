# Reproducing the paper and the dataset

From a clean clone to every table and figure in `paper/`, and to the dataset release folder.
`scripts/reproduce.sh` runs the stages below in order; each stage skips itself when its outputs
already exist, so you can stop and rerun at any point.

```bash
git clone https://github.com/pikulsomesh/markov-lean && cd markov-lean
bash scripts/reproduce.sh --dry-run          # the plan and every command; runs nothing
bash scripts/reproduce.sh --labels <labels.parquet>   # everything on the Mac, with judge labels you have
```

`make reproduce-plan` and `make reproduce` do the same. Nothing in this repo uploads anything,
creates cloud resources or spends money unless you ask for it with a flag (below).

## Where each stage runs and what it costs

| stage | runs on | needs | cost | outputs |
|---|---|---|---|---|
| `setup` | Mac (or EC2) | Python 3.10+ | free | `.venv`, tests pass |
| `data` | Mac | network to Zenodo, Hugging Face, GitHub | free | `data/raw/`, `data/processed/*.parquet` |
| `synthetic` | Mac | | free | `results/pilot_synthetic/` (planted-area sanity check) |
| `pilot` | Mac, 16 GB | `data/processed/leandojo_train.parquet` | free | `results/pilot3/msm/`, `judge/sample_steps.parquet` |
| `judge` | Mac | cached labels, **or** `ANTHROPIC_API_KEY` and `--run-judges` | free with cached labels; about **$78** to judge again | `results/pilot3/judge/labels.parquet` |
| `score` | Mac | the labels | free | `results/pilot3/pilot_report.json` (gate 1, agreement) |
| `full` | Mac, 16 GB | `data/processed/` | free | `results/full/` (model on all train proofs, cross-dataset analysis) |
| `search` | **EC2 g6.xlarge** (L4 GPU) | `scripts/setup_ec2.sh`, `--run-search` | about **$30** (miniF2F core + PutnamBench subset) to **$46** (with ablations) of instance time | `results/search_eval/` |
| `report` | Mac | the search runs, copied back | free | `results/search_eval/report/` |
| `paper` | Mac | LaTeX for the PDF (`brew install --cask mactex-no-gui`) | free | `paper/figures/*.pdf`, `paper/tables/*.tex`, `paper/main.pdf` |
| `release` | Mac | the labels | free | `release/` (dataset folder, checked) |

Run a subset with `--stage <name>` (repeatable) or `--from <name>` (that stage and everything
after it). Run times depend on the machine and have not been measured end to end yet.

### The Anthropic key and the judge cost

Only the `judge` stage talks to the Anthropic API, and only with `--run-judges`. It runs
`python -m markov_lean.judge estimate` first, then asks you to type `yes`, then submits Message
Batches and waits for them. `ANTHROPIC_API_KEY` is read from your shell; never put it in a file in
this repo.

- The pilot3 run judged 8,423 proofs with two judges (16,846 requests): the estimate was **$77.64**
  at batch prices (`configs/judge.yaml` has the prices and heuristics). The hard cap is `$85`
  (`budget.max_dollars`; `MAX_DOLLARS=` can only lower it).
- If `results/pilot3/judge/run/batches.json` exists, the stage collects those batches instead of
  submitting again, so a rerun after an interruption does not pay twice.
- **Reusing labels is free.** Pass `--labels <file>` (the `labels.parquet` written by
  `judge collect`), or leave `results/pilot3/judge/labels.parquet` in place. The released
  dataset's `data/labels.parquet` has the same columns (`proof_id`, `step`, `judge`, `msc`,
  `confidence`) plus `consensus` rows, which the pilot uses directly. The judge sample is drawn
  with a fixed seed, so pass 1 on the same data picks the same proofs.

### The GPU instance

The search stage runs only on a g6.xlarge you launch yourself; `docs/search_eval.md` is the full
runbook (setup, the four runs with their `--estimate` and `--max-usd` caps, copying results back,
stopping the instance). The scripts never call the AWS API. On the instance:

```bash
bash scripts/setup_ec2.sh
bash scripts/reproduce.sh --stage search --run-search   # serves the prover, runs the three plans
```

Then on the Mac copy `results/search_eval/` back with `rsync` and run `--from report`. The area
conditions need the Markov model: copy `results/pilot3/msm/` to the instance first (see the runbook).

## Which stage makes each table and figure

`scripts/paper_assets.py` (the `paper` stage, or `make paper-assets`) builds every asset whose
inputs exist and lists the rest as skipped. Figures go to `paper/figures/`, where `\maybefigure`
uses them in place of the synthetic previews. Table bodies go to `paper/tables/`; paste them over
the `\tbd` rows (or `\input` them).

| asset in the paper | file | made from | stage |
|---|---|---|---|
| Table `tab:data` (proofs and steps per source) | `paper/tables/data.tex` | `results/pilot3/pilot_report.json`, `data/processed/`, `results/pilot_synthetic/` | `data`, `synthetic`, `score` |
| Table `tab:areas` (NMI and ARI against the judges) | `paper/tables/areas.tex` | pilot3 assignments and consensus labels; folder and random-partition baselines; synthetic run | `score`, `synthetic` |
| Table `tab:search` (pass@k) | `paper/tables/search.tex` | `results/search_eval/report/pass_at_k.tex` | `search`, `report` |
| Figure `implied_timescales` | `paper/figures/implied_timescales.pdf` | `<msm>/implied_timescales.csv` | `full` (else `pilot`) |
| Figure `transition_matrix` | `paper/figures/transition_matrix.pdf` | `<msm>/coarse_T.npy` | `full` (else `pilot`) |
| Figure `chapman_kolmogorov` | `paper/figures/chapman_kolmogorov.pdf` | `<msm>/ck_test.csv` | `full` (else `pilot`) |
| Figure `judge_confusion` | `paper/figures/judge_confusion.pdf` | pilot3 assignments and consensus labels | `score` |
| Figure `area_flow` | `paper/figures/area_flow.pdf` | consensus step labels | `judge` |
| Figure `search_ablation` | `paper/figures/search_ablation.pdf` | `results/search_eval/report/fig_pass_at_1.pdf` | `search`, `report` |
| Algorithm 1 | `paper/sections/search.tex` | written by hand | |

`<msm>` is `results/full/msm` when the full fit exists, else `results/pilot3/msm`. The full run
also writes cross-dataset figures and tables to `results/full/figures/` and `results/full/project/`
(see `results/README.md`) for the appendix.

## The dataset release

```bash
markov-lean export-dataset --run results/pilot3 --out release/     # or: make release
markov-lean check-release release/                                  # or: make release-check
```

The export writes `data/steps.parquet`, `data/labels.parquet`, `data/proofs.parquet`, a dataset
card (`README.md` with a Hugging Face YAML header), `SCHEMA.md`, `run_summary.json`,
`MANIFEST.json` and `SHA256SUMS`. It drops judge labels with invalid codes or unknown steps
(and counts them in the card), keeps only relative Mathlib file paths, and refuses to write
anything that looks like an API key. `--no-text` leaves the tactic and goal-state text out.

The check fails on a missing or unexpected file, a checksum or row-count mismatch, columns that
differ from the schema, repeated keys, codes that are not MSC 2020 top-level codes, labels with no
matching step, a card without the required YAML keys or with TODO items, and anything in any file
that looks like a key or a local path (`/Users/...`, `/home/...`, `~/...`). Run it again after
editing the card by hand, and rerun the export to refresh the checksums.

Uploading to Hugging Face or Zenodo is a manual step from the checked folder. Before uploading:
confirm the licence in the card (labels CC BY 4.0; goal states and tactics stay under Mathlib's
Apache-2.0 and LeanDojo Benchmark 4's CC BY 2.0), and for the anonymous submission period keep the
upload anonymous (the card names no author).

## Tests

`tests/test_release.py` and `tests/test_reproduce.py` run the export, the check, `paper_assets.py`
and `reproduce.sh` (pilot, labels, score, paper, release stages) end to end on synthetic data, with
fake judges and no network or API.
