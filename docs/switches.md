# Within-proof area switching (`markov-lean switches`)

Gate 1 (`markov-lean significance`) shows that macrostates agree with the judges' MSC labels. Agreement alone does not show that a single proof moves from one area to another: a partition can match the labels while every proof stays in one area. `markov-lean switches` tests movement inside each proof.

```
markov-lean switches --run results/pilot3
markov-lean switches --run results/real-labelfree/full --sources macrostate \
    --steps data/leandojo/leandojo_train.parquet --out results/real-switches
```

It makes no API call and refits nothing. Outputs go to `<run>/switches/` unless `--out` is given.

## Area sequences

Each proof gives one sequence of areas in step order, separately for each source:

| source | from | notes |
|---|---|---|
| `judge` | `<run>/judge/labels.parquet`, step labels | consensus judge, else the two judges' agreed steps, else the one judge (`--judge` picks one). Long proofs are judged on an even sample of at most 48 steps, so the sequence is the labelled steps in order. |
| `macrostate` | `<run>/msm/assignments.parquet` | every assigned step |
| `true_area` | synthetic steps table | used in place of `judge` on synthetic runs without labels |

Proofs with fewer than two area-labelled steps cannot switch and are dropped (the count is reported). Mathlib folders come from the steps table (`--steps`, default `<run>/judge/sample_steps.parquet`, or rebuilt from `--config`).

## Statistics (each proof weighted equally)

| statistic | meaning |
|---|---|
| `has_switch` | share of proofs with at least one switch |
| `switch_rate` | switches / (length - 1), so long proofs don't dominate |
| `mean_dwell` | length / number of runs: steps spent in an area before switching (runs at proof ends are censored; nulls are censored the same way) |
| `off_mode` | share of a proof's steps outside its most common area |

## Nulls (all keep each proof's length)

| null | keeps | what it tests |
|---|---|---|
| `shuffle` | each proof's own area composition, order randomised | Ordering. Switch rate below it (dwell above it) means a proof's steps in one area come in runs, segments rather than flicker. `has_switch` and `off_mode` are fixed by this null. |
| `markov` | a first-order chain per Mathlib folder (one chain without folders), started at the proof's first area | How switches spread over proofs. `has_switch` below it means switching is concentrated in a minority of proofs; dwell above it means memory beyond one step. Calibrated on synthetic first-order chains (tests). `--markov-group mode` (per main area) is offered but biased, since it conditions on the outcome. |
| `pooled` | the area mix of each Mathlib folder; steps dealt to that folder's proofs at random (`--pooled-within all` for all proofs) | Coherence. Switching far below it means proofs keep to an area beyond what their folder explains. |
| `noise` | each proof in its main area throughout, each step moved elsewhere with probability epsilon | Movement beyond label noise. `off_mode` above it means proofs spend more of their steps away from their main area than a labelling error rate explains. For `judge`, epsilon = 1 - sqrt(step agreement of the two judges); `--noise-rate` sets it for every source; otherwise skipped. |

`--n-perm` draws per null give permutation p-values, `(1 + #{as or more extreme}) / (n + 1)`, two-sided as twice the smaller tail. Paired differences (observed minus each proof's own null mean) get 95% percentile bootstrap intervals over proofs (`--n-boot`). Everything is seeded.

## Strata

Every statistic and null is repeated by Mathlib folder (folders with at least `--min-proofs` proofs; the rest as `other`) and by length (labelled steps 2-3, 4-7, 8-15, 16-31, 32+), with Holm and Benjamini-Hochberg corrections across the strata of each kind.

## Outputs

| file | contents |
|---|---|
| `switches.json` | settings, inputs, `headline` numbers per source, every statistic, null, stratum, interval and p-value |
| `switches.csv` | the same as one long table (source x stratum x null x statistic) |
| `proofs.csv` | per proof and source: length, folder, main area, switches, statistics, per-proof null means |
| `dwell.csv` | run-length distribution, observed and under each null (first 100 draws) |
| `REPORT.md` | readable summary |
| `table_switches.tex` | booktabs tabular: observed [95% CI] and each null's mean (two-sided p) |
| `switches.pdf`, `.png` | statistics against nulls, dwell distribution, switch rate by length (needs the `figures` extra) |

## In the paper

`markov-lean paper` picks the run up as its `switches` input (`--switches <folder>`, else `<pilot>/switches`, `results/switches` or `<full>/switches`). It copies `table_switches.tex` into the switching table (`\resultstable{table_switches}`, `tab:switches`) and `switches.pdf` into its figure (`\maybefigure{switches}`, `fig:switches`), both in `paper/sections/experiments.tex`, and writes the `\resSwitch...` numbers from `switches.json["headline"]` (list in `docs/paper_build.md`). Until a run is in, the table shows red dashes and the figure the synthetic preview of `scripts/preview_figures.py`.

```
markov-lean paper --switches results/real-switches/train_main
make -C paper
```
