# Results

Run outputs land here. They can be large (the full run caches its features there), so commit only this file. Time is counted in proof steps throughout.

## `results/full/` — `markov-lean full`

The Markov state model fit on every LeanDojo train proof with at least 8 steps (the pilot's filter), with 10 macrostates at lag 1, and every other dataset placed on it. Settings are in `configs/full.yaml`.

| path | what it holds |
|---|---|
| `RUN.md` | headline table: switch rate and dwell per dataset with 95% intervals |
| `full_report.json` | training counts, model summary and failed checks, skipped datasets, seconds per stage, peak memory |
| `msm/` | the fitted model in the `MSMResult.save` layout: `featurizer.pkl`, `centers.npy`, `memberships.npy`, `micro_to_macro.npy`, `connected.npy`, `coarse_T.npy`, `micro_T.npy`, `assignments.parquet`, `implied_timescales.csv`, `ck_test.csv`, `microstate_cv.csv`, `msm_report.json`. `prior_loader.build_area_prior(msm_dir="results/full/msm")` reads it for the search. |
| `cache/` | streamed training features (`features.npy`, float32 memmap), the featurizer, and `meta.json`; a rerun with the same data and settings reuses them (`--refit-features` to rebuild) |
| `project/` | the cross-dataset analysis (below) |
| `figures/` | PDFs in the paper style: `implied_timescales`, `chapman_kolmogorov`, `model_transitions`, `transitions_<dataset>`, `switch_rates`, `occupancy`, `dwell_times`, `switches_per_proof` |

## `project/` — per dataset and across datasets

Every goal state goes to its nearest k-means microstate and that microstate's macrostate; nothing is refit. `leandojo_train` is in-sample (the fit's own assignments); every other dataset is out of sample. Intervals are 95% percentile bootstraps that resample whole proofs (1,000 resamples, seeded).

| file | one row per | columns |
|---|---|---|
| `summary.parquet` | dataset | `n_proofs`, `n_steps`, `n_moves`, `has_transitions`, `median_proof_length`, and for each statistic below its value with `_lo` / `_hi` |
| `transitions.parquet` | dataset × from × to | `count` of moves, `probability` (row-normalized), `lo`, `hi` |
| `occupancy.parquet` | dataset × macrostate | `n_steps`, `share` of goal states, `lo`, `hi` |
| `dwell_hist.parquet` | dataset × dwell length | `n_runs`, `n_uncensored`, `n_left_censored`, `n_right_censored` |
| `differences.parquet` | dataset × statistic | the dataset minus the reference (`leandojo_test`), with an interval from independent bootstraps |
| `summary.json` | | all of the above nested by dataset, transition matrices with intervals, and the model's own `coarse_T` |
| `<dataset>/placed.parquet` | goal state | `proof_id`, `step`, `microstate`, `macrostate`, `membership` |
| `<dataset>/proof_stats.parquet` | proof | `n_steps`, `n_moves`, `n_switches`, `switch_rate`, `n_macrostates_visited`, `dominant_macrostate` |
| `<dataset>/dwell_runs.parquet` | run of one macrostate | `proof_id`, `macrostate`, `dwell`, `left_censored`, `right_censored` |

Statistics (a move is a pair of steps `t` and `t + lag` in the same proof):

- `pooled_switch_rate`: switches over all moves in the dataset.
- `mean_switch_rate`: the average of each proof's own switch rate, so long proofs do not dominate.
- `mean_switches_per_proof`, `share_proofs_with_switch`: over proofs with at least one move.
- `mean_dwell`: average run length in one macrostate, over proofs with a move. Runs that touch the start or end of a proof are censored (the real dwell is at least that long); `mean_dwell_uncensored` uses only runs with a switch on both sides.
- `occupancy`, `transition_matrix`, `stay_probability` (its diagonal).

### What each dataset can show

- **LeanDojo val and test**: held-out proofs with full trajectories, filtered like the fit (8+ steps); `leandojo_test_all` keeps every multi-step proof.
- **`leandojo_train_short`**: the 2 to 7 step training proofs the fit left out.
- **miniF2F, ProofNet, PutnamBench, Lean Workbook**: the processed files hold one goal state per theorem (`step = -1`), and Workbook's tactic file one step per problem, so these give **occupancy only** (`has_transitions = false`). Their transition statistics need proofs: the `*_search` entries read `results/search/<benchmark>/steps.parquet` from `markov_lean.search` when those runs exist, and `markov-lean project --dataset name=path` adds any other steps table.

## `results/pilot*/` — `markov-lean pilot`

The Phase 2 pilot and its gate 1 call; see `src/markov_lean/pilot.py`.
