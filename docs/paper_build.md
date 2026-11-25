# Building the paper from results

`markov-lean paper` turns whatever result folders exist into the paper's figures, tables and headline numbers, and keeps the synthetic preview for everything that has not run yet. It makes no API calls and launches nothing, so run it after every run that lands, then rebuild the PDF.

```bash
markov-lean paper                        # reads results/, writes paper/figures/
markov-lean paper --dry-run              # says what would be real; writes nothing
make -C paper                            # rebuild main.pdf
```

It prints one line per item: `real` (drawn from a run now), `preview` (still the synthetic preview or a red dash) or `earlier` (a real figure from an earlier build that this run could not redraw; delete it to go back to the preview).

## What it reads

Every input is optional. A `--<name> PATH` flag picks a folder instead of searching `--results`.

| input | written by | used for |
|---|---|---|
| `--pilot` | `markov-lean pilot` (real data); the labelled `results/*/pilot_report.json` with the highest number wins | data counts, NMI/ARI of the macrostates and the folder baseline, gate 1, judge kappa, `judge_confusion`, `area_flow` |
| `--synthetic` | `markov-lean pilot` with `data.source: synthetic` | the synthetic rows of tables 1 and 2 |
| `--full` | `markov-lean full` | LeanDojo counts of the full fit (replacing the pilot's), Workbook and PutnamBench counts from `project/summary.parquet` |
| `--msm` | `<full>/msm`, else `<pilot>/msm` | `implied_timescales`, `transition_matrix`, `chapman_kolmogorov`, lag, m, slowest timescale |
| `--search` | `markov-lean search-eval` (`results/search_eval`) | table 3, `search_ablation`, seeds, per-benchmark gain of `--main-condition` (default `full`) over `--baseline` (default `lm`) |
| `<search>/report` | `markov-lean search-report --runs <search>` | its `.tex` tables and `fig_*.pdf` figures, copied as `search_<name>` |
| `--robustness` | `markov-lean robustness` (`<pilot>/robustness`) | table 2: every grouping of `baselines.csv` (macrostates, Mathlib folder and namespace at 1 and 2 levels, k-means on goal states and on microstate centers, microstates, the judges' proof label, random) with NMI, AMI, ARI and their intervals, all on the same steps, and the row-shuffle and proof-swap nulls of `robustness.json`; the judge ceiling; its figures are copied as `robustness_<name>.pdf` |
| `--significance` | `markov-lean significance` (`<pilot>/significance`) | table 2: the within-proof shuffle null, and the intervals of the macrostate and folder rows when there is no robustness run |
| `--bridges` | `markov-lean bridges` (`results/bridges`) | `bridges.pdf`, copied as is |
| `--switches` | `markov-lean switches` (`<pilot>/switches`, else `results/switches`, else `<full>/switches`; the pilot's comes first because it has judge labels as well as macrostates) | the switching table (`table_switches.tex`, copied, or rebuilt from `switches.json` when only that is there), `switches.pdf` (copied), and the `\resSwitch...` numbers from `switches.json["headline"]` |

`area_flow` uses the judges' MSC codes once labels exist and the macrostates before that (`--flow-areas` fixes one). Flows count only steps `t` and `t + 1` of the same proof.

## What it writes

Everything goes to `paper/figures/`:

| file | read by |
|---|---|
| `<figure>.pdf` | `\maybefigure{<figure>}` |
| `table_data.tex`, `table_areas.tex`, `table_search.tex`, `table_switches.tex` | `\resultstable{<name>}` inside each table environment (the caption and label stay in the section file) |
| `results_macros.tex` | `paper/results.tex`, input by `main.tex` after `macros.tex` |
| `bridges.pdf`, `robustness_<name>.pdf`, `search_<name>.tex` / `.pdf` | nothing yet; there for the text to pick up |

`paper/figures/preview/` holds the preview of each table and of the macros (red dashes only; in table 2 a plain dash marks a cell no run fills, such as AMI under a null). The command rewrites them on every run so they match the code; `--redraw-previews` also redraws the preview figures with `scripts/preview_figures.py`. A table or macros file with no real number is not written, and an old generated one is removed, so the PDF never shows stale numbers as real.

## Numbers the text can use

Each is a macro that prints a red dash until its run is in. The full list, with where each comes from, is at the top of `paper/figures/preview/results_macros.tex`.

| macro | value |
|---|---|
| `\resNumProofs`, `\resNumSteps` | LeanDojo proofs and steps the model was fit on |
| `\resNumMicrostates`, `\resLag`, `\resNumMacrostates`, `\resChecksFailed` | the model |
| `\resSlowestTimescale` | slowest implied timescale, in proof steps |
| `\resMacroNMI`, `\resMacroARI`, `\resFolderNMI`, `\resFolderARI`, `\resRandomNMI`, `\resRandomARI` | table 2 |
| `\resMacroAMI`, `\resMacroNMILo`, `\resMacroNMIHi`, `\resFolderAMI`, `\resNamespaceNMI`, `\resNamespaceAMI`, `\resKmeansNMI`, `\resKmeansAMI` | table 2's AMI, the macrostates' NMI interval, and the namespace and k-means baselines |
| `\resNullRowNMI`, `\resNullRowP`, `\resNullProofNMI`, `\resNullProofP`, `\resNullWithinNMI`, `\resNullWithinP` | mean NMI of the macrostates under each null and the observed NMI's p-value |
| `\resGateOne` | `passed` or `failed` |
| `\resJudgeKappa`, `\resJudgeAgreement` (%), `\resJudgeCeilingNMI`, `\resNumLabelledSteps` | the judges |
| `\resSyntheticNMI` | macrostates against the planted areas |
| `\resFlowsShown`, `\resFlowCoverage` (%) | the flow figure's caption |
| `\resSearchSeeds` | seeds per problem |
| `\resSearchBase<B>`, `\resSearchArea<B>`, `\resSearchGain<B>` | pass@1 (%) with the LM prior, with the area prior, and the difference in points, for `<B>` = `MiniF`, `Putnam` (any other benchmark that runs gets its own, named from it) |

| `\resSwitch<S>Proofs`, `\resSwitch<S>Has` (%), `\resSwitch<S>HasLo`, `\resSwitch<S>HasHi` | proofs with two or more labelled steps; share of them with at least one switch and its 95% interval, for `<S>` = `Judge` (judge MSC codes) or `Macro` (macrostates) |
| `\resSwitch<S>Rate`, `\resSwitch<S>Dwell`, `\resSwitch<S>OffMode` (%) | switches per move, mean dwell in steps, share of steps off the proof's main area |
| `\resSwitch<S>RateShuffle`, `\resSwitch<S>RateShuffleP` | switches per move under the within-proof shuffle, and the two-sided p |
| `\resSwitch<S>HasMarkov` (%), `\resSwitch<S>HasMarkovP`, `\resSwitch<S>HasPooled` (%), `\resSwitch<S>HasPooledP` | share of proofs with a switch under the Markov and folder-pool nulls, and the p of the observed share |
| `\resSwitch<S>OffModeNoise` (%), `\resSwitch<S>OffModeNoiseP` | share off the main area under the noise floor (judge labels, or any source run with `--noise-rate`) |

Use a macro followed by `{}` when a word comes next (`\resSearchSeeds{} seeds`), so TeX keeps the space.
