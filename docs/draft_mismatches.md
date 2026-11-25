# Where the draft disagrees with the repo or the data

Checked on Nov 25, 2026 against `main` at 6bf66e9 and the real LeanDojo runs in the project folder
(`results/real-labelfree`, `results/real-switches`). The paper prose is the author's to rewrite; each row
gives a one-line suggestion only. Line numbers are in `paper/sections/`.

## Numbers, sizes and settings

| where | the draft says | the repo or data says | suggested change |
|---|---|---|---|
| `method.tex:60` | the judges label "each of 1,000 sampled proofs" | `configs/pilot.yaml` `judge.sample_proofs: 10000`, above the 8,423 eligible proofs, so the judges label all of them | Say the judges label all 8,423 proofs with 8+ steps. |
| `method.tex:60` | the judges label "each of their steps" | `configs/judge.yaml` `prompt.max_steps_per_proof: 48`: longer proofs show an even sample of 48 steps; 170 of the 8,423 proofs are longer than 48 (max 128) | Say every step of proofs up to 48 steps, and an even sample of 48 steps for the 170 longer ones. |
| `experiments.tex:12` | "keep proofs with at least 8 steps and sample up to 20,000 of them" | only 8,423 train proofs have 8+ steps (59,555 proofs in all, median length 2), so the 20,000 cap never binds | Say all 8,423 proofs with 8+ steps (128,559 steps) are kept; this also fills the TODO on line 12. |
| `search.tex:14` | "at most 600 expansions, a depth of 64 and 600 seconds of wall time" | that is `configs/search.yaml`; the benchmark runs use `configs/search_eval.yaml`: 64 expansions, depth 32, 300 s, 128 tokens per tactic (also `docs/search_eval.md:22`) | Use the `search_eval.yaml` budget, since that is what the reported runs use. |
| `search.tex:71-72` | ablation over λ ∈ {0, 0.1, 0.3, 0.5} | `configs/search_eval.yaml` runs `lm` (λ = 0), `area_l0.1`, `area_l0.3`, `area_l0.6` | Change the grid to {0, 0.1, 0.3, 0.6}, or add a 0.5 condition before the runs. |
| `search.tex:36-37` | TODO: report β and the progress threshold | `AreaPrior` defaults and the `full` condition: β (`escape_strength`) = 2.0, threshold 1.0 goals closed over the last w = 8 expansions | State β = 2 and "fewer than one goal closed in the last 8 expansions". |
| `search.tex:65-66` | search error bars are "95% bootstrap intervals over \resSearchSeeds{} seeds" | `search_report.py` bootstraps over problems; `search_eval.yaml` has `seeds: [0]`, so there is one seed | Say "percentile bootstrap over problems"; mention seeds only if more are added. |
| `appendix.tex:8` | `configs/search.yaml` holds "the prover and the search budget" | the budget for the reported runs is in `configs/search_eval.yaml` (it reads `search.yaml` for the prover and Lean settings) | Name both files: `search.yaml` for the prover, `search_eval.yaml` for the budget and conditions. |
| `appendix.tex:11` | TODO: instance type | `search_eval.yaml` cost block: g6.xlarge (us-east-1) | Fill in g6.xlarge. |
| `appendix.tex:12` | "a spending cap of \$20 per submission" | `configs/judge.yaml` `budget.max_dollars: 85.0` ("All 8,423 pilot proofs estimate about $78") | Use the cap the real submit ran under, once it has run. |
| `abstract.tex:12` | "DeepSeek-Prover-V1.5" | `search.yaml` and `search.tex:9`: DeepSeek-Prover-V1.5-RL | Name the RL checkpoint in the abstract too. |

## Method details

| where | the draft says | the repo or data says | suggested change |
|---|---|---|---|
| `method.tex:47-49` | the lag is "the smallest lag at which [the slow timescales] stop changing" | `configs/full.yaml` fixes `lag: 1`; on the real fit the implied-timescale convergence check fails (max change 0.56 from lag 1 to 2) | Say lag 1 is fixed, and report that the timescales have not converged at it. |
| `method.tex:29-30` | keep the largest strongly connected set | `MSMConfig.disconnected = "nearest"`: steps in dropped microstates get the macrostate of the nearest kept centroid, so every step keeps a label | Add half a sentence on how dropped microstates are labelled. |
| `method.tex:61-62`, `experiments.tex:15`, `experiments.tex:71`, `related.tex:54` | each judge's accuracy on statements whose MSC code is known; Lean Workbook "contributes goal states for judge validation" | `main` has no dataset of statements with known MSC codes and no command that scores a judge on Workbook; open PR 41 adds a gold set (`gold-set`) and `judge-accuracy` | Once PR 41 merges, name the gold set and its source here; until then these lines have nothing behind them. |
| `abstract.tex:8-11`, `intro.tex:30`, `method.tex:58` | the comparison uses "two LLM judges" | still true for the reported comparison; a third local judge (`local-judge`, Qwen 2.5 7B through Ollama) now exists for agreement checks | Mention the local judge only if its agreement numbers go in the paper. |

## Claims the data does not support yet

| where | the draft says | the repo or data says | suggested change |
|---|---|---|---|
| `abstract.tex:10-11` | "the slowest area-switching process relaxes over about \resSlowestTimescale{} tactic steps" | real fit: slowest timescale 79 steps, longer than almost every proof (median 12 among 8+ step proofs), and one macrostate holds 85.7% of steps | Hold this sentence until the giant-state fix; as it stands the slow process is mostly leaving the giant state. |
| `discussion.tex:6-11` | two outcomes framed as conditional (convergence passes, or timescales keep growing) | real fit: Chapman-Kolmogorov passes (max error 0.015) but timescale convergence, spectral gap (1.05) and PCCA+ crispness (0.55) fail | Lead with the mixed result: CK passes, convergence and the gap do not. |
| `discussion.tex:24` | the filter to long proofs "over-represents some areas" | no folder or area breakdown before and after the filter is in the repo yet | Add the folder shares before and after the 8-step filter, or soften to "may over-represent". |
| `experiments.tex` (new `sec:switches`) | TODO only | `results/real-switches/train_main`: macrostates only, 6.5% [5.9, 7.0] of proofs switch; rate 0.012 per step against 0.022 under shuffle (p = 0.002); 11.4% under the Markov null; 56.5% under the folder pool | When writing the paragraph, say proofs switch less than the shuffle null, and note there is no judge row yet. |

## Checked and consistent

TF-IDF over identifiers with prefixes and split parts, 128 SVD dimensions and the dropped single-character
and numeric tokens (`featurize.py`); n ∈ {100, 200, 400} by cross-validated VAMP-2; reversible MLE with
deeptime; m = 10 fixed and the sweep {2, …, 12}; bootstrap error bars over proofs with a fixed seed; the
8-step filter; NMI and ARI, AMI in the robustness table; the folder baseline at the top-level Mathlib folder;
gate NMI 0.15; 16 tactic samples per expansion; λ = 0.3, α = 0.5, w = 8 in the `full` condition; closing
candidates scoring 1; miniF2F test (244); the case-study thresholds (0.7 confidence, 80% agreement, 4 to 14
steps).
