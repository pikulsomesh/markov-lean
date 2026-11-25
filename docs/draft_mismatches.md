# Where the draft disagrees with the repo or the data

Checked on Nov 25, 2026 against `main` (with PR 41 merged) and the real LeanDojo runs in the project folder
(`results/real-labelfree`, `results/real-switches`). Line numbers are in `paper/sections/` and refer to the
draft before the fixes. **Fixed** rows were edited to match the code and data, changing only the wrong
sentence or number; **Left** rows wait on the author or on a result.

## Numbers, sizes and settings

| where | the draft said | the repo or data says | status |
|---|---|---|---|
| `method.tex:60` | the judges label "each of 1,000 sampled proofs" | `configs/pilot.yaml` `judge.sample_proofs: 10000`, above the 8,423 eligible proofs, so the judges label all of them | **Fixed:** "each of the \resNumProofs{} proofs". |
| `method.tex:60` | the judges label "each of their steps" | `configs/judge.yaml` `prompt.max_steps_per_proof: 48`: longer proofs show an even sample of 48 steps; 170 of the 8,423 proofs are longer than 48 (max 128) | **Fixed:** adds "(an even sample of 48 steps for the few longer proofs)". |
| `experiments.tex:12` | "keep proofs with at least 8 steps and sample up to 20,000 of them" | only 8,423 train proofs have 8+ steps (59,555 proofs in all, median length 2), so the 20,000 cap never binds | **Fixed:** "which leaves \resNumProofs{} proofs with \resNumSteps{} steps"; the TODO keeps only the folder share. |
| `search.tex:14` | "at most 600 expansions, a depth of 64 and 600 seconds of wall time" | that is `configs/search.yaml`; the benchmark runs use `configs/search_eval.yaml`: 64 expansions, depth 32, 300 s (also `docs/search_eval.md:22`) | **Fixed:** 64 expansions, depth 32, 300 s. |
| `search.tex:71-72` | ablation over λ ∈ {0, 0.1, 0.3, 0.5} | `configs/search_eval.yaml` runs `lm` (λ = 0), `area_l0.1`, `area_l0.3`, `area_l0.6` | **Fixed:** {0, 0.1, 0.3, 0.6}. |
| `search.tex:36-37` | TODO: report β and the progress threshold | `AreaPrior` defaults and the `full` condition: β (`escape_strength`) = 2.0, threshold 1.0 goals closed over the last w = 8 expansions, which the text's "closed no goals" already says | **Fixed:** adds β = 2; the TODO keeps only how λ was chosen. |
| `search.tex:65-66` | search error bars are "95% bootstrap intervals over \resSearchSeeds{} seeds" | `search_report.py` bootstraps over problems; `search_eval.yaml` has `seeds: [0]`, so there is one seed | **Fixed:** "95% bootstrap intervals over problems". |
| `appendix.tex:8` | `configs/search.yaml` holds "the prover and the search budget" | the budget for the reported runs is in `configs/search_eval.yaml` (it reads `search.yaml` for the prover and Lean settings) | **Fixed:** names both files. |
| `appendix.tex:11` | TODO: instance type | `search_eval.yaml` cost block: g6.xlarge (us-east-1) | **Fixed:** "an AWS g6.xlarge instance". |
| `appendix.tex:12` | "a spending cap of \$20 per submission" | `configs/judge.yaml` `budget.max_dollars: 85.0` ("All 8,423 pilot proofs estimate about $78") | **Left:** the author's $20 rule and the config's $85 cap disagree, so which one the real submit runs under is his call. |
| `abstract.tex:12` | "DeepSeek-Prover-V1.5" | `search.yaml` and `search.tex:9`: DeepSeek-Prover-V1.5-RL | **Fixed:** "DeepSeek-Prover-V1.5-RL". |

## Method details

| where | the draft said | the repo or data says | status |
|---|---|---|---|
| `method.tex:47-49` | the lag is "the smallest lag at which [the slow timescales] stop changing" | `configs/full.yaml` fixes `lag: 1` ("the lag the pilot's timescale check picked"); on the real fit the implied-timescale convergence check fails (max change 0.56 from lag 1 to 2) | **Fixed:** the pilot picked the lag with this check, the full fit fixes it at $\lag = $ \resLag, and the paper reports whether the $t_i$ have stopped changing there. Whether they have is a result for the author's text. |
| `method.tex:29-30` | keep the largest strongly connected set | `MSMConfig.disconnected = "nearest"`: steps in dropped microstates get the macrostate of the nearest kept centroid | **Fixed:** adds that clause. |
| `method.tex:61-62`, `experiments.tex:15`, `experiments.tex:71`, `related.tex:54` | each judge's accuracy on statements whose MSC code is known; Lean Workbook "contributes goal states for judge validation" | since PR 41, `gold-set` builds the known-code set from miniF2F, PutnamBench and ProofNet metadata (strict, family and excluded rows) and `judge-accuracy` scores a judge on it; Workbook is not part of it | **Fixed:** method names the gold set (\resGoldProblems{} problems) and its three sources; Workbook "contributes no transitions" and is a projection row in Table 1; the experiments TODO carries the new red `\resGoldAcc...` macros, which `markov-lean paper --accuracy` fills from a `judge-accuracy` run. `related.tex:54` already matches. |
| `abstract.tex:8-11`, `intro.tex:30`, `method.tex:58` | the comparison uses "two LLM judges" | still true for the reported comparison; a third local judge (`local-judge`, Qwen 2.5 7B through Ollama) exists for agreement checks | **No change needed:** true as written. |

## Claims the data does not support yet

| where | the draft said | the repo or data says | status |
|---|---|---|---|
| `abstract.tex:10-11` | "the slowest area-switching process relaxes over about \resSlowestTimescale{} tactic steps" | real fit: slowest timescale 79 steps, longer than almost every proof, and one macrostate holds 85.7% of steps | **Fixed:** "the model's slowest process", which claims nothing about areas. |
| `discussion.tex:6-11` | two outcomes framed as conditional (convergence passes, or timescales keep growing) | real fit: Chapman-Kolmogorov passes (max error 0.015) but timescale convergence, spectral gap (1.05) and PCCA+ crispness (0.55) fail | **Left:** true as a conditional, and its TODO asks the author to lead with the result; the numbers may move with the giant-state fix. |
| `discussion.tex:24` | the filter to long proofs "over-represents some areas" | no folder or area breakdown before and after the filter is in the repo yet | **Fixed:** "may over-represent". |
| `experiments.tex` (new `sec:switches`) | TODO only | `results/real-switches/train_main`: macrostates only, 6.5% [5.9, 7.0] of proofs switch; rate 0.012 per step against 0.022 under shuffle (p = 0.002); 11.4% under the Markov null; 56.5% under the folder pool | **Left:** the author's paragraph; the numbers are in the `\resSwitch...` macros. |

## Outside the draft

- `configs/full.yaml` says the pilot picked 200 microstates; the real fit picked 100. A comment only, left alone
  (configs are out of scope here).
- `scripts/paper_assets.py` called Lean Workbook "judge validation"; it now says projection, matching Table 1.
  `docs/data_card.md` says the same and is left alone here, because open PR 44 edits the next line.

## Checked and consistent

TF-IDF over identifiers with prefixes and split parts, 128 SVD dimensions and the dropped single-character
and numeric tokens (`featurize.py`); n ∈ {100, 200, 400} by cross-validated VAMP-2; reversible MLE with
deeptime; m = 10 fixed and the sweep {2, …, 12}; bootstrap error bars over proofs with a fixed seed; the
8-step filter; NMI and ARI, AMI in the robustness table; the folder baseline at the top-level Mathlib folder;
gate NMI 0.15; 16 tactic samples per expansion; λ = 0.3, α = 0.5, w = 8 in the `full` condition; closing
candidates scoring 1; miniF2F test (244); the case-study thresholds (0.7 confidence, 80% agreement, 4 to 14
steps).
