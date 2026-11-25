# Gate 1 robustness

`markov-lean robustness` reads a finished pilot folder and checks whether the gate 1 result holds up. It makes no API calls and launches nothing; it only reads what `markov-lean pilot` saved.

```bash
# after the judges' labels are in results/pilot3/judge/labels.parquet
markov-lean robustness --pilot results/pilot3 --jobs 6          # full grid
markov-lean robustness --pilot results/pilot3 --jobs 6 --quick  # small grid, a first look
```

It reads `msm/assignments.parquet`, `msm/msm_report.json`, `msm/centers.npy`, `judge/labels.parquet` and `judge/sample_steps.parquet` (pass `--steps` or `--labels` to use other files). Everything goes to `<pilot>/robustness/` unless `--out` says otherwise. The exit code is 0 when the macrostates clear every robustness test below and 1 otherwise, like `markov-lean pilot`.

## What it computes

All scores use the judges' consensus label of each step, filled from the whole-proof label where a step has none, exactly as the pilot's gate does, and every grouping is scored on the same steps.

| output | what it answers |
|---|---|
| `baselines.csv` | NMI, AMI and ARI of the macrostates and of each grouping with no dynamics: Mathlib folder (1 and 2 levels), theorem namespace (1 and 2 levels), k-means on the same goal-state features, k-means on the same microstate centers, the judges' own whole-proof label on every step, and a random partition. 95% bootstrap intervals over proofs. AMI corrects for NMI rising with the number of groups. |
| `robustness.json` → `baselines.bootstrap.paired` | Paired bootstrap over proofs of the macrostates' lead over each grouping, with a one-sided p-value. |
| `robustness.json` → `baselines.null` | Permutation nulls: macrostate sequences shuffled over all steps, and whole proofs' sequences swapped between proofs of the same length (keeps within-proof runs, the honest null here). |
| `baselines_per_judge.csv` | The same scores against each judge alone, so the result is not an artifact of keeping only agreed labels. |
| `robustness.json` → `judge_ceiling` | Judge-vs-judge NMI, ARI and kappa on steps and whole proofs: the most a partition can expect. |
| `robustness.json` → `label_dynamics` | How often the judges' area changes inside a proof. |
| `kmeans_by_k.csv` | k-means on the features (no dynamics) at every k. |
| `sweep.csv` | One row per featurizer × microstates × lag × macrostates: NMI, AMI, ARI, and the model checks (Chapman-Kolmogorov error over 2 and 4 lag multiples, implied-timescale change to the next lag, spectral gap, PCCA+ crispness, timescales resolved), each with pass/fail at the pilot's thresholds. |
| `memory.csv` | Held-out log-likelihood per step of order 0 to 3 Markov chains over the judges' areas, the macrostates and the microstates (5-fold over proofs; each order's smoothing tuned on training proofs only; all orders scored on the same steps). A positive gain of order 2 over order 1 means memory at lag 1. |
| `robustness.md` | The verdict and every table above in prose. |
| `figures/*.pdf` | `baselines_nmi`, `baselines_ari`, `nmi_by_k`, `ari_by_k`, `sweep_nmi`, `sweep_ck_error`, `ck_by_lag`, `memory_gain`, in the paper style of `figures.py`. |

The sweep refits the model for every cell; its cell at the pilot's own setting reproduces the saved pilot model exactly (the report prints both numbers).

## Notes on the statistics

- Bootstrap intervals resample proofs, not steps, because steps of one proof are correlated. Duplicated proofs push plug-in NMI up for groupings with many small groups, so each interval is the bootstrap spread placed around the full-data estimate; `nmi_bootstrap_bias` reports the shift.
- The gate is called at the configured k = 10. The sweep over k is reported next to it, not used to pick k.
- The default grid (4 featurizers × 4 microstate counts × lags 1, 2, 3, 5 × k 2–20) is 1,216 settings. `--quick` runs 2 × 2 × 3 × 8.
- `--jobs` runs sweep cells in separate processes, one BLAS thread each. On the 16 GB Mac Mini, 6 jobs leave headroom.
