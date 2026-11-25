# Reversible vs non-reversible Markov models

A Lean proof runs one way, from the theorem statement to `no goals`. The main pipeline fits a reversible (detailed-balance) MSM and coarse-grains it with PCCA+, which needs reversibility. `markov-lean nonrev-check` fits three models on the same microstates and lag and compares them, so the paper can show what the reversibility assumption does and does not change. It makes no API calls.

```bash
# on a saved fit (pilot or full): reuses its microstates, lag and number of macrostates
markov-lean nonrev-check --msm results/full/msm --out results/full/nonrev
# with judge labels, scored per fit
markov-lean nonrev-check --msm results/pilot/msm --labels results/pilot/judge/labels.parquet --out results/pilot/nonrev
# fit the usual pipeline on a steps table first (MSMConfig fields from the YAML's `msm:` section)
markov-lean nonrev-check --steps data/processed/leandojo_train.parquet --config configs/full.yaml --out results/nonrev
# synthetic demo: a one-way area chain, or a symmetric one
markov-lean nonrev-check --synthetic directed --out results/nonrev_synthetic
```

## The three fits

| fit | estimator | coarse-graining | proof ends |
|---|---|---|---|
| `reversible` | deeptime `MaximumLikelihoodMSM(reversible=True)` on the largest strongly connected set | PCCA+ | censored (nothing counted across proofs) |
| `nonreversible` | `MaximumLikelihoodMSM(reversible=False)` (row-normalized counts) on the same set | GPCCA | censored |
| `absorbing` | row-normalized counts over every visited microstate plus a `done` state | GPCCA with one extra set | every window past the last step lands in `done`, which is absorbing |

**GPCCA** is built in (`nonrev.gpcca`, numpy and scipy only), following Reuter et al. (2018, 2019): an ordered real Schur form of `D^1/2 P D^-1/2` gives a basis of the dominant invariant subspace, re-based so its first vector is constant and it is orthonormal in the `eta`-weighted inner product; the PCCA+ inner simplex algorithm and the Roeblitz-Weber crispness optimization then give memberships. A cut that would split a complex-conjugate pair moves up by one set. `eta` is the share of steps in each state (`--eta uniform` is the alternative). On a reversible matrix with `eta` its stationary distribution, GPCCA gives the same memberships as deeptime's PCCA+ (tested). pyGPCCA is the reference implementation if you want a cross-check; it is not a dependency.

**Why not restart trajectories.** Joining each proof's end to another proof's start would make the chain ergodic by construction and invent end-to-start transitions whose statistics depend on how proofs are ordered. Censoring keeps every counted transition real; the absorbing fit models the end explicitly.

## What it reports

| output | contents |
|---|---|
| `nonrev_report.json` → `fits.<fit>` | timescales (`-lag / ln|lambda|`, moduli for complex eigenvalues), how many slow eigenvalues are complex, macrostate shares (`largest_macrostate_share`), the coarse matrix, and the checks: spectral gap, crispness, Chapman-Kolmogorov, timescale convergence |
| `agreement` | step-level NMI/ARI between fits, plus GPCCA vs PCCA+ on the reversible matrix (separates the method from the assumption) |
| `vs_labels` | NMI/ARI/AMI of each fit against the labels, when given |
| `irreversibility.macrostate_<fit>` | detailed balance on the macrostate chain: likelihood ratio of the non-reversible vs reversible MLE (chi-square reference), entropy production, held-out log-likelihood per transition (3 folds over proofs), and net flux `(n_AB - n_BA) / (n_AB + n_BA)` |
| `irreversibility.microstate`, `cv_loglik_per_transition_microstates` | the same on microstates; counts are sparse there, so sampling noise alone makes them look irreversible. Use the macrostate numbers for claims |
| `absorbing_view.<fit>` | on the lag-1 absorbing chain: probability of reaching each other macrostate before `no goals`, of finishing without leaving the current one, of a proof switching at least once, and expected steps left; each next to its empirical value |
| `checks.absorbing_markov_<fit>` | max model-vs-empirical gap of those probabilities (threshold 0.10) |
| `implied_timescales.csv`, `ck_test.csv`, `occupancy.csv`, `assignments.parquet`, `coarse_T_<fit>.npy` | the tables behind the report |
| `nonrev_comparison.pdf` | timescales by lag, CK predicted vs estimated, macrostate shares, reversible vs non-reversible contingency |
| `nonrev_absorbing.pdf` | model and empirical probability of reaching each macrostate before `no goals` |
| `SUMMARY.md` | the numbers above in a few lines |

The absorbing view assumes the chance of finishing depends only on the current microstate. Where proof length is set some other way (the synthetic generator draws it uniformly), the model and empirical switch probabilities disagree and `absorbing_markov_*` fails; that is the check doing its job.

## On synthetic proofs

`nonrev.generate_directed` writes proofs over a cyclic six-area chain (stay 0.8; otherwise step forward with probability `forward`, back otherwise), with the same goal-state text model as `synthetic.generate`. With 200 proofs, 40 microstates, lag 1, over seeds 0 to 2:

| | one-way (`forward=1`) | symmetric (`forward=0.5`) |
|---|---|---|
| detailed balance on macrostates (LR p) | about 1e-171 | 0.2 to 0.99 |
| held-out log-lik, non-rev minus rev (nats/transition) | +0.13 | -0.003 to 0.000 |
| CK max error, reversible / non-reversible | 0.05 to 0.06 / 0.02 to 0.03 | 0.02 to 0.04 / 0.02 to 0.04 |
| ARI reversible vs non-reversible partition | 1.00 | 1.00 |
| NMI with the true areas (non-reversible) | 0.98 | 0.98 |

On one-way proofs the reversible fit finds the same areas but symmetrizes the kinetics: its coarse matrix puts about 0.1 on each neighbor in both directions, where the truth (and the non-reversible fit) puts 0.2 forward and none back.

## Methods note (to adapt)

> Because proofs are one-way trajectories that start at a theorem statement and end at *no goals*, the reversible MSM is an assumption, not a property of the data. We therefore also fit a non-reversible maximum-likelihood MSM on the same microstates and lag and coarse-grain it with GPCCA (Reuter et al., 2018), which generalizes PCCA+ to non-reversible chains by replacing eigenvectors with a Schur basis of the dominant invariant subspace. Proof ends are treated as censoring; in a third model they are an absorbing *done* state, which keeps theorem-statement states that the strongly connected set would drop and lets us ask forward questions directly: the probability that a proof reaches another macrostate before it is finished. We test detailed balance on the macrostate chain with a likelihood-ratio test and held-out log-likelihood (folds over proofs), and compare the fits on implied timescales, Chapman-Kolmogorov error and macrostate assignments (NMI/ARI). On synthetic proofs with one-way area changes, the test rejects detailed balance and only the non-reversible model recovers the direction of flow; on symmetric ones it does not, and the two fits agree.
