# The giant macrostate and the jump-chain option

On the real LeanDojo train proofs (8+ steps), `configs/full.yaml` puts 85.7% of steps in one macrostate. This page says why and what the options added for it do. The full diagnosis, every candidate fit and the figures are in the project's results folder (`results/giant-state/README.md`); no judge labels were used, and the selection rule was written before the comparison.

## Why

- The nine small macrostates are 18 of the 100 microstates, and each of the slowest eigenvectors sits on a single microstate. Those microstates are slow because a few long proofs dwell in them (self-transition 0.95 on average against 0.87, as few as 7 effective proofs). PCCA+ gives each its own macrostate and pools the rest.
- Goal-state text varies far more between proofs than within one: 43% of proofs never leave their k-means microstate, and hypotheses are 87% of a state's characters.
- The spectrum has no gap (0.987 down to 0.959), so 10 PCCA+ sets have nothing to anchor them.
- Microstates carry the area signal (folder AMI 0.37–0.40); the coarse-graining loses it (0.10–0.18).

## Options

| option | values | what it does |
|---|---|---|
| `featurizer_kwargs.view` | `full` (default), `typed`, `goal` | which part of the state is tokenized: everything; hypothesis types and goals without binder names or `case` tags; or goals only (`featurize.state_view`) |
| `MSMConfig.kmeans_weight` | `step` (default), `proof` | `proof` weighs each step by 1 / proof length, so each proof counts once in k-means |
| `MSMConfig.jump_chain` | `false` (default), `true` | collapse runs of one microstate within a proof before counting, so only changes of microstate are transitions; lag and timescales are then in microstate jumps (`report["time_unit"]`) |

Defaults are unchanged, so the pilot and `configs/full.yaml` give the same results as before. `configs/full_jump.yaml` is `configs/full.yaml` with `view: typed`, `jump_chain: true` and 100 microstates fixed:

```
markov-lean full --config configs/full_jump.yaml --out results/full_jump
```

The projection (`markov-lean project`, switch rates, dwell times) still works per proof step; only the MSM's counting changes. With lag 1 both readings use consecutive steps.

## What it gives on train (100 microstates, 3 k-means seeds)

- Largest macrostate 63%, 27% and 52% (default 86%, 86%, 94%); effective number of macrostates 4.7 (default 1.7).
- Step AMI with the Mathlib folder 0.29 and with the theorem namespace 0.28 (default 0.14 and 0.13).
- 25% of proofs visit two or more macrostates (default 4.6%).

Limits: the partition still changes with the k-means seed (mean pairwise ARI 0.32), the spectral gap check still fails (1.03), and at seed 0 the largest state is 62% of val steps and 69% of test steps. At 400 microstates, which cross-validated VAMP-2 prefers on the jump chain, the largest state is back to 64%, so the config fixes 100. View and weighting alone do not remove the giant state; the jump chain does.
