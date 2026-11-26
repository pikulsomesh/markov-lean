# PFR case study

Does a famous cross-area proof look different from routine Mathlib proofs? The Lean 4
proof of the polynomial Freiman-Ruzsa conjecture (teorth/pfr) proves additive-combinatorics
statements with Shannon entropy. We extract its tactic steps the way LeanDojo does and run
the same judges and switch statistics as for LeanDojo.

## Run (Mac Mini)

```
bash scripts/pfr_extract.sh
```

It clones teorth/pfr at commit `1899ce8` (Lean v4.37.0-rc1, its own Mathlib pin), fetches
Mathlib's prebuilt cache, builds PFR's 71 files, extracts steps with
`scripts/pfr/ExtractSteps.lean`, writes free labels and prints the judge cost estimate.
No API key is used. The build and extraction run under the memory guard with its shared
lock, so they wait for the overnight search to finish and yield to runner jobs. Rerun to
resume; `--status` shows progress.

Disk: it needs about 12 GB while it runs (the Lean 4.37 toolchain, about 3.3 GB if new;
PFR's own Mathlib checkout and prebuilt files, about 6 to 7 GB; their compressed cache,
about 2 GB) and stops before downloading anything unless 15 GB are free (`MIN_DISK_GB`).
Once the steps are extracted and converted, stage 6 deletes PFR's build folder, the Mathlib
cache files it downloaded (files that were already there, such as markov-lean's own, stay)
and the toolchain if it installed it. `KEEP_PFR_BUILD=1` keeps them.

Then, after checking the console balance:

```
.venv/bin/python -m markov_lean.judge submit --steps results/pfr/pfr_steps.parquet --out results/pfr/api --max-dollars 10
```

The cap makes the submit refuse if the estimate is over $10; add `--sample N` to judge a
random N proofs instead.

## Outputs (`results/pfr/`)

`pfr_steps.parquet` (the `steps` table, `source = "pfr"`), `premises.parquet`,
`free_labels.parquet` and `summary.json`; see `src/markov_lean/pfr.py`.

Free labels, no API spend:

- `file_area`, `chapter_area`: where the proof lives (its file, by `configs/pfr_msc.yaml`,
  or its blueprint chapter). One area per proof, used as the home area.
- `premise_area`: majority Mathlib folder area of the step's premises, as
  `markov-lean premise-labels`.
- `premise_pfr_area`: the same, also counting PFR's own lemmas by their file's area
  (entropy files 94, sumset and coset files 11, probability 60).

## Extraction rules

As LeanDojo's ExtractData.lean: a step is a tactic whose parent is a tactic sequence or an
`rw` rule list, kept unless its state before is `no goals` or the state is unchanged;
focusing dots are dropped. Premises are identifiers in the step that elaborate to a
constant, with the module defining it. The Mathlib commit differs from LeanDojo Benchmark 4's,
so tactic names and goal printing can differ slightly.

## Caveats

One project (n = 1). The blueprint factors the proof into lemmas the way Mathlib does, so
cross-area jumps may hide inside lemma statements. Entropic Ruzsa distance sits on the
94/11 boundary, so the file table is a judgement call.
