# Shared data schema

Every module reads and writes these tables as Parquet. Change a column only in a PR that updates this file.

## `steps` (one row per tactic step)

| column | type | notes |
|---|---|---|
| `proof_id` | str | stable id: `<source>:<full theorem name>` |
| `step` | int | 0-based position of the tactic in the proof |
| `source` | str | `leandojo`, `workbook`, `minif2f`, `proofnet`, `putnam`, `synthetic` |
| `split` | str | `train`, `val`, `test` (from the source's own split when it has one) |
| `theorem` | str | full Lean name |
| `file_path` | str | Mathlib path, empty when unknown |
| `tactic` | str | tactic text as written |
| `state_before` | str | pretty-printed goal state before the tactic |
| `state_after` | str | goal state after (`no goals` when closed) |
| `true_area` | str or null | synthetic data only: the generating MSC code |

## `labels` (one row per judged step or proof)

| column | type | notes |
|---|---|---|
| `proof_id` | str | |
| `step` | int | `-1` for a whole-proof label |
| `judge` | str | model id, e.g. `claude-haiku-5-5` |
| `msc` | str | two-digit MSC 2020 top-level code, e.g. `26` |
| `confidence` | float | 0 to 1, as the judge reports it |

## `assignments` (output of the Markov model pipeline)

| column | type | notes |
|---|---|---|
| `proof_id` | str | |
| `step` | int | |
| `microstate` | int | k-means cluster |
| `macrostate` | int | PCCA+ metastable set |
| `membership` | float | PCCA+ membership of the assigned macrostate |
