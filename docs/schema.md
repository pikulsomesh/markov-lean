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

## `search_trace` (one row per tactic tried during proof search)

Written by `markov_lean.search` as `trace.parquet` (and `trace.jsonl` while running). The first ten columns are the `steps` columns, so rows with `ok` true can be read as steps; the proofs found are also written on their own as `steps.parquet`.

| column | type | notes |
|---|---|---|
| `proof_id` … `true_area` | | as in `steps`; `step` is the depth of the state the tactic ran on, `state_after` is empty when the tactic failed, `true_area` is null |
| `node_id` | int | search node the tactic was run on (`0` is the theorem's root) |
| `parent_id` | int or null | that node's parent |
| `child_id` | int or null | node created for the resulting state; null when it failed, repeated a state or was pruned |
| `ok` | bool | Lean accepted the tactic |
| `error` | str or null | Lean's error, `timeout`, `forbidden tactic`, `duplicate state` or `zero prior weight` |
| `solved` | bool | the tactic closed the proof |
| `lm_logprob` | float | prover log-probability of the tactic text (duplicates merged by log-sum-exp) |
| `n_tokens` | int | tokens counted in `lm_logprob` |
| `count` | int | how many of the `n` samples gave this tactic |
| `prior_weight` | float or null | weight the prior gave it; null when the prior never saw it |
| `log_policy` | float or null | log of the normalized prior weight |
| `path_score` | float or null | cumulative `log_policy` from the root; the search pops the highest first |
| `elapsed` | float | seconds Lean took on the tactic |

## `search_eval` summary (one row per problem, condition and seed)

Written by `markov-lean search-eval` as `<out>/<benchmark>/<condition>/seed<k>/summary.jsonl`, next to that run's `trace.jsonl` (the `search_trace` table above). A problem is done once its line exists; a resumed run may repeat a line, and readers keep the last.

| column | type | notes |
|---|---|---|
| `proof_id`, `theorem`, `source`, `split` | str | as in `steps` |
| `solved` | bool | |
| `stop_reason` | str | `solved`, `budget`, `wall_time`, `exhausted`, or `statement error: ...` (counted as unsolved) |
| `n_tactics` | int | length of the proof found, 0 if none |
| `expansions`, `nodes` | int | prover calls made, search nodes created |
| `elapsed` | float | seconds for the problem |
| `proof` | str | the proof as Lean source, empty if unsolved |
| `benchmark`, `condition` | str | names from `configs/search_eval.yaml` |
| `seed` | int | sampling seed sent to vLLM |
| `order` | int | position in the fixed problem order (`subset_seed`) |
| `max_expansions`, `wall_time` | int, float | the budget the problem ran under |
| `root_area` | int or null | most likely Markov-model area of the first goal state, when `msm_dir` loads |
| `name_area` | str or null | miniF2F problem group from the name (`algebra`, `number_theory`, `amc`, `aime`, `imo`, `induction`) |
| `worker`, `finished_at` | int, str | worker process and UTC time |
