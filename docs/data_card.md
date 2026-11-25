# Data card

What `scripts/fetch_data.sh` downloads, what we use from each source, and the known gaps. Raw files go to `data/raw/` and the converted tables to `data/processed/<source>_<split>.parquet`; neither is committed.

Licenses below come from each source's own page as last checked; recheck before redistributing anything.

| source | where it comes from | size | what we use | rows in `steps` |
|---|---|---|---|---|
| LeanDojo Benchmark 4 | Zenodo record 12740403 (v10), `leandojo_benchmark_4.tar.gz`; URL and MD5 from ReProver's `scripts/download_data.py` | about 68 MB compressed | `random/{train,val,test}.json`: per theorem `full_name`, `file_path` and the `traced_tactics` list (`tactic`, `state_before`, `state_after`) | one per traced tactic |
| Lean Workbook (steps) | Hugging Face `internlm/Lean-Workbook`, file `wkbk_1009.parquet` | about 5 MB | `id`, `status`, `tactic`, `state_before`, `state_after` | one per problem with a tactic (`workbook_train`) |
| Lean Workbook (statements) | same dataset, file `lean_workbook.json` | about 95 MB | `formal_statement` only; this file has no tactic data | about 140k, `step = -1` (`workbook_statements_train`) |
| miniF2F (Lean 4) | `datasets/minif2f.jsonl` in GitHub `deepseek-ai/DeepSeek-Prover-V1.5` | small | `name`, `split`, `goal`, `formal_statement` | 488 (244 val, 244 test), `step = -1` |
| ProofNet (Lean 4) | `datasets/proofnet.jsonl` in the same repo (the original ProofNet repo points to this port) | small | `name`, `split`, `goal`, `formal_statement` | 371 (185 val, 186 test), `step = -1` |
| PutnamBench | GitHub `trishullab/PutnamBench`, Lean 4 files under `lean4/` (672 Lean 4 problems per its site) | small | theorem statements | one per theorem, `step = -1` |

## Field notes

- **LeanDojo step numbers.** LeanDojo's export drops tactics whose `state_before` is `no goals` and focusing dots (`·`), so `step` counts the kept tactics. Order within a proof is the order LeanDojo traced them, which follows the source. Field names confirmed from the export code in `lean-dojo/LeanDojo`, `scripts/generate-benchmark-lean4.ipynb`.
- **LeanDojo splits.** We read the `random` split strategy. `novel_premises` is available through `load_leandojo(..., strategy="novel_premises")`.
- **LeanDojo Mathlib version.** The benchmark was traced on one fixed Mathlib commit (the `commit` field). File paths and names can differ from current Mathlib; the Lean project in `lean/` pins a newer Mathlib for the search phase.
- **Lean Workbook.** `wkbk_1009.parquet` records at most one tactic per problem, so it carries no step-to-step transitions. It is useful for extra goal states (projected onto the fitted model), not for the transition matrix. Rows whose `status` says disproved are dropped by default. `lean_workbook.json` has no tactics at all (first run on the Mac Mini, 2026-11-25: 140,214 rows, `status` absent, `proof` empty), so it is loaded as statements.
- **Statement-only benchmarks.** miniF2F, ProofNet and PutnamBench have no proofs; they are search targets, not training trajectories. For miniF2F and ProofNet, `state_before` is the initial goal state from DeepSeek-Prover's files, which also give the valid/test split. We use these copies because DeepSeek-Prover-V1.5 is our search prover, so its evaluations and ours see the same statements. For PutnamBench, `state_before` is the statement text from the `.lean` file (keyword to `:=`, doc comment dropped).
- **Repeated names.** ProofNet reuses names such as `exercise_1_1` across textbooks; repeats get `#2`, `#3` suffixes in `proof_id`.
- **Areas.** No source carries MSC codes. Labels come from the LLM judges (`labels` table), and the Mathlib folder in `file_path` is the baseline.

## Memory on the Mac Mini

`load_leandojo` reads one split file at a time; `random/train.json` is a few hundred MB unpacked. `build_all(..., max_theorems=N)` caps the theorems per split if memory is tight.
