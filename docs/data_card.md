# Data card

What `scripts/fetch_data.sh` downloads, what we use from each source, and the known gaps. Raw files go to `data/raw/` and the converted tables to `data/processed/<source>_<split>.parquet`; neither is committed.

Licenses below come from each source's own page as last checked; recheck before redistributing anything.

| source | where it comes from | size | what we use | rows in `steps` |
|---|---|---|---|---|
| LeanDojo Benchmark 4 | Zenodo record 12740403 (v10), `leandojo_benchmark_4.tar.gz`; URL and MD5 from ReProver's `scripts/download_data.py` | about 68 MB compressed | `random/{train,val,test}.json`: per theorem `full_name`, `file_path` and the `traced_tactics` list (`tactic`, `state_before`, `state_after`) | one per traced tactic |
| Lean Workbook | Hugging Face `internlm/Lean-Workbook`, file `lean_workbook.json` (the repo also holds `wkbk_1009.parquet`, not used) | about 95 MB | `id`, `status`, `tactic`, `state_before`, `state_after`; `formal_statement` kept for later | one per problem with a tactic |
| miniF2F (Lean 4) | GitHub `yangky11/miniF2F-lean4` (its Reservoir page says it isn't maintained regularly) | small | theorem statements from `MiniF2F/Test` and `MiniF2F/Valid` | one per theorem, `step = -1` |
| ProofNet (Lean 4) | GitHub `rahul3613/ProofNet-lean4`, a port of `zhangir-azerbayev/ProofNet` | small | theorem statements | one per theorem, `step = -1` |
| PutnamBench | GitHub `trishullab/PutnamBench`, Lean 4 files under `lean4/` (672 Lean 4 problems per its site) | small | theorem statements | one per theorem, `step = -1` |

## Field notes

- **LeanDojo step numbers.** LeanDojo's export drops tactics whose `state_before` is `no goals` and focusing dots (`·`), so `step` counts the kept tactics. Order within a proof is the order LeanDojo traced them, which follows the source. Field names confirmed from the export code in `lean-dojo/LeanDojo`, `scripts/generate-benchmark-lean4.ipynb`.
- **LeanDojo splits.** We read the `random` split strategy. `novel_premises` is available through `load_leandojo(..., strategy="novel_premises")`.
- **LeanDojo Mathlib version.** The benchmark was traced on one fixed Mathlib commit (the `commit` field). File paths and names can differ from current Mathlib; the Lean project in `lean/` pins a newer Mathlib for the search phase.
- **Lean Workbook.** Each row records at most one tactic, so these proofs carry no step-to-step transitions. They are useful as extra goal states for featurizing and for judge validation, not for the transition matrix. Rows whose `status` says disproved are dropped by default. The dataset viewer shows 25.2k rows; check the count printed after download.
- **Statement-only benchmarks.** miniF2F, ProofNet and PutnamBench have no proofs, so their `state_before` holds the theorem statement (from the keyword to `:=`, doc comment dropped). They are search targets, not training trajectories. Splits come from the folder or file name (`test`, `valid`).
- **Areas.** No source carries MSC codes. Labels come from the LLM judges (`labels` table), and the Mathlib folder in `file_path` is the baseline.

## Memory on the Mac Mini

`load_leandojo` reads one split file at a time; `random/train.json` is a few hundred MB unpacked. `build_all(..., max_theorems=N)` caps the theorems per split if memory is tight.
