# markov-lean

Do Lean proofs move between areas of mathematics in a regular way, and can a prover use that to decide where to branch?

This repo treats each proof as a trajectory of goal states, clusters the states into microstates, builds a Markov state model over them, and coarse-grains it into metastable "areas" with PCCA+. The areas are checked against MSC 2020 top-level codes assigned by two LLM judges. The second half of the project feeds the area model into a prover's search as a prior.

## Layout

| path | what it holds |
|---|---|
| `src/markov_lean/` | Python package: data loaders, featurizer, Markov model, judges, evaluation |
| `scripts/` | setup and download scripts for the Mac Mini and EC2 |
| `lean/` | Lean project pinned to a Mathlib release (search phase) |
| `configs/` | run configs |
| `docs/schema.md` | the tables every module reads and writes |
| `docs/data_card.md` | where each dataset comes from and its license |

## Where things run

- **Mac Mini (M2, 16 GB):** downloads, trajectory extraction, the Markov model pilot, judge runs.
- **EC2 g6.xlarge (L4, 24 GB):** prover runs for the search experiments.
- **Cloud Claude sessions:** code, tests and analysis of the Parquet outputs pushed back to the repo. These sessions can't download from Hugging Face, Zenodo or the Lean release hosts.

Keys stay in environment variables on the machine that uses them (`ANTHROPIC_API_KEY`, AWS CLI profile). Never commit them.

## Quick start (Mac Mini)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,judge]"
pytest
```
