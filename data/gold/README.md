# Gold MSC set

`msc_gold.csv` gives benchmark problems an MSC 2020 top-level class taken from the benchmark's own metadata
(miniF2F names, PutnamBench tags, ProofNet textbook and chapter), never from a model. `msc_gold_balance.json`
holds its class counts. Rules, balance and commands: `docs/judge_accuracy.md`. Columns: `docs/schema.md`.

Rebuild: `./scripts/fetch_gold_sources.sh && markov-lean gold-set build`.

The `statement` column is copied from DeepSeek-Prover-V1.5's Lean 4 miniF2F and ProofNet files (MIT,
Copyright (c) 2023 DeepSeek; ProofNet itself MIT, Copyright (c) 2022 zhangir-azerbayev) and from
PutnamBench (Apache License 2.0, trishullab). ProofNet book and chapter ids come from the original ProofNet
benchmark files.
