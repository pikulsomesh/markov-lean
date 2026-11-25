# Lean project

Used only for the search phase (running a prover against benchmarks). The Markov model pilot reads the pre-traced LeanDojo data and needs no Lean install.

Run `../scripts/setup_lean.sh` once on the Mac Mini or EC2. It pins Mathlib to the newest stable release, fetches the prebuilt cache and builds `MarkovLean.lean` as a smoke test.
