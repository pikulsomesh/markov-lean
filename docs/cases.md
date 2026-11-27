# Cross-area case studies

PFR showed that a famous cross-area proof mixes more areas than a typical Mathlib proof
but orders them the same way (judge labels: 0.242 switches per move against a shuffle of
0.352, ratio 0.69; the Mathlib pilot is 0.115 against 0.162, ratio 0.71). This page extends
that to other proofs that are publicly described as crossing fields, with within-area
controls, in two stages:

1. **Source proxy, anywhere, seconds** (`python -m markov_lean.source_cases`): reads the
   `.lean` files, splits tactic proofs into steps by layout, and labels each step by the
   Mathlib folders of the lemmas it cites, with the same resolver as the LeanDojo premise
   labels. It also runs on all of Mathlib with the same code, so comparisons are like for like.
2. **Exact extraction, Mac** (`scripts/case_extract.sh configs/cases/<name>.yaml`): the PFR
   route for any project. It builds the project from its own Mathlib cache and extracts steps,
   goal states and elaborated premises with `scripts/pfr/ExtractSteps.lean`. Its
   `steps.parquet` then goes to the judges (after Dec 1) and to `markov-lean project` for the
   label-free macrostates, exactly as PFR did.

## Candidates

| project | config | Lean | complete? | areas crossed (published) |
|---|---|---|---|---|
| Polynomial Freiman-Ruzsa (teorth/pfr) | `pfr` | 4.37-rc1 | yes | entropy and probability for additive combinatorics ([Tao's blog](https://terrytao.wordpress.com/2023/11/13/on-a-conjecture-of-marton/), [arXiv 2311.05762](https://arxiv.org/abs/2311.05762), [Quanta](https://www.quantamagazine.org/a-team-of-math-proves-a-critical-link-between-addition-and-sets-20231206/)) |
| Sphere packing in dimension 8, human-led (thefundamentaltheor3m) | `sphere_packing` | 4.37-rc1 | no, ~60 sorried proofs (dropped) | modular forms, Fourier analysis and the Cohn-Elkies LP bound for discrete geometry ([arXiv 2604.23468](https://arxiv.org/abs/2604.23468), [Cohn's survey](https://arxiv.org/abs/1611.01685), [Quanta](https://www.quantamagazine.org/sphere-packing-solved-in-higher-dimensions-20160330)) |
| Sphere packing, Math Inc completion (largely AI-written) | `sphere_packing_gauss` | 4.28 | yes (dims 8 and 24) | same mathematics; a human-versus-AI style contrast |
| Prime Number Theorem and more (PNT+, Kontorovich and Tao) | `pnt` | 4.36.0 | Weak and Medium PNT done; IEANTN folder excluded | complex analysis (residues, Perron, Mellin) and Fourier analysis for primes ([blueprint](https://alexkontorovich.github.io/PrimeNumberTheoremAnd/web)) |
| Kelley-Meka bounds (LeanAPAP) | `apap` | 4.37-rc1 | finite-field case only | discrete Fourier analysis and almost-periodicity for 3-AP-free sets ([Quanta](https://quantamagazine.org/surprise-computer-science-proof-stuns-mathematicians-20230321), [Bloom-Sisask](https://arxiv.org/abs/2302.07211)) |
| Sphere eversion | `sphere_eversion` | 4.35 | yes | differential topology by convex integration, with analysis and linear algebra ([arXiv 2210.07746](https://arxiv.org/abs/2210.07746)) |
| Carleson's theorem (control) | `carleson` | 4.36-rc1 | yes | one field: harmonic analysis on doubling metric measure spaces ([arXiv 2609.31334](https://arxiv.org/abs/2609.31334)) |
| FLT for regular primes (control) | `flt_regular` | 4.37-rc1 | yes | one field: algebraic number theory ([arXiv 2410.01466](https://arxiv.org/abs/2410.01466)) |

Not usable now: the exponential Ramsey bound, the cap set bound, unit fractions and the
Liquid Tensor Experiment are Lean 3 only; Imperial's FLT project is unfinished. Each config's
module table (which project file counts as which MSC area) is a judgement call, like
`configs/pfr_msc.yaml`; it affects only `premise_own` and `file_area`.

## Source proxy results (Nov 27)

`premise` labels, Mathlib premises only, proofs with 2+ labelled steps; shuffle and Markov
nulls with 1,000 draws (p floor 0.002).

| case | proofs | switches/move | shuffle | ratio | has switch |
|---|---|---|---|---|---|
| Mathlib (c47f9ad9) | 33,377 | 0.400 | 0.408 | 0.98 | 0.54 |
| PFR | 356 | 0.474 | 0.486 | 0.98 | 0.72 |
| Sphere packing (human) | 260 | 0.608 | 0.630 | 0.97 | 0.79 |
| Sphere packing (Math Inc) | 506 | 0.574 | 0.605 | 0.95 | 0.83 |
| PNT+ | 966 | 0.535 | 0.555 | 0.96 | 0.77 |
| LeanAPAP | 77 | 0.607 | 0.602 | 1.01 | 0.71 |
| Sphere eversion | 195 | 0.426 | 0.449 | 0.95 | 0.60 |
| Carleson (control) | 955 | 0.493 | 0.528 | 0.94 | 0.72 |
| FLT regular (control) | 98 | 0.543 | 0.560 | 0.97 | 0.79 |

Calibration: the proxy gives PFR 0.474 (0.492 with PFR's own lemmas) where the exact
extraction gave 0.446 (0.495), and Mathlib 0.400 where LeanDojo's traced tactics gave 0.392.

Reading: every research project mixes premise areas more than Mathlib does, at every proof
length (`switch_by_length.pdf`), and every one is only slightly more ordered than its own
shuffle. The cross-area projects (sphere packing, PNT+) mix the most, but the within-area
Carleson control also sits well above Mathlib, so premise labels alone do not separate
"cross-area mathematics" from "long research proofs that cite many Mathlib folders". The
judge labels, which read what a step is about, are the test that can.

## Mac commands

Each run needs about 12 GB while it builds and gives it back at the end (stage 6), so run the
projects one after another. Each waits for the overnight search's lock and yields to runner
jobs.

```
cd ~/markov-lean && git pull
bash scripts/case_extract.sh configs/cases/sphere_packing.yaml && \
bash scripts/case_extract.sh configs/cases/pnt.yaml && \
bash scripts/case_extract.sh configs/cases/carleson.yaml
bash scripts/case_extract.sh configs/cases/pnt.yaml --status
```

Rough judge cost after Dec 1, scaled from PFR ($8.24 for 861 proofs): sphere packing about
$4, PNT+ about $17, Carleson about $20. Submit each with `--max-dollars`:

```
.venv/bin/python -m markov_lean.judge submit --steps results/cases/pnt/steps.parquet --out results/cases/pnt/api --max-dollars 20
```

Rerun the source proxy (here or on the Mac; needs a mathlib4 checkout and the projects):

```
python -m markov_lean.source_cases run --mathlib ~/mathlib4 \
  --case configs/cases/pfr.yaml=~/pfr --case configs/cases/pnt.yaml=~/PrimeNumberTheoremAnd --out results/cases/source
```
