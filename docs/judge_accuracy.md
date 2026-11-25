# Judge accuracy on a gold MSC set

Judge agreement (`judge-agreement`) says whether the judges label alike, not whether they are right. This
page describes a gold set of benchmark problems whose MSC 2020 top-level class is known from the
benchmark's own metadata, and the `judge-accuracy` command that scores any judge against it.

Code: `src/markov_lean/gold.py` (the set and its rules), `src/markov_lean/judge_accuracy.py` (the report).
Data: `data/gold/msc_gold.csv` and `data/gold/msc_gold_balance.json`.

## How labels are made

**No label comes from a language model or from reading a statement.** Each label is a lookup on metadata the
benchmark authors wrote, using the tables in `gold.py`. Rebuilding from the same pinned sources gives the
same file byte for byte (`tests/test_gold.py` checks every committed row against the tables).

Each problem is one of:

- **strict**: the metadata names one MSC class. Scored by exact match; used for accuracy, macro-F1,
  balanced accuracy and the confusion matrix.
- **family**: the metadata names a group of neighbouring classes but not one (Putnam "analysis" can be 26,
  30, 40 and more). Scored only by whether the judge's code is in the group.
- **excluded**: the metadata gives no area, or one with no MSC counterpart. Never scored.

### miniF2F (by name prefix)

| prefix | label | why |
|---|---|---|
| `mathd_numbertheory_`, `numbertheory_` | strict 11 | MATH "Number Theory" subject, or the authors' number theory set |
| `mathd_algebra_`, `algebra_` | excluded | school algebra (equations, inequalities, functions) has no single MSC class |
| `induction_` | excluded | names a proof method, not an area |
| `amc12*`, `aime*`, `imo*` | excluded | the competition name gives no area |

### PutnamBench (by the `tags` in `informal/putnam.json`)

| tag | label | why |
|---|---|---|
| `number_theory` | strict 11 | |
| `combinatorics` | strict 05 | |
| `linear_algebra` | strict 15 | |
| `probability` | strict 60 | |
| `set_theory` | strict 03 | MSC puts set theory in 03E |
| `geometry` | family 51, 52 | Putnam geometry is Euclidean (51) or convex and discrete (52) |
| `analysis` | family 26 to 49 | the MSC analysis block: real functions, complex variables, series, ODEs, ... |
| `abstract_algebra` | family 08, 12, 13, 16, 17, 20 | groups, rings and fields |
| `algebra` | excluded | Putnam "algebra" spans polynomials, inequalities and sequences |

A problem with several tags is strict only when every tag is strict with the same code; it is a family (the
union) when every tag has codes; it is excluded when any tag is excluded.

### ProofNet (by textbook and chapter)

The chapter is the first number in the exercise id (for Munkres, the section).

| book | chapters | label |
|---|---|---|
| Axler, *Linear Algebra Done Right* | all | strict 15 |
| Ireland and Rosen, *A Classical Introduction to Modern Number Theory* | all | strict 11 |
| Stein and Shakarchi, *Complex Analysis* | all | strict 30 |
| Munkres, *Topology* | sections 12 to 50 (Part I) / 51 on (Part II) | strict 54 / strict 55 |
| Dummit and Foote, *Abstract Algebra* | 1 to 6 groups / 7 to 10 rings, modules / 11 vector spaces / 12 / 13 to 14 fields | strict 20 / family 13, 16 / strict 15 / family 13, 15, 16 / strict 12 |
| Herstein, *Topics in Algebra* | 2 groups / 3 rings / 4 vector spaces / 5 fields / 6 linear transformations | strict 20 / family 13, 16 / strict 15 / strict 12 / strict 15 |
| Artin, *Algebra* | 2 groups / 3 vector spaces | strict 20 / strict 15 |
| Rudin, *Principles of Mathematical Analysis* | 1 / 2 topology / 3 series / 4 continuity / 5 differentiation / 6 / 7 | family 11, 12, 26 / strict 54 / strict 40 / family 26, 54 / strict 26 / family 26, 28 / family 26, 40, 46 |
| Pugh, *Real Mathematical Analysis* | 1 / 2 topology / 3 real functions / 4 | family 26, 54 / strict 54 / strict 26 / family 26, 40, 46 |

Excluded: other Artin chapters (the 1st and 2nd editions number them differently, and ProofNet does not say
which it used), Herstein chapter 1, Munkres sections 1 to 11 (set theory), and ProofNet's Putnam problems,
which carry no topic. DeepSeek's Lean 4 copy drops the book from the exercise name and some names repeat
across books, so a repeated name is matched on the informal statement; an id with no unique match would be
excluded (none is).

## Size and balance (sources as pinned on 2026-11-25)

1,532 problems: **546 strict** in 11 classes, **346 family**, 640 excluded.

| class | n | share of strict | sources |
|---|---|---|---|
| 11 Number theory | 223 | 40.8% | miniF2F 136, Putnam 66, ProofNet 21 |
| 20 Group theory | 93 | 17.0% | ProofNet 93 |
| 15 Linear algebra | 78 | 14.3% | ProofNet 45, Putnam 33 |
| 54 General topology | 76 | 13.9% | ProofNet 76 |
| 05 Combinatorics | 20 | 3.7% | Putnam 20 |
| 30 Complex variable | 16 | 2.9% | ProofNet 16 |
| 26 Real functions | 13 | 2.4% | ProofNet 13 |
| 40 Sequences, series | 11 | 2.0% | ProofNet 11 |
| 12 Field theory | 7 | 1.3% | ProofNet 7 |
| 60 Probability | 5 | 0.9% | Putnam 5 |
| 03 Logic and foundations | 4 | 0.7% | Putnam 4 |

Family rows: Putnam analysis 176 (plus 24 multi-tag rows with analysis), Putnam geometry 40, Putnam abstract
algebra 21, Dummit and Foote rings 24, Rudin 31, and small multi-tag groups. `markov-lean gold-set summary`
prints the full table.

The classes are unbalanced (11 is 41% of strict rows), so read **macro-F1** and **balanced accuracy** next
to plain accuracy, and compare accuracy with the **majority baseline** (always answering 11 scores 0.408).
Classes with fewer than 20 problems have wide per-class intervals.

## What is scored

The benchmarks give statements, not traced proofs, so there are no per-step goal states to label. Each gold
problem goes to the judges as a one-step proof: step 0 shows the problem's initial Lean goal with tactic
`sorry`. The score uses the judge's **proof-level label** (`step -1`), the area of the statement, which is
also what the judge prompt asks for at the proof level. This measures how well a judge names a theorem's
area from its Lean statement; it is not a direct measure of step-level accuracy.

For miniF2F, DeepSeek-Prover-V2 published whole proofs of 438 of the 488 problems (131 of the strict
ones). `gold-set steps --proofs data/raw/gold/minif2f-solutions.zip` shows them as the step's tactic, for a
check of whether seeing a proof changes the label. Tracing those proofs in Lean would give real step-level
goal states; that is not done here. The proofs are not stored in the repository, since that archive states
no licence for them.

## Dev and test halves

A judge prompt designed while looking at gold problems would score well on those problems for the wrong
reason. So the gold set has a fixed dev/test split, `data/gold/msc_gold_eval_split.csv`
(`markov-lean gold-set eval-split` rewrites it, and `tests/test_prompts.py` checks it). Within each stratum
(each strict class, each family rule, the excluded rows) problems are ordered by
`sha256("20261125:" + id)`; the first half (rounded down) is dev and the rest is test. Then the 28 gold
statements read while the v2 prompt was written, before the split existed (`gold.SEEN_BEFORE_SPLIT`), are moved
to dev, so no test problem was seen during prompt design. It depends only on the ids, statuses and the seed,
not on row order. Strict rows: 285 dev, 261 test, in 10 of the 11 classes (all four 03 Logic problems were
read, so 03 has no test problem).

Prompt work uses dev only, and any prompt designed that way is scored on test:
`gold-set steps --eval-split test` writes only test problems, and `judge-accuracy --eval-split test` scores
only them (the report says which half it used). `--eval-split all`, the default, keeps the earlier
behaviour, which is right for a judge whose prompt was never tuned on the gold set, such as the API judges.

## Local judge, prompt v2 against v1, on the test half

```
cd ~/markov-lean
git pull
source .venv/bin/activate
pip install -e ".[dev,judge,search,figures]"

markov-lean gold-set steps --strict-only --eval-split test --out results/gold/steps_test.parquet
# v1 baseline on the same 261 problems (reuses what the strict25 run already labelled)
markov-lean local-judge label --steps results/gold/steps_test.parquet --all --stratify-by none \
  --out results/gold/local/qwen2.5_7b-instruct
# v2
markov-lean local-judge label --steps results/gold/steps_test.parquet --all --stratify-by none \
  --prompt v2 --out results/gold/local/qwen2.5_7b-instruct_v2
markov-lean judge-accuracy --eval-split test --out results/gold/accuracy_test \
  --labels results/gold/local/qwen2.5_7b-instruct/labels.parquet \
           results/gold/local/qwen2.5_7b-instruct_v2/labels.parquet
```

## On the Mac Mini, with the local judge (no API cost)

Run it after the current local-judge labelling finishes, so the two runs do not share the GPU.

```
cd ~/markov-lean
git pull
source .venv/bin/activate
pip install -e ".[dev,judge,search,figures]"

# 176 strict problems: at most 25 per class, fixed seed
markov-lean gold-set steps --strict-only --per-class 25 --out results/gold/steps_strict25.parquet
markov-lean local-judge label --steps results/gold/steps_strict25.parquet --all --stratify-by none \
  --out results/gold/local/qwen2.5_7b-instruct
markov-lean judge-accuracy --labels results/gold/local/qwen2.5_7b-instruct/labels.parquet --out results/gold/accuracy
```

Each problem is one short goal, so a request is mostly the shared system prompt; expect well under the
48 seconds a pilot proof takes. `label` prints the measured time per problem every 5 problems. Ctrl-C and
the same command resume as usual.

For the whole set (546 strict and 346 family problems), write `results/gold/steps_all.parquet` with
`markov-lean gold-set steps --out results/gold/steps_all.parquet` and label it into its own `--out` folder.

To score the API judges on the same problems later (this spends API credit, so check the estimate first
with `python -m markov_lean.judge estimate --steps results/gold/steps_strict25.parquet`), submit the same
steps file with `python -m markov_lean.judge submit --steps results/gold/steps_strict25.parquet --out
results/gold/api --max-dollars <cap>`, collect it with `python -m markov_lean.judge collect --out
results/gold/api`, then pass every labels file at once:

```
markov-lean judge-accuracy --labels results/gold/local/qwen2.5_7b-instruct/labels.parquet results/gold/api/labels.parquet \
  --out results/gold/accuracy
```

## The report

`judge-accuracy` prints a Markdown table and, with `--out`, writes:

| file | what it holds |
|---|---|
| `judge_accuracy.md`, `judge_accuracy.json` | per judge: coverage; on strict problems accuracy, macro-F1, balanced accuracy with 95% intervals, the majority baseline, per-class precision, recall (with interval) and F1, codes outside the gold classes, accuracy by source; family accuracy with interval and by rule |
| `confusion_<judge>.csv` | counts, gold class by judge label (`other` = any other code or an invalid answer) |
| `figures/judge_accuracy.pdf`, `.png` | left: accuracy, macro-F1 and family accuracy per judge with intervals and the majority line; right: each judge's confusion matrix, rows normalized to recall |

Intervals are 95% percentile bootstraps over problems (`--n-boot`, default 2000; `--seed`). A problem the
judge did not label is left out and shows in coverage; an answer that is not an MSC 2020 code counts as wrong.
If a judge gave only step labels, its step-0 label is scored and the report counts those problems.

## Sources and licences

| source | commit | licence | used for |
|---|---|---|---|
| deepseek-ai/DeepSeek-Prover-V1.5 `datasets/minif2f.jsonl`, `datasets/proofnet.jsonl` | `2c4ba911` | MIT | Lean 4 goals, splits, names |
| zhangir-azerbayev/ProofNet `benchmark/{valid,test}.jsonl` | `509ad797` | MIT | book and chapter of each exercise |
| trishullab/PutnamBench `informal/putnam.json`, `lean4/src/*.lean` | `a635813f` | Apache 2.0 | tags and Lean 4 statements |
| deepseek-ai/DeepSeek-Prover-V2 `minif2f-solutions.zip` | `e598a57e` | not stated | optional proofs, read locally only |

`scripts/fetch_gold_sources.sh` downloads exactly these into `data/raw/gold/`, all from GitHub;
`markov-lean gold-set build` rebuilds `data/gold/msc_gold.csv` from them.
