# A third judge on the Mac Mini (Ollama)

The two API judges, Claude Haiku and Claude Sonnet, come from one model family, so their agreement can
overstate how reproducible the MSC labels are. There is no hand-labelled gold set, so the cross-family check
is a third judge from another family: an open model served by [Ollama](https://ollama.com) on the Mac Mini.
It gets the **same system prompt, user prompt, JSON schema and 63 MSC codes** as the API judges
(`markov_lean/judge.py`), costs nothing and makes no Anthropic call.

Code: `src/markov_lean/ollama_judge.py` (the judge) and `src/markov_lean/agreement.py` (the report).

## Which model

**Default: `qwen2.5:7b-instruct`** (Alibaba's Qwen 2.5, 7B, 4-bit `Q4_K_M`, a 4.7 GB download).

- It follows a JSON schema reliably and is among the strongest 7B models on mathematics, which matters for
  reading Lean goal states.
- Its 32K-token context holds the longest pilot prompts. Pilot prompts average about 4K tokens and peak
  near 24K, so the default context of 32,768 tokens labels every proof without cutting it.
- On 16 GB: about 4.7 GB of weights plus about 1.8 GB of cache at 32K context, so about 7 GB in all. Close
  other heavy apps while it runs.

**Second choice: `llama3.1:8b-instruct-q4_K_M`** (Meta's Llama 3.1, 8B, 4.9 GB). It is a different family
from both Claude and Qwen, so running it too gives a fourth judge and shows whether one open model is an
outlier. It is weaker than Qwen at mathematics.

Bigger models (Qwen 2.5 14B is about 9 GB) fit in 16 GB only with a much smaller context and run about
half as fast, so they are not worth it here.

## Commands

Ollama 0.5 or newer is needed for schema-constrained output; older versions fall back to plain JSON mode
with repair. Keep the Ollama app running (it serves `http://localhost:11434`).

```
ollama pull qwen2.5:7b-instruct

cd ~/markov-lean
git pull
source .venv/bin/activate
pip install -e ".[dev,judge,search,figures]"

# is the server up, is the model pulled, and how long does one typical proof take
markov-lean local-judge check --run results/pilot3

# label a stratified sample of about 500 proof steps
markov-lean local-judge label --run results/pilot3 --sample-steps 500
```

`label` reads the pilot's judge sample from `results/pilot3/judge/sample_steps.parquet` and writes to
`results/pilot3/judge/local/qwen2.5_7b-instruct/`:

| file | what it holds |
|---|---|
| `labels.parquet` | the `labels` table of `docs/schema.md`, judge `ollama/qwen2.5:7b-instruct` |
| `progress.jsonl` | one line per finished proof (labels, issue, seconds, tokens); this is what makes the run resumable |
| `sample.json` | the proofs in the sample and the settings |
| `issues.json`, `summary.json` | proofs with problems, and counts, speed and token totals |

**Stopping and resuming.** Press Ctrl-C at any time, or close the laptop lid, and run the same command
again: finished proofs are skipped. The sample is taken in an interleaved order, so a run stopped halfway
is still spread over all areas. A proof where the server kept failing is redone on the next run;
`--retry-failed` also redoes proofs that ended with another issue.

**The sample.** The judges see a whole proof per request, so the sample is a set of proofs whose shown
steps add up to about `--sample-steps`. Proofs are stratified by the API judges' proof label when
`results/pilot3/judge/labels.parquet` exists, and by Mathlib folder before that. When the API labels exist,
only proofs they labelled are sampled, so every local label can be paired. If you run the local judge
before the API labels land, run the same command once more after they land: it switches to the proofs the
API judges labelled, stratified by their labels, and reuses every proof it already did.

Other options: `--all` labels every proof; `--model llama3.1:8b-instruct-q4_K_M` runs the second model into
its own folder; `--host` points at another Ollama server; `--num-ctx` changes the context window. See
`markov-lean local-judge label --help`.

## Prompt versions

`--prompt v1` (the default) is the API judges' prompt, unchanged, which is what a cross-family agreement
check needs. `--prompt v2` (`src/markov_lean/prompts.py`) is an opt-in prompt written for the 7B model:
it asks for the theorem's area first, after a short `topic` field naming the objects the statement is
about; it defines the classes common in Lean with their Lean signals and tie-breaks (a real sequence is 40,
not 26; counting is 05, not 03; a group of order 351 is 20, not 11; 03 only for statements about logic or
set theory itself); it adds twelve short examples written for the prompt, none of them a gold-set problem;
and it shows a `sorry` step without repeating the goal. v2 labels are written as judge
`ollama/<model>@v2` into `<model>_v2/` by default, and a folder started with one version refuses the
other. The API judges always use v1.

## Expected time

The first request also loads the model (a few seconds). After that, a typical pilot proof (about 15 shown
steps, 4K prompt tokens, 400 answer tokens) should take roughly 20 to 40 seconds on the M2. These are
estimates: `check` times one real proof, and `label` prints the measured seconds per proof and the minutes
left every 5 proofs.

| run | proofs | time |
|---|---|---|
| `--sample-steps 500` (default) | about 34 | about 15 to 25 minutes |
| `--sample-steps 2000` | about 135 | about 1 to 1.5 hours |
| `--all` (8,423 pilot proofs) | 8,423 | about 2.5 to 4 days |

500 steps from about 34 proofs gives wide intervals, because intervals resample proofs. For the paper,
2,000 steps (about 135 proofs) is a better target if an hour and a half is free: run with
`--sample-steps 2000` in the same folder and the first 500 are reused.

## Reading the agreement

Once the API labels are in `results/pilot3/judge/labels.parquet`:

```
markov-lean judge-agreement --run results/pilot3 \
  --labels results/pilot3/judge/labels.parquet results/pilot3/judge/local/qwen2.5_7b-instruct/labels.parquet
```

It prints the report and writes `results/pilot3/agreement/judge_agreement.md` and `.json`. Every interval
is a 95% bootstrap over proofs.

- **Step labels / Proof labels.** On the items all three judges labelled: **Fleiss' kappa** across the
  three, the share of items where all three agree, and **Cohen's kappa** and raw agreement for each pair.
  The last column gives each pair's kappa on everything that pair shares (Haiku and Sonnet share far more
  items than either shares with the local judge).
- **NMI with the gate macrostates.** Each judge's NMI and ARI with the gate's macrostates, on the steps all
  judges labelled (comparable across judges) and on each judge's own steps, with the Mathlib-folder
  baseline on the same steps.

How to read it:

- Kappa above about 0.6 is usually called substantial agreement, 0.4 to 0.6 moderate, below 0.4 weak.
- The paper's claim is safer when the cross-family pairs (Haiku vs local, Sonnet vs local) are not much
  below Haiku vs Sonnet. If they are much lower, the two Claude judges may share a bias, and the paper
  should say so.
- Gate 1 does not depend on one family's labels when every judge's NMI with the macrostates clears the
  folder baseline on the common steps. A local judge with a much lower NMI but fair kappa points to noise
  in its labels rather than to the macrostates.

The same section can go into the significance report. The local labels never enter the Haiku and Sonnet
consensus that gate 1 scores:

```
markov-lean significance --run results/pilot3 \
  --extra-labels results/pilot3/judge/local/qwen2.5_7b-instruct/labels.parquet
```

For a quick look without the macrostates, `python -m markov_lean.judge agree --labels <file> --judges A B C`
prints pairwise and Fleiss' kappa for three or more judges from one labels file.
