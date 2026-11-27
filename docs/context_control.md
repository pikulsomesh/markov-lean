# Judge-context control (`markov_lean.context_control`)

The API judges see a whole proof in one request, in order, and the prompt tells them that
bookkeeping goals take the area of the surrounding argument. Their step labels switch area 0.115
times per move against 0.162 for a within-proof shuffle (ratio 0.71). Part of that coherence could
be the judge smoothing along the order it reads rather than coherence in the proofs. This control
asks that question for the local Ollama judge, at no cost and with no API call.

## Design

The same proofs are labelled twice by the same judge, prompt and decoding:

| condition | what the judge sees |
|---|---|
| ordered | the proof exactly as the API judges see it |
| shuffled | the same theorem, statement, tactics, goals and final state, with the steps in a random order and random three-digit step ids, so neither the order nor the numbers give a step's position |

Answers in the shuffled condition are mapped back to the true step numbers. Switching is then
measured in the true step order, on the steps labelled in both conditions, and each condition is
compared with its own exact within-proof shuffle expectation (`switches.expected_shuffle_rate`),
so a change in the mix of areas does not pass for a change in order.

- Sample: 300 proofs of the 8,423 the API judges labelled, stratified by their proof label, in
  the interleaved order of `ollama_judge.stratified_sample` (a partial run is still spread over
  areas). The permutation is fixed per proof and seed.
- Judge: `qwen2.5:7b-instruct`, prompt v1 (the API judges' prompt, with its "surrounding argument"
  rule, which is the rule under test), temperature 0. `--prompt v2` runs the prompt tuned for 7B
  models into its own folder.

## Reading it

| result | reading |
|---|---|
| shuffled ratio near the ordered ratio (interval of the difference around 0) | for this judge, coherence is in the steps, not in the order it reads |
| shuffled ratio rises toward 1 | the judge smooths along what it reads, so part of the API judges' 0.71 may be the same effect |

The local judge agrees only weakly with the API judges (Cohen's kappa about 0.3 on the pilot
sample), so a null here is weaker evidence than the same control with Claude, which would cost
about $3 on the same proofs once API credit is back. The report also gives each condition's
agreement with the API judges' labels and the API judges' own switching on the same proofs.

## Commands

On the Mac it is stage 6 of the queue (`docs/mac_queue.md`), after the 64-expansion search.
To run it by itself:

```
scripts/mac_context_control.sh            # label (resumable) under the memory guard, then compare
scripts/mac_context_control.sh --status
```

Or step by step:

```
python -m markov_lean.context_control label --run results/pilot3 --proofs 300
python -m markov_lean.context_control compare --out results/context_control/qwen2.5_7b-instruct
```

Outputs in `results/context_control/qwen2.5_7b-instruct/`: `labels.parquet` (judges
`ollama/qwen2.5:7b-instruct#ordered` and `#shuffled`, true step numbers), `progress.jsonl`,
`sample.json`, `summary.json`, `comparison.json`, `proofs.csv` and `REPORT.md`.

## Time

600 requests. At the local judge's 20 to 40 seconds per pilot proof on the M2, about 3.5 to 7
hours, plus any pauses for GitHub runner jobs. Memory: about 7 GB while Qwen is loaded; the
guard unloads it at each pause.
