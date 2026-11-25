# Search smoke test on the Mac Mini

Tonight's run, from Lean setup to the overnight search, is one command: `scripts/mac_tonight.sh`
(checklist in `MAC_TONIGHT.md`). The steps below are what it runs.

A 10-problem miniF2F run of the search evaluation on the Mac Mini (M2, 16 GB), with a small
budget, before any GPU time. The prover is a 4-bit GGUF of DeepSeek-Prover-V1.5-RL that
llama.cpp runs inside the search process on the M2's GPU (Metal), in place of the vLLM
server on EC2. No Ollama or other server is needed.

What it checks: Lean, Mathlib and the REPL on a real machine, the DeepSeek prompts, the
first-tactic parsing, the priors, the logs, resume and the report. What it does not give:
numbers for the paper. The weights are quantized, the budget is tiny and the sampler is not
vLLM's, so pass rates here are not comparable with the EC2 runs.

Nothing here calls an API, uses AWS or costs money.

## What it needs

| | size | time (first run) |
|---|---|---|
| Lean toolchain (elan) | about 1 GB | 2 min |
| Mathlib prebuilt cache in `lean/.lake` | about 5 to 6 GB | 10 to 20 min, mostly download |
| leanprover-community/repl in `~/repl` | under 0.5 GB | 2 to 5 min build |
| llama-cpp-python (built with Metal) | under 0.5 GB | 5 to 10 min build |
| model `deepseek-prover-v1.5-rl_q4_k_m.gguf` in `~/models` | **4.22 GB** | download |

About 11 GB of free disk in all. While the search runs, memory use is about 9 GB: the model
(4.2 GB), its 4096-token KV cache (about 2 GB) and the Lean REPL with Mathlib loaded. Close
other big apps. The times above are estimates; only the model size is from its download page.

## Commands, in order

Run everything from the repo root (the folder with `pyproject.toml`), in Terminal or over
SSH. Steps 1 to 4 are once only.

```bash
# 0. the code (until the PR merges, use its branch)
git fetch origin && git checkout mac-smoke-backend && git pull

# 1. Python environment, plus llama.cpp's Python binding built for Metal
#    (needs the Xcode Command Line Tools: xcode-select --install)
./scripts/setup_mac.sh
source .venv/bin/activate
CMAKE_ARGS="-DGGML_METAL=on" pip install "llama-cpp-python>=0.3"
python -c "import llama_cpp; print('llama.cpp', llama_cpp.__version__, 'metal:', llama_cpp.llama_supports_gpu_offload())"
#    expect "metal: True"

# 2. Lean and Mathlib. Pin Mathlib to the newest stable release that the REPL also has,
#    so step 3 finds a matching REPL tag. Write the tag down: use the same one on EC2
#    (MATHLIB_TAG=<tag> bash scripts/setup_ec2.sh), so both machines check proofs alike.
MATHLIB_TAG=$(comm -12 \
  <(git ls-remote --tags --refs https://github.com/leanprover-community/mathlib4.git | awk -F/ '{print $3}' | grep -E '^v4\.[0-9]+\.0$' | sort) \
  <(git ls-remote --tags --refs https://github.com/leanprover-community/repl.git | awk -F/ '{print $3}' | grep -E '^v4\.[0-9]+\.0$' | sort) \
  | sort -V | tail -1)
echo "Mathlib tag: $MATHLIB_TAG"
MATHLIB_TAG=$MATHLIB_TAG ./scripts/setup_lean.sh
source "$HOME/.elan/env"

# 3. The Lean REPL, at the tag that matches lean/lean-toolchain
LEAN_TAG=$(cut -d: -f2 lean/lean-toolchain)
[ -d ~/repl/.git ] || git clone https://github.com/leanprover-community/repl.git ~/repl
git -C ~/repl fetch --tags -q && git -C ~/repl checkout -q "$LEAN_TAG"
(cd ~/repl && lake build)
printf '{"cmd": "example : 2 + 2 = 4 := by norm_num"}\n\n' | (cd lean && lake env ~/repl/.lake/build/bin/repl)
#    expect a JSON reply with "env": 0 and no "severity": "error"

# 4. The model (4.22 GB) and the benchmark
mkdir -p ~/models
curl -fL --retry 4 -C - -o ~/models/deepseek-prover-v1.5-rl_q4_k_m.gguf \
  https://huggingface.co/neopolita/deepseek-prover-v1.5-rl-gguf/resolve/main/deepseek-prover-v1.5-rl_q4_k_m.gguf
ls -lh ~/models/deepseek-prover-v1.5-rl_q4_k_m.gguf     # about 3.9G (4.22 GB)
.venv/bin/markov-lean fetch minif2f

# 5. Check the prover and Lean together (loads the model, samples 4 tactics, imports Mathlib)
.venv/bin/python -m markov_lean.search --check --config configs/search.yaml --backend llama_cpp --model-path ~/models/deepseek-prover-v1.5-rl_q4_k_m.gguf
#    expect "sample tactics (...)" with Lean-looking tactics such as omega or simp [add_comm],
#    then "lean ok: True"

# 6. The smoke run: 10 miniF2F test problems, LM prior only, 12 expansions each
.venv/bin/markov-lean search-eval --config configs/search_eval_mac.yaml --out results/mac_smoke --benchmark minif2f --limit 10 --condition lm --estimate
.venv/bin/markov-lean search-eval --config configs/search_eval_mac.yaml --out results/mac_smoke --benchmark minif2f --limit 10 --condition lm --yes

# 7. Optional: the same 10 problems with the area prior and the random control, if the
#    saved Markov model is on this Mac (results/pilot3/msm/featurizer.pkl exists)
.venv/bin/markov-lean search-eval --config configs/search_eval_mac.yaml --out results/mac_smoke --benchmark minif2f --limit 10 --condition area_l0.3 --condition random_l0.3 --yes

# 8. Report
.venv/bin/markov-lean search-report --runs results/mac_smoke --no-figures
```

The model file is a community GGUF conversion (neopolita on Hugging Face) of
deepseek-ai/DeepSeek-Prover-V1.5-RL, under DeepSeek's model license. `--model-path` can
point at any other GGUF of the same model; Q4_K_M is the size that leaves room for Lean in
16 GB.

After the smoke test, the full comparison on the Mac (one night, or all 244 problems) is
`scripts/mac_overnight.sh`: see "On the Mac Mini" in `docs/search_eval.md`.

## How long

Step 6 has a ceiling of 10 problems × 180 s = 30 minutes of search, plus loading the model
and importing Mathlib (a few minutes the first time). Problems that are solved or run out
of candidates finish sooner. The `--estimate` line prints its own guess; once some problems
have finished, a rerun of `--estimate` uses their measured times. Step 7 is up to another
hour (two conditions). Ctrl-C stops a run; rerunning the same command resumes it.

## Reading the output

Each problem logs one line while it runs:

```
[3/10] minif2f lm seed0 mathd_algebra_48: solved (solved, 2 exp, 21s) | ...
```

The part in brackets is the stop reason, the expansions (prover calls) used and the time.
When the prover closes it logs `local prover: N requests, N samples, N tokens generated,
N s (N tok/s)`; on an M2 expect very roughly 15 to 30 tokens a second.

Files, under `results/mac_smoke/minif2f/lm/seed0/`:

* `summary.jsonl`: one line per problem (`solved`, `stop_reason`, `expansions`, `elapsed`,
  and the full `proof` when solved);
* `trace.jsonl` / `trace.parquet`: every tactic tried, with Lean's error when it failed.

A quick look at both:

```bash
.venv/bin/python - <<'EOF'
import json, pandas as pd
d = "results/mac_smoke/minif2f/lm/seed0"
s = pd.DataFrame([json.loads(l) for l in open(f"{d}/summary.jsonl")])
print(s[["theorem", "solved", "stop_reason", "expansions", "elapsed"]].to_string(index=False))
t = pd.read_parquet(f"{d}/trace.parquet")
print(f"\n{len(t)} tactics tried, {t['ok'].mean():.0%} accepted by Lean")
print(t["error"].fillna("ok").str.split("\n").str[-1].str.slice(0, 70).value_counts().head(12).to_string())
EOF
```

What a healthy run looks like: every problem has a summary line; stop reasons are mostly
`solved`, `budget` or `wall_time`; the trace shows real tactics (`nlinarith [...]`,
`norm_num`, `omega`, `simp ...`), some accepted and many rejected with ordinary Lean errors
(`linarith failed`, `unsolved goals`). A few solved out of 10 is plausible at this budget;
zero solved with everything else healthy still means the harness works.

What points to a harness problem:

| symptom | likely cause |
|---|---|
| `stop_reason` starts with `statement error` | the statement doesn't elaborate under this Mathlib: the benchmark's header or names don't match it |
| many `unknown identifier` / `unknown constant` errors | Mathlib renamed lemmas the model was trained on; try an older `MATHLIB_TAG` |
| most errors are `timeout` | Lean is too slow for `lean.timeout` (60 s), or memory is swapping |
| `expansions` is 1 and the stop reason `exhausted` everywhere | no usable tactic came back: a prompt or parsing problem |
| `task ... failed` lines, or `stopped: 3 errors in a row` | the REPL or the model crashed; the error text says which |
| `REPL did not start` | Lean setup: the message ends with the REPL's own stderr |
| `metal: False` in step 1 | llama-cpp-python was built without Metal; reinstall with `CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall --no-cache-dir "llama-cpp-python>=0.3"` |

## How the local prover differs from vLLM

Same prompt, stop strings, first-tactic parsing and de-duplication. Sampling is plain
temperature 1.0 and top-p 0.95 over the whole vocabulary (no top-k, min-p or repetition
penalty), and each candidate's score is the raw log-probability of its first tactic, as
vLLM reports it. Sample `i` of a request with seed `s` uses its own fixed random stream, so
every prior condition sees the same samples at the same goal state. Differences: the `n`
samples run one after another; generation stops at the end of the first tactic instead of
running to `max_tokens`; and a prompt longer than the 4096-token context gets no candidates
instead of an error. See `src/markov_lean/search/local_prover.py`.
