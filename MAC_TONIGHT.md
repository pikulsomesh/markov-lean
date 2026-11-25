# Tonight on the Mac Mini

One command picks up where you stopped (after `scripts/setup_mac.sh`, during the Mathlib setup)
and ends with the overnight search running. Every stage is skipped when it already finished, so
if anything stops, fix it and run the same command again.

## Before you start

1. If the Mathlib setup from earlier is still running in a Terminal window, let it finish (or
   press Ctrl-C there). The script stops with a plain message if it finds it still running, and
   it reuses the same Mathlib tag either way.
2. Open the Ollama app (stage 6 needs it).
3. Get the code and set your key in this Terminal only (stages 5 and 7 need it; never put it in
   a file in the repo):

```bash
cd ~/markov-lean
git checkout main && git pull
export ANTHROPIC_API_KEY=...
```

## The command

```bash
scripts/mac_tonight.sh
```

From a second Terminal, at any time: `scripts/mac_tonight.sh --status`.

To answer the API question first instead of waiting about 3 hours for it, run
`scripts/mac_tonight.sh --only 7` (a minute), answer it, then run the command above.

## What it does, and roughly how long

The times are estimates, except the model's size.

| stage | what | time |
|---|---|---|
| 1 | Lean and Mathlib: reuses the tag already in `lean/`, or picks it as `MAC_SMOKE.md` does. Saves it to `results/mac/mathlib_tag.txt` | a few minutes if the earlier setup finished, 10 to 20 if not |
| 2 | Lean REPL in `~/repl` at the matching version, then a `2 + 2 = 4` check | 2 to 5 min |
| 3 | prover model, 4.22 GB, to `~/models` (resumes a partial download), then miniF2F | 5 to 15 min, depending on your connection |
| 4 | `search --check`, then the 10-problem smoke search and its report in `results/mac_smoke` | up to about 35 min |
| 5 | `scripts/after_judges.sh`: collects the finished judge batches (free), runs every analysis, writes `~/markov_lean_results.zip` | about 10 to 30 min |
| 6 | local judge on the gold test half (261 problems), prompt v1 then v2, then their accuracy in `results/gold/accuracy_test` | about 1 to 2 hours; it prints minutes left |
| 7 | gold-set API scoring: prints the estimate (about $3.85 last time), then asks y/N. Submits only on your y, capped at $10 | a minute, plus your answer |
| 8 | `scripts/mac_overnight.sh`: all 244 miniF2F problems under the three conditions, then its report | about 26 to 37 hours, plus pauses for runner jobs |

Stages 4, 6 and 8 pause on their own while one of your GitHub runner jobs is active or memory
is tight, and go on afterwards. Nothing but stage 7 can spend money.

## What you do by hand

- **The y/N at stage 7.** Type `y` and Enter to submit, anything else to skip. With no answer
  for 15 minutes it counts as no and the search starts anyway; you can submit later with
  `scripts/mac_tonight.sh --only 7`. It never submits twice.
- **The zip.** When stage 5 prints `Drag ~/markov_lean_results.zip into the Claude Duet chat`,
  do that (any time tonight; the script carries on).
- **The Mathlib tag.** Stage 8 prints it again; paste it into the chat so EC2 can use the same
  one (`MATHLIB_TAG=<tag> bash scripts/setup_ec2.sh`).
- **If it stops.** It prints `STOPPED at stage N` and why. Fix that and rerun; to skip that
  stage for tonight, `scripts/mac_tonight.sh --from N+1` (the search still needs stages 1 to 4).
  Drag the log it names (`results/mac/tonight/logs/...`) into the chat if the reason isn't clear.

## Tomorrow

If you submitted at stage 7: `.venv/bin/python -m markov_lean.judge collect --out results/gold/api`,
then `bash scripts/after_judges.sh` again, which adds the API judges to the gold-set accuracy.

The search keeps going by itself; `scripts/mac_tonight.sh --status` shows how far it is, and
rerunning `scripts/mac_tonight.sh` after a restart resumes it.
