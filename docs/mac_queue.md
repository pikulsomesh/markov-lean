# The Mac run queue

One command runs the Mac Mini's long jobs one after another, never side by side, over as many
days as they take. The queue is `configs/mac_queue.yaml`; `scripts/mac_queue.sh` starts and
watches it.

| # | stage | what it does | time | disk while it runs |
|---|---|---|---|---|
| 1 | `putnam_wait` | waits for the PutnamBench search started on Nov 26 to end (it does not start or restart it) | until that run ends | none |
| 2 | `sphere_packing` | builds sphere packing in dimension 8 and extracts its steps (`scripts/case_extract.sh`) | hours | about 12 GB, freed at its end |
| 3 | `pnt` | the same for PNT+ | hours | about 12 GB, freed at its end |
| 4 | `carleson` | the same for Carleson's theorem (the within-area control) | hours | about 12 GB, freed at its end |
| 5 | `minif2f_b64` | the 64-expansion miniF2F search, 244 problems under 3 conditions (`configs/search_eval_mac_b64.yaml`, results in `results/mac_overnight_b64`) | about 90 h plus pauses | about 1 GB |

The extraction times are a guess: these builds have not run on the Mac yet. Stages 2 to 5 each
need stage 1 to finish; they do not depend on each other.

## Commands

```
cd ~/markov-lean && git pull
scripts/mac_queue.sh            # start it in the background; closing the terminal is fine
scripts/mac_queue.sh --status   # every stage: state, start and end, and the current stage's progress
```

`--stop` stops it cleanly (the current stage saves its work first), `--list` shows each stage's
command, and `--foreground` runs it in the terminal instead.

## What it guarantees

- **One at a time, in order.** The queue holds its own lock, so starting it twice only reports
  that it is running. Each stage keeps its memory guard: it stops while a GitHub runner job is
  active or memory is short, then resumes on its own.
- **Memory bounds.** Every stage's guard has a lower and an upper bound on available memory,
  set as environment variables when you start the queue (or a script on its own):

  | Variable | Bound | Default | What it does |
  |---|---|---|---|
  | `MIN_FREE_GB` | lower | 1 | stops the running job when available memory falls below it |
  | `RESUME_FREE_GB` | upper | 3 | starts or resumes a job only once at least this much is available |

  The job also stops at once for a GitHub runner job, critical memory pressure, or swap growing
  by more than `MAX_SWAP_GROWTH_GB` (1.5); pressure *warn* alone does not stop it. On Nov 27 the
  loaded prover sat at 1.4 to 2.2 GB available and the idle Mac at 7 to 9 GB, so 1 and 3 sit
  below and above those. An extraction with `JOBS=2` can need 4 to 8 GB, so if one stops and
  starts over and over, start the queue with `JOBS=1` or a higher `RESUME_FREE_GB`. The upper bound can't be under the lower one. For example
  `MIN_FREE_GB=1 RESUME_FREE_GB=5 scripts/mac_queue.sh` applies to every stage. Each guard prints
  its bounds when it starts and in `--status` ("memory bounds: stops below ... starts at ...").
  New values take effect when that job starts again, so change them with `--stop` and a restart.
- **Logs and state per stage.** `results/mac_queue/logs/NN_<stage>.log` (one per stage, every
  attempt appended), `results/mac_queue/state.json` (state, start, end, attempts, exit code) and
  `results/mac_queue/queue.log` (the queue's own lines).
- **Done means checked.** A stage is done when its command exits 0 and its outputs exist (for an
  extraction, its steps table and summary; for the search, 244 finished problems per condition).
  Exit 0 with work left, as when a time limit runs out, is *incomplete*.
- **Resume.** Starting the queue again skips done stages, resumes an interrupted one (after a
  reboot, a crash or `--stop`), and runs failed and incomplete ones again; each stage's script
  skips the work it already finished.
- **Disk.** Before each stage starts, the disk must have room (15 GB for an extraction, 3 GB for
  the search); `case_extract.sh` checks again before each of its steps.
- **No spending.** API and AWS credentials are removed from every stage's environment.

After a reboot, run `scripts/mac_queue.sh` again. To have that happen at login instead, there is
an optional launchd agent (`scripts/mac_queue.sh --agent-plist` prints it with install and remove
lines); it starts the queue once per login and never restarts a queue that stopped on its own.

## When a stage fails

The queue records it (`--status` shows the reason and its log) and goes on to the next stage,
since none depends on another. A failed extraction also frees its build files at once, so the
next project has the disk; the steps it already extracted stay in `results/cases/<name>/raw`.

1. Read the end of its log: `tail -50 results/mac_queue/logs/02_sphere_packing.log`
   (send it to Claude if the cause is not obvious).
2. Once it is fixed (more disk, a pull with a fix), start the queue again: `scripts/mac_queue.sh`.
   Failed and incomplete stages run again; done ones are skipped.
3. To leave a stage out for good: `scripts/mac_queue.sh --skip sphere_packing`. To run a done
   stage again: `scripts/mac_queue.sh --retry pnt`. Both need the queue stopped.

A stage whose extraction finishes with a few files failing is still done: those files are
listed in `results/cases/<name>/raw/failed.txt`, and the rest of the project is used.
