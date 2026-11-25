"""The dataset card (README.md with a Hugging Face YAML header) for a release folder.

``render_card(summary)`` fills the template from ``run_summary.json``.  The
licence facts were read from each source's own page on 2026-11-25:

* LeanDojo Benchmark 4, Zenodo record 12740403 (v10): Creative Commons
  Attribution 2.0 Generic (CC BY 2.0).  Concept DOI 10.5281/zenodo.8040109.
* Mathlib 4 (leanprover-community/mathlib4, LICENSE): Apache License 2.0.
* MSC 2020 (msc2020.org): CC BY-NC-SA.  The release ships the two-digit codes
  only, never the class titles, so none of the MSC text is redistributed.

The YAML keys follow the Hugging Face dataset card spec
(huggingface/hub-docs ``datasetcard.md``): ``license`` from the Hub licence
list, ``task_categories`` from huggingface.js ``pipelines.ts``, and ``configs``
with ``config_name`` and ``data_files`` as ``{split, path}``.
"""

from __future__ import annotations

from typing import Any

import yaml

PRETTY_NAME = "Lean proof steps with MSC 2020 area labels"
# Labels, model outputs and the card itself.  The goal-state and tactic text is
# derived from Mathlib through LeanDojo and stays under those terms (below).
LICENSE = "cc-by-4.0"

SOURCES = [
    ("LeanDojo Benchmark 4", "Zenodo record 12740403 (v10), `leandojo_benchmark_4.tar.gz`, `random` split",
     "CC BY 2.0", "theorem names, file paths, tactics and goal states (traced from Mathlib)"),
    ("Mathlib 4", "leanprover-community/mathlib4, at the commit LeanDojo traced", "Apache-2.0",
     "the mathematics the tactics and goal states come from"),
    ("MSC 2020", "msc2020.org, two-digit level", "CC BY-NC-SA (codes only are used)",
     "the label vocabulary: 63 top-level codes"),
]

CITATIONS = """```bibtex
@inproceedings{yang2023leandojo,
  title     = {{LeanDojo}: Theorem Proving with Retrieval-Augmented Language Models},
  author    = {Yang, Kaiyu and Swope, Aidan and Gu, Alex and Chalamala, Rahul and Song, Peiyang and
               Yu, Shixing and Godil, Saad and Prenger, Ryan and Anandkumar, Anima},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}
@inproceedings{mathlib2020,
  title     = {The {Lean} Mathematical Library},
  author    = {{The mathlib Community}},
  booktitle = {Proceedings of the 9th ACM SIGPLAN International Conference on Certified Programs and Proofs},
  pages     = {367--381},
  year      = {2020}
}
```"""


def size_category(n: int) -> str:
    """Hugging Face ``size_categories`` bucket for ``n`` rows."""
    for limit, name in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
                        (1_000_000, "100K<n<1M"), (10_000_000, "1M<n<10M"), (100_000_000, "10M<n<100M")):
        if n < limit:
            return name
    return "100M<n<1B"


def header(summary: dict) -> dict:
    steps = int(summary.get("counts", {}).get("steps", 0))
    return {
        "pretty_name": PRETTY_NAME,
        "license": LICENSE,
        "language": ["en"],
        "task_categories": ["text-classification"],
        "size_categories": [size_category(steps)],
        "tags": ["lean4", "mathlib", "theorem-proving", "formal-mathematics", "msc2020", "llm-labels",
                 "markov-state-model"],
        "source_datasets": ["LeanDojo Benchmark 4 (zenodo:12740403)"],
        "configs": [
            {"config_name": "steps", "default": True,
             "data_files": [{"split": "train", "path": "data/steps.parquet"}]},
            {"config_name": "labels", "data_files": [{"split": "train", "path": "data/labels.parquet"}]},
            {"config_name": "proofs", "data_files": [{"split": "train", "path": "data/proofs.parquet"}]},
        ],
    }


def _num(x: Any, digits: int = 3) -> str:
    if isinstance(x, bool) or x is None:
        return "n/a"
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        return "n/a" if x != x else f"{x:.{digits}f}"
    return str(x)


def _agreement_lines(summary: dict) -> list[str]:
    ag = summary.get("judge_agreement") or {}
    out = []
    for level in ("step", "proof"):
        a = ag.get(level)
        if isinstance(a, dict) and a.get("n"):
            out.append(f"- {level}-level: {_num(a.get('n'))} pairs, raw agreement {_num(a.get('agreement'))}, "
                       f"Cohen's kappa {_num(a.get('kappa'))}")
    return out or ["- not recorded in this run"]


def _eval_lines(summary: dict) -> list[str]:
    ev = summary.get("eval") or {}
    out = []
    for key, name in (("macrostates", "Markov-model macrostates"), ("microstates", "k-means microstates"),
                      ("folder_baseline", "Mathlib folder baseline")):
        v = ev.get(key)
        if isinstance(v, dict):
            out.append(f"- {name}: NMI {_num(v.get('nmi'))}, ARI {_num(v.get('ari'))} over {_num(v.get('n'))} steps")
    g = ev.get("gate1")
    if isinstance(g, dict):
        out.append(f"- Gate 1 ({'passed' if g.get('passed') else 'not passed'}): {g.get('rule', '')}")
    return out or ["- not recorded in this run"]


def render_card(summary: dict) -> str:
    c = summary.get("counts", {})
    msm = summary.get("msm") or {}
    judges = summary.get("judges") or []
    by_judge = c.get("labels_by_judge", {})
    judge_lines = [f"- `{j}`: {_num(by_judge.get(j, 0))} labels" for j in judges] or ["- see `labels.judge`"]
    if "consensus" in by_judge:
        judge_lines.append(f"- `consensus`: {_num(by_judge['consensus'])} labels where both judges gave the same code")
    text_note = ("Tactic and goal-state text is included." if summary.get("includes_text", True) else
                 "This copy was exported with `--no-text`: the tactic and goal-state columns are null. "
                 "Rebuild the text from LeanDojo Benchmark 4 by `proof_id` and `step`.")
    src_rows = "\n".join(f"| {a} | {b} | {c_} | {d} |" for a, b, c_, d in SOURCES)
    failed = summary.get("msm_checks_failed") or []
    issues = summary.get("judge_issues") or {}
    issue_txt = ", ".join(f"{k} {v}" for k, v in sorted(issues.items())) or "none recorded"
    filt = summary.get("label_filtering") or {}

    body = f"""# {PRETTY_NAME}

Tactic steps from Lean 4 proofs in Mathlib, each goal state labelled with the area of mathematics it
belongs to (a two-digit MSC 2020 code) by two LLM judges, plus the area a Markov state model over goal
states assigns to it. It was built to ask whether proofs move between areas of mathematics as they go,
and how often.

## Contents

| table | rows | one row per |
|---|---|---|
| `steps` | {_num(c.get('steps'))} | tactic step: theorem, file, tactic, goal state before and after, Markov-model area |
| `labels` | {_num(c.get('labels'))} | judge label for a step, or for a whole proof (`step = -1`) |
| `proofs` | {_num(c.get('proofs'))} | proof: length, Mathlib folder, area switches, consensus area |

Every column is described in `SCHEMA.md`. `MANIFEST.json` and `SHA256SUMS` give each file's size and
SHA-256; check a download with `sha256sum -c SHA256SUMS` (Linux) or `shasum -a 256 -c SHA256SUMS` (macOS).
{text_note}

```python
import pandas as pd
steps = pd.read_parquet("data/steps.parquet")
labels = pd.read_parquet("data/labels.parquet")
consensus = labels[(labels.judge == "consensus") & (labels.step >= 0)]
steps.merge(consensus, on=["proof_id", "step"])
```

## Sources and licences

| source | what was taken | licence | used for |
|---|---|---|---|
{src_rows}

The labels, the Markov-model assignments and this card are released under CC BY 4.0. The goal states,
tactics, theorem names and file paths are derived from Mathlib (Apache-2.0) through LeanDojo Benchmark 4
(CC BY 2.0) and remain under those terms: keep the attribution below when you redistribute them. The MSC
class titles (CC BY-NC-SA) are not included; look codes up at msc2020.org.

## How it was made

1. **Proofs.** Proofs from LeanDojo Benchmark 4 (`random` split, `train`), keeping the long ones (the
   pilot keeps proofs with at least 8 traced steps; the shortest here has {_num(c.get('min_proof_steps'))}), steps in
   LeanDojo's traced order. Steps count the kept tactics; LeanDojo drops tactics whose goal
   state is already closed and focusing dots.
2. **Markov-model areas.** Goal states before each tactic were featurized (TF-IDF over Lean tokens, then
   truncated SVD), clustered into microstates with k-means (count chosen by cross-validated VAMP-2), and a
   reversible Markov state model was fitted with time counted in proof steps. PCCA+ coarse-grained it into
   a fixed, preregistered number of macrostates. This run: {_num(msm.get('n_microstates'))} microstates,
   lag {_num(msm.get('lag'))} step(s), {_num(msm.get('n_macrostates'))} macrostates. Model checks that did not
   pass: {', '.join(f'`{x}`' for x in failed) if failed else 'none'}.
3. **Judge labels.** Two LLM judges labelled each proof independently through the Anthropic Message
   Batches API: one code for the whole proof and one per step (proofs over 48 steps show an even sample of
   48; goal states over 1,500 characters are cut in the middle). Each answer is a two-digit MSC 2020 code
   and a confidence. The judges are:
{chr(10).join('   ' + x for x in judge_lines)}

   Judge requests that failed: {issue_txt}. Labels dropped at export: {_num(filt.get('labels_invalid_code', 0))}
   with an invalid code, {_num(filt.get('labels_outside_sample', 0))} for proofs outside the sample,
   {_num(filt.get('labels_unknown_step', 0))} for unknown steps.
4. **Consensus.** A `consensus` row exists where both judges gave the same code; its confidence is the
   mean of the two.

### Judge agreement

{chr(10).join(_agreement_lines(summary))}

### Areas against the consensus labels

{chr(10).join(_eval_lines(summary))}

## Intended use

- Studying how formal proofs move between areas of mathematics, and testing area-aware proof search.
- Benchmarking area classifiers for Lean goal states against LLM labels, with the folder baseline as a
  reference.
- Not as ground truth for MSC classification: no human checked these labels.

## Known limits

- **Labels are model outputs.** Both judges are LLMs and can share the same mistakes; agreement between
  them is not accuracy. Use the `consensus` rows or filter on `confidence` when that matters, and expect
  weaker labels on short, generic goal states (arithmetic side goals, `simp` clean-up).
- **One library, one version.** Every proof comes from Mathlib at the single commit LeanDojo traced, so
  the area mix reflects what Mathlib had formalized then (algebra, topology and analysis heavy). File
  paths and names may differ from current Mathlib.
- **Long proofs only.** Short proofs were left out (the shortest here has {_num(c.get('min_proof_steps'))} steps), so
  the data says nothing about how short proofs behave.
- **Step order is LeanDojo's.** It follows the source text, not the order Lean elaborates goals, and
  multi-goal states are labelled as one state.
- **Two-digit codes are coarse.** Many Mathlib files sit across classes (e.g. 13 and 14), and the judges
  must pick one.
- **Markov-model areas are unsupervised.** Macrostate numbers carry no meaning across runs, and the model
  checks above say where the fit is weak.

## Reproducing

`REPRODUCE.md` in the code repository walks from a clean clone to this folder
(`markov-lean export-dataset --run <pilot run> --out release/`, then `markov-lean check-release release/`).
Release format {summary.get('release_format', 1)}, built {summary.get('created', 'n/a')} from code commit
`{summary.get('code_commit') or 'unknown'}`.

## Citation

Please cite the sources this data is built from:

{CITATIONS}
"""
    head = yaml.safe_dump(header(summary), sort_keys=False, allow_unicode=True, width=100)
    return f"---\n{head}---\n\n{body}"
