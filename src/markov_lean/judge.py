"""LLM judges that label Lean proofs and proof steps with MSC 2020 top-level codes.

There is no hand-made gold set, so two judges label the same proofs
independently (by default Claude Haiku 5.5 and Claude Sonnet 5.5) and the
agreement between them, plus accuracy on statements whose MSC code is known,
says how far the labels can be trusted.

Flow (all requests go through the Message Batches API at half price):

    estimate  -> prints the dollar estimate, spends nothing
    submit    -> prints the estimate, refuses if it is over --max-dollars,
                 writes requests + manifest, then creates the batches
                 (--dry-run stops before the API is called)
    collect   -> polls the batches, checks stop_reason, writes the `labels`
                 table from docs/schema.md
    agree     -> agreement statistics between two judges
    validate  -> MSC-180 style check: judge accuracy on statements with known codes

Run as ``python -m markov_lean.judge <command> --help``. The API key is read
from ANTHROPIC_API_KEY by the SDK; this module never reads, prints or writes it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import yaml

from markov_lean import msc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "judge.yaml"
LABEL_COLUMNS = ["proof_id", "step", "judge", "msc", "confidence"]
PROOF_LEVEL_STEP = -1
CONSENSUS_JUDGE = "consensus"


# --------------------------------------------------------------------------- config


@dataclass
class JudgeSpec:
    model: str
    max_tokens: int = 4096
    effort: str = "low"


@dataclass
class Price:
    input: float  # USD per million input tokens, standard (non-batch) rate
    output: float  # USD per million output tokens


@dataclass
class JudgeConfig:
    judges: list[JudgeSpec]
    pricing: dict[str, Price]
    max_steps_per_proof: int = 48
    max_state_chars: int = 1500
    batch_discount: float = 0.5
    max_dollars: float = 20.0
    chunk_size: int = 10_000
    poll_seconds: float = 60.0
    timeout_hours: float = 24.0
    chars_per_token: float = 3.0
    thinking_tokens: int = 400
    output_tokens_fixed: int = 60
    output_tokens_per_step: int = 30

    @property
    def models(self) -> list[str]:
        return [j.model for j in self.judges]


def load_config(path: str | Path = DEFAULT_CONFIG) -> JudgeConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    judges = [JudgeSpec(**j) for j in raw["judges"]]
    pricing = {m: Price(**p) for m, p in raw["pricing"]["models"].items()}
    for j in judges:
        if j.model not in pricing:
            raise ValueError(f"no price for judge model {j.model} in {path}")
    prompt, batch, budget, est = (raw.get(k, {}) for k in ("prompt", "batch", "budget", "estimate"))
    return JudgeConfig(
        judges=judges,
        pricing=pricing,
        max_steps_per_proof=prompt.get("max_steps_per_proof", 48),
        max_state_chars=prompt.get("max_state_chars", 1500),
        batch_discount=raw["pricing"].get("batch_discount", 0.5),
        max_dollars=budget.get("max_dollars", 20.0),
        chunk_size=batch.get("chunk_size", 10_000),
        poll_seconds=batch.get("poll_seconds", 60.0),
        timeout_hours=batch.get("timeout_hours", 24.0),
        chars_per_token=est.get("chars_per_token", 3.0),
        thinking_tokens=est.get("thinking_tokens", 400),
        output_tokens_fixed=est.get("output_tokens_fixed", 60),
        output_tokens_per_step=est.get("output_tokens_per_step", 30),
    )


# --------------------------------------------------------------------------- proofs


@dataclass
class Step:
    step: int
    tactic: str
    state_before: str
    state_after: str


@dataclass
class Proof:
    proof_id: str
    theorem: str
    statement: str
    steps: list[Step] = field(default_factory=list)
    n_steps_total: int = 0


def _truncate(text: str, limit: int) -> str:
    text = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n  ... [truncated] ...\n" + text[-half:]


def _even_sample(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)}) if k > 1 else [0]


def proofs_from_steps(steps: pd.DataFrame, max_steps: int = 48, max_state_chars: int = 1500) -> list[Proof]:
    """Group a `steps` table into proofs. Long proofs show an even sample of steps."""
    proofs = []
    for proof_id, g in steps.sort_values(["proof_id", "step"]).groupby("proof_id", sort=False):
        rows = list(g.itertuples(index=False))
        keep = _even_sample(len(rows), max_steps)
        shown = [
            Step(
                step=int(rows[i].step),
                tactic=_truncate(rows[i].tactic, 400),
                state_before=_truncate(rows[i].state_before, max_state_chars),
                state_after=_truncate(rows[i].state_after, max_state_chars),
            )
            for i in keep
        ]
        proofs.append(
            Proof(
                proof_id=str(proof_id),
                theorem=str(rows[0].theorem),
                statement=_truncate(rows[0].state_before, max_state_chars),
                steps=shown,
                n_steps_total=len(rows),
            )
        )
    return proofs


def proofs_from_known(known: pd.DataFrame, max_state_chars: int = 1500) -> list[Proof]:
    """Statements with known MSC codes (columns: id, statement, msc) as step-less proofs."""
    return [
        Proof(proof_id=str(r.id), theorem=str(getattr(r, "name", r.id)),
              statement=_truncate(r.statement, max_state_chars))
        for r in known.itertuples(index=False)
    ]


# --------------------------------------------------------------------------- prompt

SYSTEM_PROMPT = """You label Lean 4 proofs with areas of mathematics from the Mathematics Subject \
Classification 2020 (MSC 2020), using its two-digit top-level codes only.

For each proof step you are shown the tactic and the goal state it acts on. Give the area that \
the mathematics of that goal state belongs to, as a working mathematician would classify it. \
Classify the mathematical content of the goal, not the tactic: `simp`, `ring` or `omega` say \
nothing about the area by themselves. A step can sit in a different area from the theorem, for \
example a real-analysis estimate inside a number-theory proof; label what the step is about. \
Goals that are pure logic or bookkeeping (introducing variables, splitting conjunctions) take \
the area of the surrounding argument, not 03, unless the goal itself is about logic or set theory.

Then give one area for the whole proof: the area of the theorem's statement.

For every label give a confidence between 0 and 1 for how sure you are of the area. Use the \
full range: about 0.9 when the area is unmistakable, about 0.5 when two areas are equally \
plausible.

MSC 2020 top-level classes:
""" + msc.prompt_listing()


def output_schema() -> dict:
    code = {"type": "string", "enum": list(msc.CODES)}
    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer"},
                        "msc": code,
                        "confidence": {"type": "number"},
                    },
                    "required": ["step", "msc", "confidence"],
                    "additionalProperties": False,
                },
            },
            "proof_msc": code,
            "proof_confidence": {"type": "number"},
        },
        "required": ["steps", "proof_msc", "proof_confidence"],
        "additionalProperties": False,
    }


def render_user(proof: Proof) -> str:
    parts = [f"Theorem: {proof.theorem}", "", "Statement (initial goal):", proof.statement, ""]
    if not proof.steps:
        parts.append("No proof steps are given. Return an empty `steps` list and label the statement.")
        return "\n".join(parts)
    if proof.n_steps_total > len(proof.steps):
        parts.append(f"The proof has {proof.n_steps_total} steps; {len(proof.steps)} of them are shown.")
    parts.append("Proof steps. Label each one, using the step numbers given:")
    for s in proof.steps:
        parts += [f"[step {s.step}] tactic: {s.tactic}", "goal before:", s.state_before, ""]
    parts += ["State after the last shown step:", proof.steps[-1].state_after]
    return "\n".join(parts)


def _custom_id(model: str, i: int) -> str:
    # custom_id must match ^[a-zA-Z0-9_-]{1,64}$; proof ids contain ':' and '.', so use an index
    return f"{model.replace('.', '-')}-{i:07d}"


def build_requests(proofs: list[Proof], cfg: JudgeConfig) -> tuple[list[dict], dict[str, dict]]:
    """One Message Batches request per (proof, judge), plus the manifest that maps them back."""
    requests, manifest = [], {}
    schema = output_schema()
    for judge in cfg.judges:
        for i, proof in enumerate(proofs):
            cid = _custom_id(judge.model, i)
            requests.append({
                "custom_id": cid,
                "params": {
                    "model": judge.model,
                    "max_tokens": judge.max_tokens,  # thinking counts toward this
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": render_user(proof)}],
                    "output_config": {"effort": judge.effort, "format": {"type": "json_schema", "schema": schema}},
                },
            })
            manifest[cid] = {"proof_id": proof.proof_id, "judge": judge.model,
                             "steps": [s.step for s in proof.steps]}
    return requests, manifest


# --------------------------------------------------------------------------- cost


@dataclass
class CostEstimate:
    per_judge: dict[str, dict[str, float]]
    batch_discount: float

    @property
    def total(self) -> float:
        return sum(v["dollars"] for v in self.per_judge.values())

    def summary(self) -> str:
        lines = ["Estimated cost (Message Batches, "
                 f"{int((1 - self.batch_discount) * 100)}% off standard rates):"]
        for m, v in self.per_judge.items():
            lines.append(f"  {m}: {int(v['requests'])} requests, ~{int(v['input_tokens']):,} input + "
                         f"~{int(v['output_tokens']):,} output tokens = ${v['dollars']:.2f}")
        lines.append(f"  total: ${self.total:.2f} (heuristic token counts; real usage can differ)")
        return "\n".join(lines)


def _tokens(text: str, chars_per_token: float) -> int:
    return math.ceil(len(text) / chars_per_token)


def estimate_cost(requests: list[dict], manifest: dict[str, dict], cfg: JudgeConfig) -> CostEstimate:
    system_tokens = _tokens(SYSTEM_PROMPT + json.dumps(output_schema()), cfg.chars_per_token)
    per: dict[str, dict[str, float]] = {}
    for r in requests:
        p = r["params"]
        model = p["model"]
        n_steps = len(manifest[r["custom_id"]]["steps"])
        inp = system_tokens + _tokens(p["messages"][0]["content"], cfg.chars_per_token)
        out = min(p["max_tokens"], cfg.thinking_tokens + cfg.output_tokens_fixed
                  + cfg.output_tokens_per_step * n_steps)
        v = per.setdefault(model, {"requests": 0, "input_tokens": 0, "output_tokens": 0, "dollars": 0.0})
        price = cfg.pricing[model]
        v["requests"] += 1
        v["input_tokens"] += inp
        v["output_tokens"] += out
        v["dollars"] += (inp * price.input + out * price.output) / 1e6 * cfg.batch_discount
    return CostEstimate(per, cfg.batch_discount)


def synthetic_proofs(n_proofs: int, mean_steps: float, mean_state_chars: int, seed: int = 0) -> list[Proof]:
    """Placeholder proofs of a given size, for planning a budget before data exists."""
    rng = random.Random(seed)
    proofs = []
    for i in range(n_proofs):
        n = max(1, round(rng.expovariate(1 / mean_steps)))
        state = "x" * mean_state_chars
        steps = [Step(k, "simp [foo, bar]", state, state) for k in range(n)]
        proofs.append(Proof(f"synthetic:t{i}", f"Synthetic.theorem_{i}", state, steps, n))
    return proofs


class BudgetExceeded(RuntimeError):
    pass


def check_budget(est: CostEstimate, cfg: JudgeConfig, max_dollars: float | None) -> float:
    cap = cfg.max_dollars if max_dollars is None else min(max_dollars, cfg.max_dollars)
    if est.total > cap:
        raise BudgetExceeded(f"estimated ${est.total:.2f} is over the ${cap:.2f} cap; nothing was submitted")
    return cap


# --------------------------------------------------------------------------- batches


def make_client():  # pragma: no cover - needs the real SDK and a key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set in this shell")
    import anthropic

    return anthropic.Anthropic()


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def submit(
    requests: list[dict],
    manifest: dict[str, dict],
    cfg: JudgeConfig,
    out_dir: str | Path,
    client: Any = None,
    max_dollars: float | None = None,
    dry_run: bool = False,
    echo: Callable[[str], None] = print,
) -> list[str]:
    """Estimate, check the budget, save the run, then create the batches. Returns batch ids."""
    est = estimate_cost(requests, manifest, cfg)
    echo(est.summary())
    cap = check_budget(est, cfg, max_dollars)
    echo(f"Within the ${cap:.2f} cap.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "requests.jsonl", requests)
    (out / "manifest.json").write_text(json.dumps(manifest))
    if dry_run:
        echo(f"Dry run: {len(requests)} requests written to {out / 'requests.jsonl'}; nothing submitted.")
        return []
    if client is None:
        client = make_client()
    batch_ids = []
    for start in range(0, len(requests), cfg.chunk_size):
        batch = client.messages.batches.create(requests=requests[start:start + cfg.chunk_size])
        batch_ids.append(batch.id)
        echo(f"Created batch {batch.id} ({min(cfg.chunk_size, len(requests) - start)} requests)")
    (out / "batches.json").write_text(json.dumps({"batch_ids": batch_ids,
                                                  "estimated_dollars": round(est.total, 4)}))
    return batch_ids


def wait(client: Any, batch_ids: list[str], poll_seconds: float, timeout_hours: float,
         sleep: Callable[[float], None] = time.sleep, echo: Callable[[str], None] = print) -> None:
    deadline = time.monotonic() + timeout_hours * 3600
    pending = list(batch_ids)
    while pending:
        still = []
        for bid in pending:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status != "ended":
                still.append(bid)
                echo(f"{bid}: {b.processing_status}, {b.request_counts.processing} processing")
        pending = still
        if pending:
            if time.monotonic() > deadline:
                raise TimeoutError(f"batches still running after {timeout_hours} h: {pending}")
            sleep(poll_seconds)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _clip(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return min(1.0, max(0.0, v)) if not math.isnan(v) else v


def parse_result(result: Any, entry: dict) -> tuple[list[dict], dict | None]:
    """Turn one batch result into label rows. Returns (rows, issue or None)."""
    cid = _get(result, "custom_id")
    res = _get(result, "result")
    rtype = _get(res, "type")
    base = {"custom_id": cid, "proof_id": entry["proof_id"], "judge": entry["judge"]}
    if rtype != "succeeded":
        err = _get(_get(res, "error"), "type")
        return [], {**base, "issue": rtype, "detail": err}
    message = _get(res, "message")
    stop = _get(message, "stop_reason")
    if stop == "refusal":
        category = _get(_get(message, "stop_details"), "category")
        return [], {**base, "issue": "refusal", "detail": category}
    if stop == "max_tokens":
        return [], {**base, "issue": "max_tokens", "detail": "raise max_tokens or lower effort"}
    text = next((_get(b, "text") for b in _get(message, "content", []) if _get(b, "type") == "text"), None)
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return [], {**base, "issue": "bad_json", "detail": (text or "")[:200]}

    rows, dropped = [], 0
    pc = msc.normalize(data.get("proof_msc"))
    if pc:
        rows.append({"proof_id": entry["proof_id"], "step": PROOF_LEVEL_STEP, "judge": entry["judge"],
                     "msc": pc, "confidence": _clip(data.get("proof_confidence"))})
    else:
        dropped += 1
    wanted, seen = set(entry["steps"]), set()
    for s in data.get("steps", []):
        step, code = s.get("step"), msc.normalize(s.get("msc"))
        if step not in wanted or step in seen or code is None:
            dropped += 1
            continue
        seen.add(step)
        rows.append({"proof_id": entry["proof_id"], "step": int(step), "judge": entry["judge"],
                     "msc": code, "confidence": _clip(s.get("confidence"))})
    missing = len(wanted - seen)
    issue = None
    if dropped or missing:
        issue = {**base, "issue": "partial", "detail": f"{dropped} invalid, {missing} steps missing"}
    return rows, issue


def collect(client: Any, batch_ids: list[str], manifest: dict[str, dict]) -> tuple[pd.DataFrame, list[dict]]:
    rows, issues = [], []
    for bid in batch_ids:
        for result in client.messages.batches.results(bid):  # any order: key by custom_id
            entry = manifest.get(_get(result, "custom_id"))
            if entry is None:
                issues.append({"custom_id": _get(result, "custom_id"), "issue": "unknown_custom_id"})
                continue
            r, issue = parse_result(result, entry)
            rows += r
            if issue:
                issues.append(issue)
    return labels_frame(rows), issues


def labels_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    return df.astype({"proof_id": str, "step": "int64", "judge": str, "msc": str, "confidence": "float64"})


# --------------------------------------------------------------------------- agreement


def _kappa(a: list[str], b: list[str]) -> float:
    if not a:
        return float("nan")
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def paired(labels: pd.DataFrame, judge_a: str, judge_b: str) -> pd.DataFrame:
    a = labels[labels.judge == judge_a].drop_duplicates(["proof_id", "step"])
    b = labels[labels.judge == judge_b].drop_duplicates(["proof_id", "step"])
    return a.merge(b, on=["proof_id", "step"], suffixes=("_a", "_b"))


def agreement(labels: pd.DataFrame, judge_a: str, judge_b: str, top_k: int = 10) -> dict:
    """Raw agreement, Cohen's kappa and the most common disagreements, for steps and whole proofs."""
    pairs = paired(labels, judge_a, judge_b)
    out: dict[str, Any] = {"judges": [judge_a, judge_b]}
    for level, sel in (("step", pairs.step >= 0), ("proof", pairs.step == PROOF_LEVEL_STEP)):
        p = pairs[sel]
        a, b = p.msc_a.tolist(), p.msc_b.tolist()
        agree = (p.msc_a == p.msc_b)
        conf = p[["confidence_a", "confidence_b"]].min(axis=1)
        bins = pd.cut(conf, [0, 0.5, 0.7, 0.9, 1.0], include_lowest=True)
        by_conf = agree.groupby(bins, observed=False).agg(["mean", "size"])
        out[level] = {
            "n": int(len(p)),
            "agreement": float(agree.mean()) if len(p) else float("nan"),
            "kappa": _kappa(a, b),
            "agreement_by_min_confidence": {
                str(k): {"agreement": None if math.isnan(r["mean"]) else float(r["mean"]), "n": int(r["size"])}
                for k, r in by_conf.iterrows()
            },
            "top_disagreements": [
                {"a": x, "b": y, "n": int(n)}
                for (x, y), n in p[~agree].groupby(["msc_a", "msc_b"]).size().nlargest(top_k).items()
            ],
        }
    return out


def consensus(labels: pd.DataFrame, judge_a: str, judge_b: str) -> pd.DataFrame:
    """Rows where both judges agree, as judge='consensus' with the mean confidence."""
    p = paired(labels, judge_a, judge_b)
    p = p[p.msc_a == p.msc_b]
    return labels_frame([
        {"proof_id": r.proof_id, "step": r.step, "judge": CONSENSUS_JUDGE, "msc": r.msc_a,
         "confidence": (r.confidence_a + r.confidence_b) / 2}
        for r in p.itertuples(index=False)
    ])


# --------------------------------------------------------------------------- validation


def _known_codes(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).replace(",", ";").split(";")
    codes = [msc.normalize(c) for c in raw]
    return [c for c in codes if c]


def score_known(labels: pd.DataFrame, known: pd.DataFrame) -> dict:
    """MSC-180 style validation hook.

    ``known`` has columns ``id`` and ``msc`` (primary code first; several codes
    may be separated by ';' and finer codes such as '11A41' are cut to the top
    level). Scores the proof-level labels: top-1 accuracy against the primary
    code, any-match accuracy against all listed codes, and accuracy per area.
    """
    truth = {str(r.id): _known_codes(r.msc) for r in known.itertuples(index=False)}
    truth = {k: v for k, v in truth.items() if v}
    proof = labels[labels.step == PROOF_LEVEL_STEP]
    report: dict[str, Any] = {}
    for judge, g in proof.groupby("judge"):
        g = g[g.proof_id.isin(truth)]
        top1 = [r.msc == truth[r.proof_id][0] for r in g.itertuples(index=False)]
        anym = [r.msc in truth[r.proof_id] for r in g.itertuples(index=False)]
        per_area: dict[str, list[bool]] = {}
        for r, ok in zip(g.itertuples(index=False), top1):
            per_area.setdefault(truth[r.proof_id][0], []).append(ok)
        report[judge] = {
            "n": len(g),
            "coverage": len(g) / len(truth) if truth else float("nan"),
            "top1_accuracy": sum(top1) / len(top1) if top1 else float("nan"),
            "any_match_accuracy": sum(anym) / len(anym) if anym else float("nan"),
            "per_area_top1": {a: {"accuracy": sum(v) / len(v), "n": len(v)} for a, v in sorted(per_area.items())},
        }
    return report


def score_synthetic(labels: pd.DataFrame, steps: pd.DataFrame) -> dict:
    """Step-level accuracy on synthetic rows, whose `true_area` is the generating code."""
    if "true_area" not in steps:
        return {}
    truth = steps.dropna(subset=["true_area"])[["proof_id", "step", "true_area"]]
    truth = truth.assign(true_area=truth.true_area.map(msc.normalize)).dropna()
    m = labels[labels.step >= 0].merge(truth, on=["proof_id", "step"])
    return {j: {"n": int(len(g)), "accuracy": float((g.msc == g.true_area).mean())}
            for j, g in m.groupby("judge")}


# --------------------------------------------------------------------------- command line


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".jsonl", ".json"):
        return pd.read_json(p, lines=p.suffix == ".jsonl")
    return pd.read_csv(p, dtype=str)


def _sample(proofs: list[Proof], n: int | None, seed: int) -> list[Proof]:
    if n is None or n >= len(proofs):
        return proofs
    return random.Random(seed).sample(proofs, n)


def _load_proofs(args: argparse.Namespace, cfg: JudgeConfig) -> list[Proof]:
    if getattr(args, "known", None):
        return proofs_from_known(_read_table(args.known), cfg.max_state_chars)
    steps = _read_table(args.steps)
    proofs = proofs_from_steps(steps, cfg.max_steps_per_proof, cfg.max_state_chars)
    return _sample(proofs, args.sample, args.seed)


def main(argv: list[str] | None = None, client: Any = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m markov_lean.judge")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_input(p: argparse.ArgumentParser) -> None:
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--steps", help="a `steps` table (parquet, csv or jsonl)")
        src.add_argument("--known", help="statements with known MSC codes: columns id, statement, msc")
        p.add_argument("--sample", type=int, help="judge a random sample of this many proofs")
        p.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("estimate", help="print the cost of judging a steps table; spends nothing")
    add_input(e)
    es = sub.add_parser("estimate-synthetic", help="cost for placeholder proofs of a given size")
    es.add_argument("--n-proofs", type=int, default=1000)
    es.add_argument("--mean-steps", type=float, default=10)
    es.add_argument("--mean-state-chars", type=int, default=600)
    s = sub.add_parser("submit", help="estimate, check the cap, then create Message Batches")
    add_input(s)
    s.add_argument("--out", required=True, help="run directory for requests, manifest and batch ids")
    s.add_argument("--max-dollars", type=float, help="hard cap; the lower of this and the config cap applies")
    s.add_argument("--dry-run", action="store_true", help="write the requests and stop before any API call")
    c = sub.add_parser("collect", help="wait for the run's batches and write labels.parquet")
    c.add_argument("--out", required=True)
    c.add_argument("--no-wait", action="store_true")
    a = sub.add_parser("agree", help="agreement statistics between two judges")
    a.add_argument("--labels", required=True)
    a.add_argument("--judges", nargs=2, help="default: the two judges in the config")
    a.add_argument("--consensus-out", help="also write consensus labels here")
    v = sub.add_parser("validate", help="score proof-level labels against statements with known codes")
    v.add_argument("--labels", required=True)
    v.add_argument("--known", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "estimate":
        reqs, man = build_requests(_load_proofs(args, cfg), cfg)
        print(estimate_cost(reqs, man, cfg).summary())
    elif args.cmd == "estimate-synthetic":
        proofs = synthetic_proofs(args.n_proofs, args.mean_steps, args.mean_state_chars)
        reqs, man = build_requests(proofs, cfg)
        print(estimate_cost(reqs, man, cfg).summary())
    elif args.cmd == "submit":
        reqs, man = build_requests(_load_proofs(args, cfg), cfg)
        try:
            submit(reqs, man, cfg, args.out, client=client, max_dollars=args.max_dollars, dry_run=args.dry_run)
        except BudgetExceeded as exc:
            print(exc, file=sys.stderr)
            return 2
    elif args.cmd == "collect":
        out = Path(args.out)
        batch_ids = json.loads((out / "batches.json").read_text())["batch_ids"]
        manifest = json.loads((out / "manifest.json").read_text())
        client = client or make_client()
        if not args.no_wait:
            wait(client, batch_ids, cfg.poll_seconds, cfg.timeout_hours)
        labels, issues = collect(client, batch_ids, manifest)
        labels.to_parquet(out / "labels.parquet", index=False)
        (out / "issues.json").write_text(json.dumps(issues, indent=1))
        counts: dict[str, int] = {}
        for i in issues:
            counts[i["issue"]] = counts.get(i["issue"], 0) + 1
        print(f"{len(labels)} labels written to {out / 'labels.parquet'}; issues: {counts or 'none'}")
    elif args.cmd == "agree":
        labels = _read_table(args.labels)
        ja, jb = args.judges or cfg.models[:2]
        print(json.dumps(agreement(labels, ja, jb), indent=1))
        if args.consensus_out:
            consensus(labels, ja, jb).to_parquet(args.consensus_out, index=False)
    elif args.cmd == "validate":
        print(json.dumps(score_known(_read_table(args.labels), _read_table(args.known)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
