"""A third MSC judge that runs locally through Ollama, for a cross-family agreement check.

The two API judges (Claude Haiku and Sonnet, ``markov_lean.judge``) share a
model family, so their agreement can overstate how reproducible the labels
are. This module asks an open model served by Ollama on the Mac Mini to label
the same proofs with the **same system prompt, user prompt, output schema and
MSC label set**, and writes its answers as a ``labels`` table
(``docs/schema.md``) that ``markov-lean judge-agreement`` and
``markov-lean significance --extra-labels`` read next to the API labels.

It talks to Ollama's local HTTP API (``POST /api/chat``) with the standard
library only, so it needs no new dependency, makes no Anthropic call and costs
nothing. Structured outputs (``format`` set to the JSON schema, Ollama 0.5 or
newer) constrain decoding to the schema; on older servers it falls back to
``format: "json"`` and the JSON repair below.

Commands (``markov-lean local-judge <command>`` or
``python -m markov_lean.ollama_judge <command>``):

    check   is the server up, is the model pulled, and how fast is one proof
    label   label a stratified sample of proofs (``--sample-steps``, default
            500 steps) or every proof (``--all``); resumable, re-run the same
            command after a stop and finished proofs are skipped
    export  rewrite labels.parquet, issues.json and summary.json from the
            progress log without calling the server

The sample is drawn from proofs, because the judges see a whole proof in one
request. Proofs are stratified (by the API judges' proof-level label when
``--labels`` is given, otherwise by Mathlib folder, otherwise by source) and
taken in an interleaved order, so a run stopped halfway is still balanced.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from markov_lean import judge as J
from markov_lean.evaluate import folder_label

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
JUDGE_PREFIX = "ollama/"
PROGRESS = "progress.jsonl"
SAMPLE = "sample.json"


def judge_name(model: str) -> str:
    """The ``judge`` value written to the labels table, e.g. ``ollama/qwen2.5:7b-instruct``."""
    return model if model.startswith(JUDGE_PREFIX) else JUDGE_PREFIX + model


# --------------------------------------------------------------------------- HTTP client


class OllamaError(RuntimeError):
    pass


class RetryableError(OllamaError):
    pass


@dataclass
class OllamaClient:
    """A minimal client for the parts of Ollama's API the judge needs."""

    host: str = DEFAULT_HOST
    timeout: float = 600.0

    def _url(self, path: str) -> str:
        return self.host.rstrip("/") + path

    def _request(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self._url(path), data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code >= 500 or e.code == 429:
                raise RetryableError(f"HTTP {e.code} from {path}: {detail}") from e
            raise OllamaError(f"HTTP {e.code} from {path}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise RetryableError(f"cannot reach Ollama at {self.host} ({e})") from e
        except json.JSONDecodeError as e:
            raise RetryableError(f"non-JSON reply from {path}") from e

    def version(self) -> str:
        return str(self._request("/api/version").get("version", "0"))

    def models(self) -> list[str]:
        return [m.get("name") or m.get("model") for m in self._request("/api/tags").get("models", [])]

    def chat(self, body: dict) -> dict:
        return self._request("/api/chat", {**body, "stream": False})


def has_model(available: list[str], model: str) -> bool:
    want = model if ":" in model else model + ":latest"
    return any(m == model or m == want for m in available)


def supports_schema(version: str) -> bool:
    """Structured outputs (a JSON schema in ``format``) arrived in Ollama 0.5.0."""
    try:
        parts = [int(x) for x in version.split("-")[0].split(".")[:2]]
    except ValueError:
        return True  # a dev build; assume current
    return tuple(parts + [0])[:2] >= (0, 5)


# --------------------------------------------------------------------------- JSON repair


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _outer_object(text: str) -> str | None:
    """The first balanced ``{...}`` in ``text``, respecting strings. None if it never closes."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _drop_trailing_commas(text: str) -> str:
    out, in_str, esc = [], False, False
    for i, c in enumerate(text):
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        if c == ",":
            j = i + 1
            while j < len(text) and text[j] in " \t\r\n":
                j += 1
            if j < len(text) and text[j] in "}]":
                continue
        out.append(c)
    return "".join(out)


def parse_json(text: str | None) -> dict | None:
    """Parse a judge answer, repairing what small models get wrong: code fences, chatter
    around the object and trailing commas. Returns None when it cannot be repaired."""
    if not text:
        return None
    for candidate in (text, _strip_fences(text)):
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    obj = _outer_object(_strip_fences(text))
    if obj is None:
        return None
    for candidate in (obj, _drop_trailing_commas(obj)):
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
    return None


# --------------------------------------------------------------------------- one proof


@dataclass
class LocalSettings:
    model: str = DEFAULT_MODEL
    num_ctx: int = 32768          # Qwen2.5 7B and Llama 3.1 8B both take 32K; pilot prompts peak near 24K tokens
    temperature: float = 0.0
    seed: int = 0
    retries: int = 3              # extra attempts after a failed one (server errors, bad JSON, cut-off answers)
    backoff_seconds: float = 2.0
    keep_alive: str = "30m"
    chars_per_token: float = 3.0  # same heuristic as the API estimate; errs high for Lean
    schema: bool = True           # False on Ollama < 0.5: plain JSON mode plus repair

    def num_predict(self, n_steps: int) -> int:
        return min(4096, 128 + 48 * max(1, n_steps))


@dataclass
class ProofResult:
    proof_id: str
    rows: list[dict]
    issue: dict | None
    attempts: int
    seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0

    @property
    def status(self) -> str:
        if self.issue is None:
            return "ok"
        return "partial" if self.issue["issue"] == "partial" else self.issue["issue"]

    def to_json(self) -> dict:
        return {"proof_id": self.proof_id, "status": self.status, "rows": self.rows, "issue": self.issue,
                "attempts": self.attempts, "seconds": round(self.seconds, 2),
                "prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens}


def chat_body(proof: J.Proof, s: LocalSettings, attempt: int = 0, num_predict: int | None = None) -> dict:
    """The /api/chat request: the API judges' prompt and schema, deterministic decoding."""
    return {
        "model": s.model,
        "messages": [{"role": "system", "content": J.SYSTEM_PROMPT},
                     {"role": "user", "content": J.render_user(proof)}],
        "format": J.output_schema() if s.schema else "json",
        "options": {
            "temperature": s.temperature if attempt == 0 else max(s.temperature, 0.3),
            "seed": s.seed + attempt,
            "num_ctx": s.num_ctx,
            "num_predict": num_predict or s.num_predict(len(proof.steps)),
        },
        "keep_alive": s.keep_alive,
    }


def estimated_prompt_tokens(proof: J.Proof, s: LocalSettings) -> int:
    chars = len(J.SYSTEM_PROMPT) + len(J.render_user(proof)) + len(json.dumps(J.output_schema()))
    return math.ceil(chars / s.chars_per_token)


def label_proof(client: OllamaClient, proof: J.Proof, s: LocalSettings,
                sleep: Callable[[float], None] = time.sleep) -> ProofResult:
    """Label one proof, retrying server errors, unparseable answers and answers cut off by length."""
    name = judge_name(s.model)
    entry = {"proof_id": proof.proof_id, "judge": name, "steps": [st.step for st in proof.steps]}
    base = {"proof_id": proof.proof_id, "judge": name}
    t0 = time.monotonic()
    budget = s.num_ctx - s.num_predict(len(proof.steps))
    est = estimated_prompt_tokens(proof, s)
    if est > budget:
        return ProofResult(proof.proof_id, [], {**base, "issue": "too_long",
                                                "detail": f"~{est} prompt tokens > {budget}; raise --num-ctx"},
                           0, 0.0)
    num_predict = s.num_predict(len(proof.steps))
    last_issue: dict | None = None
    best: tuple[list[dict], dict | None] | None = None
    ptok = otok = 0
    attempts = 0
    for attempt in range(s.retries + 1):
        attempts = attempt + 1
        try:
            resp = client.chat(chat_body(proof, s, attempt, num_predict))
        except RetryableError as e:
            last_issue = {**base, "issue": "server_error", "detail": str(e)[:300]}
            sleep(s.backoff_seconds * 2 ** attempt)
            continue
        ptok += int(resp.get("prompt_eval_count") or 0)
        otok += int(resp.get("eval_count") or 0)
        text = (resp.get("message") or {}).get("content")
        prompt_count = int(resp.get("prompt_eval_count") or 0)  # cached prefix tokens are not counted
        if prompt_count >= s.num_ctx - 1:
            # Ollama silently cuts prompts longer than num_ctx; a full-context count means it did
            return ProofResult(proof.proof_id, [], {**base, "issue": "too_long",
                                                    "detail": f"{prompt_count} prompt tokens filled the context"},
                               attempts, time.monotonic() - t0, ptok, otok)
        data = parse_json(text)
        if data is None:
            cut = resp.get("done_reason") == "length"
            last_issue = {**base, "issue": "max_tokens" if cut else "bad_json", "detail": (text or "")[:200]}
            if cut:
                num_predict = min(8192, num_predict * 2)
            continue
        rows, issue = J.rows_from_output(data, entry, base)
        if issue is None:
            return ProofResult(proof.proof_id, rows, None, attempts, time.monotonic() - t0, ptok, otok)
        if best is None or len(rows) > len(best[0]):
            best = (rows, issue)
        last_issue = issue
    if best is not None:  # keep the most complete partial answer
        return ProofResult(proof.proof_id, best[0], best[1], attempts, time.monotonic() - t0, ptok, otok)
    return ProofResult(proof.proof_id, [], last_issue, attempts, time.monotonic() - t0, ptok, otok)


# --------------------------------------------------------------------------- stratified sample


def proof_strata(steps: pd.DataFrame, labels: pd.DataFrame | None = None, by: str = "auto") -> dict[str, str]:
    """One stratum per proof: the API judges' proof label, the Mathlib folder, or the source."""
    first = steps.sort_values(["proof_id", "step"]).groupby("proof_id", sort=False).first()
    ids = first.index.astype(str)
    if by == "auto":
        by = "label" if labels is not None and len(labels) else "folder"
    if by == "none":
        return {p: "all" for p in ids}
    if by == "label":
        if labels is None or labels.empty:
            raise ValueError("--stratify-by label needs --labels")
        lab = labels[labels["judge"].astype(str) != J.CONSENSUS_JUDGE].copy()
        lab["proof_id"] = lab["proof_id"].astype(str)
        whole = lab[lab["step"] == J.PROOF_LEVEL_STEP]
        # proof-level label of the first judge present; else the most common step label
        judges = list(dict.fromkeys(lab["judge"].astype(str)))
        out: dict[str, str] = {}
        for j in judges:
            for r in whole[whole["judge"] == j].itertuples(index=False):
                out.setdefault(r.proof_id, str(r.msc))
        steps_lab = lab[lab["step"] >= 0]
        if len(steps_lab):
            mode = steps_lab.groupby("proof_id")["msc"].agg(lambda s: s.astype(str).mode().iloc[0])
            for p, m in mode.items():
                out.setdefault(p, m)
        return {p: out.get(p, "unlabelled") for p in ids}
    if by == "folder":
        fp = first["file_path"] if "file_path" in first else pd.Series(index=first.index, dtype=object)
        src = first["source"] if "source" in first else pd.Series("unknown", index=first.index)
        return {str(p): (folder_label(f) or f"source:{s}") for p, f, s in zip(first.index, fp, src)}
    if by == "source":
        src = first["source"] if "source" in first else pd.Series("unknown", index=first.index)
        return {str(p): str(s) for p, s in src.items()}
    raise ValueError(f"unknown stratification {by!r}")


def stratified_sample(proofs: list[J.Proof], strata: dict[str, str], n_steps: int | None,
                      seed: int = 0) -> list[J.Proof]:
    """Proofs whose shown steps add up to about ``n_steps``, allocated to strata in proportion
    to their steps (every stratum gets at least one proof while the budget allows), returned in
    an interleaved order so that any prefix of the list is itself close to stratified.
    ``n_steps=None`` keeps every proof, in the same interleaved order."""
    rng = random.Random(seed)
    groups: dict[str, list[J.Proof]] = {}
    for p in proofs:
        groups.setdefault(strata.get(p.proof_id, "unknown"), []).append(p)
    for g in groups.values():
        rng.shuffle(g)
    cost = {h: sum(len(p.steps) or 1 for p in g) for h, g in groups.items()}
    total_cost = sum(cost.values())
    budget = total_cost if n_steps is None else n_steps
    target = {h: budget * c / total_cost for h, c in cost.items()} if total_cost else {}
    taken = {h: 0 for h in groups}
    nxt = {h: 0 for h in groups}
    order: list[J.Proof] = []
    used = 0
    # one proof per stratum first, biggest strata first, so small areas are not left out
    for h in sorted(groups, key=lambda h: (-cost[h], h)):
        if used >= budget:
            break
        p = groups[h][0]
        order.append(p)
        nxt[h] = 1
        taken[h] += len(p.steps) or 1
        used += len(p.steps) or 1
    while used < budget:
        open_ = [h for h in groups if nxt[h] < len(groups[h])]
        if not open_:
            break
        h = max(open_, key=lambda h: (target[h] - taken[h], -nxt[h], h))
        p = groups[h][nxt[h]]
        nxt[h] += 1
        order.append(p)
        taken[h] += len(p.steps) or 1
        used += len(p.steps) or 1
    return order


# --------------------------------------------------------------------------- the run


@dataclass
class RunPaths:
    out: Path
    progress: Path = field(init=False)
    sample: Path = field(init=False)

    def __post_init__(self):
        self.out = Path(self.out)
        self.progress = self.out / PROGRESS
        self.sample = self.out / SAMPLE


def read_progress(path: Path) -> dict[str, dict]:
    """Last record per proof from the append-only progress log (a cut-off last line is ignored)."""
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[str(rec["proof_id"])] = rec
    return done


RETRY_ON_RESUME = {"server_error"}


def export(out: str | Path, model: str | None = None, echo: Callable[[str], None] = print) -> dict:
    """Write labels.parquet, issues.json and summary.json from the progress log."""
    paths = RunPaths(out)
    done = read_progress(paths.progress)
    rows = [r for rec in done.values() for r in rec.get("rows", [])]
    labels = J.labels_frame(rows)
    labels.to_parquet(paths.out / "labels.parquet", index=False)
    issues = [rec["issue"] for rec in done.values() if rec.get("issue")]
    (paths.out / "issues.json").write_text(json.dumps(issues, indent=1))
    sample = json.loads(paths.sample.read_text()) if paths.sample.exists() else {}
    counts: dict[str, int] = {}
    for rec in done.values():
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    secs = [rec["seconds"] for rec in done.values() if rec.get("seconds")]
    summary = {
        "judge": judge_name(model or sample.get("model", DEFAULT_MODEL)),
        "proofs_in_sample": len(sample.get("proof_ids", [])),
        "proofs_done": len(done),
        "status_counts": counts,
        "labels": int(len(labels)),
        "step_labels": int((labels["step"] >= 0).sum()),
        "proof_labels": int((labels["step"] == J.PROOF_LEVEL_STEP).sum()),
        "mean_seconds_per_proof": round(sum(secs) / len(secs), 2) if secs else None,
        "prompt_tokens": int(sum(rec.get("prompt_tokens", 0) for rec in done.values())),
        "output_tokens": int(sum(rec.get("output_tokens", 0) for rec in done.values())),
        "settings": sample.get("settings"),
    }
    (paths.out / "summary.json").write_text(json.dumps(summary, indent=1))
    echo(f"{len(labels)} labels ({summary['step_labels']} steps, {summary['proof_labels']} proofs) written to "
         f"{paths.out / 'labels.parquet'}; status: {counts or 'none'}")
    return summary


def plan_sample(steps: pd.DataFrame, cfg: J.JudgeConfig, labels: pd.DataFrame | None, stratify_by: str,
                n_steps: int | None, seed: int) -> list[J.Proof]:
    steps = steps.assign(proof_id=steps["proof_id"].astype(str))
    proofs = J.proofs_from_steps(steps, cfg.max_steps_per_proof, cfg.max_state_chars)
    if labels is not None and len(labels):
        labelled = set(labels["proof_id"].astype(str))
        within = [p for p in proofs if p.proof_id in labelled]
        if within:  # judge the proofs the API judges labelled, so every local label can be paired
            proofs = within
    strata = proof_strata(steps, labels, stratify_by)
    return stratified_sample(proofs, strata, n_steps, seed)


def run_label(steps: pd.DataFrame, out: str | Path, settings: LocalSettings, *, client: OllamaClient,
              cfg: J.JudgeConfig, labels: pd.DataFrame | None = None, stratify_by: str = "auto",
              n_steps: int | None = 500, fresh: bool = False, retry_failed: bool = False,
              limit_proofs: int | None = None, echo: Callable[[str], None] = print,
              sleep: Callable[[float], None] = time.sleep, progress_every: int = 5) -> dict:
    """Label a stratified sample (or everything with ``n_steps=None``) and write the labels table."""
    paths = RunPaths(out)
    paths.out.mkdir(parents=True, exist_ok=True)
    if fresh:
        for p in (paths.progress, paths.sample):
            p.unlink(missing_ok=True)

    version = client.version()
    available = client.models()
    if not has_model(available, settings.model):
        raise OllamaError(f"model {settings.model!r} is not pulled; run: ollama pull {settings.model}")
    if not supports_schema(version):
        settings.schema = False
        echo(f"Ollama {version} predates structured outputs (0.5); using JSON mode plus repair. "
             "Upgrading Ollama is better.")

    steps = steps.assign(proof_id=steps["proof_id"].astype(str))
    by_id = {p.proof_id: p for p in J.proofs_from_steps(steps, cfg.max_steps_per_proof, cfg.max_state_chars)}
    ids = [p.proof_id for p in plan_sample(steps, cfg, labels, stratify_by, n_steps, settings.seed)]
    if paths.sample.exists():
        old = json.loads(paths.sample.read_text())
        if old.get("model") != settings.model:
            raise OllamaError(f"{paths.out} holds labels from {old.get('model')}; use another --out or --fresh")
    # the plan is deterministic, so re-running the same command resumes the same sample; a changed
    # sample (a bigger --sample-steps, or API labels that arrived since) reuses every proof already done
    paths.sample.write_text(json.dumps({
        "model": settings.model, "n_steps": n_steps, "stratify_by": stratify_by, "seed": settings.seed,
        "proof_ids": ids, "settings": {k: v for k, v in vars(settings).items()}}))

    done = read_progress(paths.progress)
    skip = {p for p, rec in done.items()
            if rec["status"] not in RETRY_ON_RESUME and (rec["status"] == "ok" or not retry_failed)}
    todo = [i for i in ids if i not in skip]
    if limit_proofs is not None:
        todo = todo[:limit_proofs]
    n_steps_todo = sum(len(by_id[i].steps) for i in todo)
    echo(f"Local judge {judge_name(settings.model)} (Ollama {version}): {len(ids)} proofs in the sample, "
         f"{sum(i in skip for i in ids)} already done, {len(todo)} to label now ({n_steps_todo} steps).")

    t_start = time.monotonic()
    finished = 0
    try:
        with open(paths.progress, "a") as log:
            for i, pid in enumerate(todo, 1):
                res = label_proof(client, by_id[pid], settings, sleep=sleep)
                log.write(json.dumps(res.to_json()) + "\n")
                log.flush()
                finished += 1
                if i % progress_every == 0 or i == len(todo):
                    per = (time.monotonic() - t_start) / finished
                    left = per * (len(todo) - i)
                    echo(f"[{i}/{len(todo)}] {per:.1f} s per proof, about {left / 60:.0f} min left "
                         f"(last: {res.status})")
                if res.status == "server_error":
                    echo(f"Ollama kept failing on {pid}: {res.issue['detail']}. Stopping; re-run to resume.")
                    break
    except KeyboardInterrupt:  # pragma: no cover - manual stop
        echo("Stopped. Re-run the same command to resume.")
    return export(paths.out, settings.model, echo)


# --------------------------------------------------------------------------- check


def check(client: OllamaClient, settings: LocalSettings, steps: pd.DataFrame | None, cfg: J.JudgeConfig,
          echo: Callable[[str], None] = print) -> int:
    try:
        version = client.version()
    except RetryableError as e:
        echo(f"{e}. Start the Ollama app (or `ollama serve`) and try again.")
        return 1
    echo(f"Ollama {version} at {client.host}" + ("" if supports_schema(version)
                                                 else " (older than 0.5: no structured outputs; please upgrade)"))
    available = client.models()
    if not has_model(available, settings.model):
        echo(f"Model {settings.model} is not pulled. Run: ollama pull {settings.model}")
        echo(f"Pulled models: {', '.join(m for m in available if m) or 'none'}")
        return 1
    echo(f"Model {settings.model} is pulled.")
    if steps is None:
        return 0
    steps = steps.assign(proof_id=steps["proof_id"].astype(str))
    proofs = J.proofs_from_steps(steps, cfg.max_steps_per_proof, cfg.max_state_chars)
    if not proofs:
        return 0
    proofs.sort(key=lambda p: abs(len(p.steps) - 15))  # a typical pilot proof shows about 15 steps
    settings.schema = supports_schema(version)
    res = label_proof(client, proofs[0], settings)
    echo(f"Test proof {res.proof_id}: {res.status}, {len(res.rows)} labels, {res.seconds:.1f} s "
         f"({res.prompt_tokens} prompt + {res.output_tokens} output tokens; the first call includes loading "
         "the model).")
    return 0 if res.status in ("ok", "partial") else 1


# --------------------------------------------------------------------------- command line


def _default_steps(run: str | None) -> str | None:
    if run and (Path(run) / "judge" / "sample_steps.parquet").exists():
        return str(Path(run) / "judge" / "sample_steps.parquet")
    return None


def main(argv: list[str] | None = None, client: OllamaClient | None = None,
         sleep: Callable[[float], None] = time.sleep) -> int:
    ap = argparse.ArgumentParser(prog="markov-lean local-judge",
                                 description="Label proofs with a local Ollama model as a third MSC judge.")
    ap.add_argument("--config", default=str(J.DEFAULT_CONFIG), help="judge config (prompt limits)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama server (default {DEFAULT_HOST})")
        p.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model tag (default {DEFAULT_MODEL})")
        p.add_argument("--num-ctx", type=int, default=LocalSettings.num_ctx, help="context window in tokens")
        p.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for one answer")
        p.add_argument("--run", help="pilot run folder; steps default to <run>/judge/sample_steps.parquet")
        p.add_argument("--steps", help="a `steps` table (parquet, csv or jsonl)")

    c = sub.add_parser("check", help="is the server up and the model pulled; time one proof")
    common(c)
    lb = sub.add_parser("label", help="label a stratified sample (resumable)")
    common(lb)
    size = lb.add_mutually_exclusive_group()
    size.add_argument("--sample-steps", type=int, default=500, help="about this many proof steps (default 500)")
    size.add_argument("--all", action="store_true", help="label every proof in the steps table")
    lb.add_argument("--labels", help="the API judges' labels.parquet: judge their proofs, stratified by their "
                                     "proof label (default <run>/judge/labels.parquet when it exists)")
    lb.add_argument("--stratify-by", choices=["auto", "label", "folder", "source", "none"], default="auto")
    lb.add_argument("--out", help="output folder (default <run>/judge/local/<model>)")
    lb.add_argument("--seed", type=int, default=0)
    lb.add_argument("--retries", type=int, default=3)
    lb.add_argument("--limit-proofs", type=int, help="stop after this many proofs this session")
    lb.add_argument("--retry-failed", action="store_true", help="also redo proofs that ended with an issue")
    lb.add_argument("--fresh", action="store_true", help="discard the saved sample and progress in --out")
    ex = sub.add_parser("export", help="rewrite labels.parquet from the progress log (no server needed)")
    ex.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "export":
        export(args.out)
        return 0

    cfg = J.load_config(args.config)
    client = client or OllamaClient(args.host, args.timeout)
    settings = LocalSettings(model=args.model, num_ctx=args.num_ctx)
    steps_path = args.steps or _default_steps(args.run)

    if args.cmd == "check":
        return check(client, settings, J._read_table(steps_path) if steps_path else None, cfg)

    if not steps_path:
        ap.error("label needs --steps, or --run with judge/sample_steps.parquet")
    settings.seed, settings.retries = args.seed, args.retries
    labels_path = args.labels
    if labels_path is None and args.run and (Path(args.run) / "judge" / "labels.parquet").exists():
        labels_path = str(Path(args.run) / "judge" / "labels.parquet")
    labels = J._read_table(labels_path) if labels_path else None
    out = args.out
    if out is None:
        base = Path(args.run) / "judge" / "local" if args.run else Path("results") / "local_judge"
        out = base / args.model.replace(":", "_").replace("/", "_")
    try:
        run_label(J._read_table(steps_path), out, settings, client=client, cfg=cfg, labels=labels,
                  stratify_by=args.stratify_by, n_steps=None if args.all else args.sample_steps,
                  fresh=args.fresh, retry_failed=args.retry_failed, limit_proofs=args.limit_proofs, sleep=sleep)
    except OllamaError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
