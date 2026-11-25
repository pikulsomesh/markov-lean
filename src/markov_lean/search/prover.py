"""Tactic proposals from DeepSeek-Prover-V1.5 served by vLLM's OpenAI-compatible server.

DeepSeek-Prover-V1.5 is a whole-proof model. Its own tree search (RMaxTS, in
``prover/algorithms/rmax_tree_search.py`` of deepseek-ai/DeepSeek-Prover-V1.5) gets
step-level proposals by "truncate and resume": the prompt is the theorem, the proof so far,
and a ``/- tactic state: ... -/`` comment with the current goals, and the model continues
from there. We do the same and keep only the first tactic of each continuation.

The prompt prefix is copied from ``prover/utils.py`` (``non_cot_prompt`` and
``cot_prompt``); the state comment layout from ``prover/lean/proof.py``.

Only ``httpx`` is used, against ``POST {base_url}/v1/completions``.
"""
from __future__ import annotations

import logging
import math
import os
import re
import textwrap
import time
from dataclasses import dataclass, field

import httpx

DEEPSEEK_HEADER = (
    "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)
PROMPT_PREFIX = {
    "non_cot": "Complete the following Lean 4 code:\n\n```lean4\n",
    "cot": (
        "Complete the following Lean 4 code with explanatory comments preceding each line "
        "of code:\n\n```lean4\n"
    ),
}
DEFAULT_MODEL = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
INDENT = "  "

# vLLM answers 400 "This model's maximum context length is 4096 tokens. However, you requested ..."
_CONTEXT_ERROR = re.compile(r"maximum context length|context length|context window", re.I)
log = logging.getLogger(__name__)

_STOP_LINE = re.compile(r"^(theorem|lemma|example|end|namespace|section|#)\b")


@dataclass
class Candidate:
    """One proposed tactic. ``logprob`` sums the token log-probabilities of the
    continuation up to the end of the tactic (any comment the model wrote first included);
    duplicates are merged with log-sum-exp and counted in ``count``."""

    tactic: str
    logprob: float
    n_tokens: int
    count: int = 1
    raw: list[str] = field(default_factory=list)


def normalize_statement(statement: str) -> str:
    """``... :=`` or ``... := by`` (DeepSeek's files have both) -> ``... := by\\n``."""
    s = statement.rstrip()
    if s.endswith(":= by"):
        return s + "\n"
    if s.endswith(":="):
        return s + " by\n"
    raise ValueError(f"statement must end with ':=' or ':= by': {statement[-60:]!r}")


def state_comment(state_text: str, indent: str = INDENT) -> str:
    nl = "\n" + indent
    return nl.join([indent + "/- tactic state:", INDENT + state_text.replace("\n", nl + INDENT), "-/\n"])


def build_prompt(
    statement: str,
    tactics: list[str],
    state_text: str,
    header: str = DEEPSEEK_HEADER,
    informal_prefix: str = "",
    mode: str = "non_cot",
    with_state: bool = True,
) -> str:
    """Prompt asking for the next tactic after ``tactics`` at goal state ``state_text``."""
    body = "".join(textwrap.indent(t.rstrip("\n"), INDENT, lambda _l: True) + "\n" for t in tactics)
    prompt = PROMPT_PREFIX[mode] + header + informal_prefix + normalize_statement(statement) + body
    if with_state:
        prompt += state_comment(state_text)
    return prompt


def first_tactic(text: str) -> tuple[str, int] | None:
    """First tactic in a model continuation, dedented, and the character offset in
    ``text`` where it ends. Skips leading blank lines and comments; a tactic runs on over
    lines indented deeper than its first line."""
    cut = text.find("```")
    if cut >= 0:
        text = text[:cut]
    lines = text.splitlines(keepends=True)
    pos = 0
    i = 0
    in_block = False
    while i < len(lines):
        s = lines[i].strip()
        if in_block:
            if "-/" in s:
                in_block = False
        elif not s or s.startswith("--"):
            pass
        elif s.startswith("/-"):
            in_block = "-/" not in s[2:]
        else:
            break
        pos += len(lines[i])
        i += 1
    if i == len(lines):
        return None
    first = lines[i]
    if _STOP_LINE.match(first.strip()):
        return None
    base = len(first) - len(first.lstrip(" "))
    block = [first]
    end = pos + len(first)
    for line in lines[i + 1 :]:
        if not line.strip():
            break
        if len(line) - len(line.lstrip(" ")) <= base:
            break
        block.append(line)
        end += len(line)
    tactic = textwrap.dedent("".join(block)).strip()
    # Drop a trailing line comment on a one-line tactic.
    if "\n" not in tactic and " --" in tactic:
        tactic = tactic.split(" --", 1)[0].rstrip()
    if not tactic:
        return None
    return tactic, end_offset(text, end)


def end_offset(text: str, end: int) -> int:
    """Offset just past the tactic's last non-space character."""
    while end > 0 and text[end - 1].isspace():
        end -= 1
    return end


def prefix_logprob(logprobs: dict | None, end: int) -> tuple[float, int]:
    """Sum of token log-probabilities for tokens starting before character ``end`` of the
    generated text, and how many there were. vLLM's ``text_offset`` may count from the start
    of the prompt, so offsets are taken relative to the first token."""
    if not logprobs:
        return 0.0, 0
    lps = logprobs.get("token_logprobs") or []
    offsets = logprobs.get("text_offset")
    if not offsets or len(offsets) != len(lps):
        toks = logprobs.get("tokens") or []
        offsets, acc = [], 0
        for t in toks:
            offsets.append(acc)
            acc += len(t)
    o0 = offsets[0] if offsets else 0
    total, n = 0.0, 0
    for off, lp in zip(offsets, lps):
        if off - o0 >= end:
            break
        if lp is not None and math.isfinite(lp):
            total += lp
        n += 1
    return total, n


def _logaddexp(a: float, b: float) -> float:
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def parse_choices(choices: list[dict]) -> list[Candidate]:
    """Turn completion choices into de-duplicated candidates, best first."""
    by_tactic: dict[str, Candidate] = {}
    for ch in choices:
        text = ch.get("text") or ""
        got = first_tactic(text)
        if got is None:
            continue
        tactic, end = got
        lp, n = prefix_logprob(ch.get("logprobs"), end)
        c = by_tactic.get(tactic)
        if c is None:
            by_tactic[tactic] = Candidate(tactic=tactic, logprob=lp, n_tokens=n, raw=[text])
        else:
            c.logprob = _logaddexp(c.logprob, lp)
            c.count += 1
            c.raw.append(text)
    return sorted(by_tactic.values(), key=lambda c: c.logprob, reverse=True)


class ProverClient:
    """Samples tactic candidates from a vLLM OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = DEFAULT_MODEL,
        n: int = 16,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 256,
        mode: str = "non_cot",
        header: str = DEEPSEEK_HEADER,
        timeout: float = 120.0,
        retries: int = 3,
        api_key: str | None = None,
        seed: int | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if mode not in PROMPT_PREFIX:
            raise ValueError(f"mode must be one of {sorted(PROMPT_PREFIX)}")
        self.model = model
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.mode = mode
        self.header = header
        self.retries = retries
        self.seed = seed
        api_key = api_key or os.environ.get("VLLM_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ProverClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def models(self) -> list[str]:
        r = self._client.get("/v1/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def complete(self, prompt: str, n: int | None = None) -> list[dict]:
        body = {
            "model": self.model,
            "prompt": prompt,
            "n": n or self.n,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "logprobs": 1,
            "stop": ["```", "/- tactic state"],
        }
        if self.seed is not None:
            body["seed"] = self.seed
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post("/v1/completions", json=body)
                if r.status_code == 400 and _CONTEXT_ERROR.search(r.text):
                    # A deep node's prompt can outgrow the context. That node gets no
                    # candidates; raising would fail the whole problem, and on resume
                    # fail it again, until the worker gives up.
                    log.warning("prompt too long for the model's context; no candidates: %s", r.text[:200])
                    return []
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"server error {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                return r.json()["choices"]
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                    raise
                last = e
                if attempt < self.retries:
                    time.sleep(min(2.0**attempt, 10.0))
        assert last is not None
        raise last

    def propose(
        self,
        statement: str,
        tactics: list[str],
        state_text: str,
        informal_prefix: str = "",
        header: str | None = None,
        n: int | None = None,
    ) -> list[Candidate]:
        prompt = build_prompt(
            statement,
            tactics,
            state_text,
            header=self.header if header is None else header,
            informal_prefix=informal_prefix,
            mode=self.mode,
        )
        return parse_choices(self.complete(prompt, n))


BACKENDS = ("vllm", "llama_cpp")
# Settings only the HTTP client uses; the local backend ignores them, so one config file
# (with search.yaml's prover section merged in) works for both.
_HTTP_ONLY = ("base_url", "timeout", "retries", "api_key", "client")


def make_proposer(config: dict | None = None, seed: int | None = None):
    """The tactic proposer a ``prover`` config section asks for.

    ``backend: vllm`` (the default) is :class:`ProverClient` against a vLLM server;
    ``backend: llama_cpp`` is :class:`~markov_lean.search.local_prover.LlamaCppProver`
    on a local GGUF file (``model_path``), for smoke tests on a Mac. ``seed``, when given,
    overrides the config's.
    """
    cfg = dict(config or {})
    backend = cfg.pop("backend", None) or "vllm"
    cfg.pop("factory", None)
    if seed is not None:
        cfg["seed"] = seed
    if backend == "vllm":
        cfg.pop("model_path", None)
        for k in ("n_ctx", "n_gpu_layers", "n_threads", "n_batch", "verbose", "early_stop"):
            cfg.pop(k, None)
        return ProverClient(**cfg)
    if backend == "llama_cpp":
        from markov_lean.search.local_prover import LlamaCppProver

        for k in _HTTP_ONLY:
            cfg.pop(k, None)
        if not cfg.get("model_path"):
            raise ValueError("backend llama_cpp needs prover.model_path (or --model-path), a GGUF file")
        return LlamaCppProver(**cfg)
    raise ValueError(f"unknown prover backend {backend!r}; use one of {list(BACKENDS)}")
