"""Tactic proposals from a local GGUF of DeepSeek-Prover-V1.5 through llama.cpp.

For smoke tests on Apple silicon (Mac Mini M2, 16 GB) before GPU time: a 4-bit GGUF
(Q4_K_M, about 4.2 GB) runs on the M2's GPU through Metal. It is a stand-in for the vLLM
server in :mod:`~markov_lean.search.prover`, not a replacement for the paper's runs: the
weights are quantized and sampling uses another random number stream, so numbers from it
are not comparable with vLLM runs. Select it with ``prover.backend: llama_cpp`` in the
config or ``--backend llama_cpp`` on the command line.

What matches the vLLM path:

* the prompt (:func:`~markov_lean.search.prover.build_prompt`), the stop strings and the
  first-tactic parsing and de-duplication (:func:`~markov_lean.search.prover.parse_choices`);
* sampling: plain temperature and top-p over the full vocabulary, with no top-k, min-p or
  repetition penalty (llama.cpp's high-level defaults would add all three);
* log-probabilities: of the raw model distribution (before temperature and top-p), as
  vLLM returns them, summed over the tokens of the first tactic;
* seeds: sample ``i`` of a request with seed ``s`` draws from ``numpy.random.default_rng
  ([s, i])``, so the same goal state gets the same samples under every prior condition.

What differs, on purpose:

* generation stops as soon as the first tactic is complete (the rest of a continuation is
  thrown away by the parser anyway), which saves most of the generation time;
* the ``n`` samples of a request run one after another, reusing the KV cache of the
  prompt; consecutive prompts share their header and statement, which is reused too;
* a prompt that does not fit the context gives no candidates (and a warning) instead of
  an error.

Only a small surface of ``llama_cpp.Llama`` is used (``tokenize``, ``detokenize``,
``eval``, ``n_tokens``, ``input_ids`` and the C call ``llama_get_logits_ith``), wrapped
in :class:`LlamaCppModel`, so tests swap in a fake model and CI needs no model or
``llama_cpp``. Install with ``pip install llama-cpp-python`` (see MAC_SMOKE.md).
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Protocol, Sequence

import numpy as np

from markov_lean.search.prover import (
    DEEPSEEK_HEADER,
    DEFAULT_MODEL,
    PROMPT_PREFIX,
    Candidate,
    build_prompt,
    first_tactic,
    parse_choices,
)

log = logging.getLogger(__name__)

STOP = ("```", "/- tactic state")
# A line deeper than any tactic, appended to probe whether the first tactic is finished.
_PROBE = " " * 256 + "x"


class LocalModel(Protocol):
    """What :class:`LlamaCppProver` needs from a model."""

    n_ctx: int
    eos: int

    def tokenize(self, text: str) -> list[int]: ...

    def detokenize(self, tokens: Sequence[int], prev: Sequence[int] = ()) -> str: ...

    def feed(self, tokens: Sequence[int]) -> np.ndarray:
        """Make the context exactly ``tokens`` (reusing the longest cached prefix) and
        return the logits for the next token."""
        ...

    def close(self) -> None: ...


class LlamaCppModel:
    """Thin wrapper over ``llama_cpp.Llama`` holding one sequence in its KV cache."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        n_batch: int = 512,
        verbose: bool = False,
    ) -> None:
        try:
            import llama_cpp
        except ImportError as e:  # pragma: no cover - depends on the machine
            raise ImportError(
                "the llama_cpp backend needs llama-cpp-python: pip install llama-cpp-python "
                "(see MAC_SMOKE.md)"
            ) from e
        path = os.path.expanduser(model_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"model file {path} not found (see MAC_SMOKE.md for the download)")
        self._lib = llama_cpp
        self._llm = llama_cpp.Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_batch=n_batch,
            logits_all=False,
            verbose=verbose,
        )
        self.n_ctx = int(self._llm.n_ctx())
        self.n_vocab = int(self._llm.n_vocab())
        self.eos = int(self._llm.token_eos())

    def tokenize(self, text: str) -> list[int]:
        return list(self._llm.tokenize(text.encode("utf-8"), add_bos=True))

    def detokenize(self, tokens: Sequence[int], prev: Sequence[int] = ()) -> str:
        out = self._llm.detokenize(list(tokens), prev_tokens=list(prev) or None)
        return out.decode("utf-8", errors="ignore")

    def feed(self, tokens: Sequence[int]) -> np.ndarray:
        llm = self._llm
        tokens = list(tokens)
        cached = llm.input_ids[: llm.n_tokens].tolist() if llm.n_tokens else []
        keep = 0
        for a, b in zip(cached, tokens):
            if a != b:
                break
            keep += 1
        keep = min(keep, len(tokens) - 1)   # always evaluate the last token for fresh logits
        llm.n_tokens = keep                 # eval() drops the KV cache from here on
        llm.eval(tokens[keep:])
        ptr = self._lib.llama_get_logits_ith(llm.ctx, -1)
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).astype(np.float64, copy=True)

    def close(self) -> None:
        close = getattr(self._llm, "close", None)
        if callable(close):
            close()


def load_model(model_path: str, **kwargs: Any) -> LocalModel:
    """The model behind :class:`LlamaCppProver`; tests replace this function."""
    return LlamaCppModel(model_path, **kwargs)


def sample_token(logits: np.ndarray, temperature: float, top_p: float,
                 rng: np.random.Generator) -> tuple[int, float]:
    """Draw a token with temperature and nucleus (top-p) sampling; return it with its
    log-probability under the raw distribution (temperature 1, no top-p), as vLLM does."""
    logits = np.asarray(logits, dtype=np.float64)
    m = float(np.max(logits))
    lse = m + math.log(float(np.sum(np.exp(logits - m))))
    if temperature <= 0:
        tok = int(np.argmax(logits))
        return tok, float(logits[tok] - lse)
    if temperature == 1.0:
        p = np.exp(logits - lse)
    else:
        z = logits / temperature
        p = np.exp(z - np.max(z))
        p /= p.sum()
    if top_p < 1.0:
        # The nucleus is the most probable tokens up to mass top_p. Sorting only the
        # tokens above a small threshold gives the same set whenever they already hold
        # that mass (almost always); otherwise sort everything.
        idx = np.flatnonzero(p >= 1e-6)
        if p[idx].sum() < top_p:
            idx = np.arange(len(p))
        order = idx[np.argsort(-p[idx], kind="stable")]
        cum = np.cumsum(p[order])
        k = min(int(np.searchsorted(cum, top_p, side="left")) + 1, len(order))
        keep = order[:k]
        q = p[keep] / p[keep].sum()
        tok = int(keep[rng.choice(k, p=q)])
    else:
        tok = int(rng.choice(len(p), p=p))
    return tok, float(logits[tok] - lse)


def tactic_complete(text: str) -> bool:
    """True once ``text`` holds a whole first tactic: more generated text could not change
    what :func:`~markov_lean.search.prover.first_tactic` returns, because the tactic's
    block has been closed by a blank line, a line indented no deeper, or a code fence."""
    got = first_tactic(text)
    if got is None:
        return False
    # A last line of only spaces is still being written (it may become a deeper
    # continuation line), so the probe extends it rather than starting a new line.
    last = text.rsplit("\n", 1)[-1]
    probe = text + ("" if not last.strip() else "\n") + _PROBE
    return first_tactic(probe) == got


class LlamaCppProver:
    """Same interface as :class:`~markov_lean.search.prover.ProverClient`, on a local model.

    ``model_path`` is the GGUF file. ``n_ctx`` is the context length (DeepSeek-Prover-V1.5
    was trained with 4096). ``n_gpu_layers=-1`` puts every layer on the GPU (Metal).
    ``model`` is only a label for logs.
    """

    def __init__(
        self,
        model_path: str,
        n: int = 16,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 256,
        mode: str = "non_cot",
        header: str = DEEPSEEK_HEADER,
        seed: int | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        n_batch: int = 512,
        model: str = DEFAULT_MODEL,
        verbose: bool = False,
        early_stop: bool = True,
        local_model: LocalModel | None = None,
    ) -> None:
        if mode not in PROMPT_PREFIX:
            raise ValueError(f"mode must be one of {sorted(PROMPT_PREFIX)}")
        self.model_path = model_path
        self.model = model
        self.n = n
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.mode = mode
        self.header = header
        self.seed = seed
        self.early_stop = early_stop
        self._model = local_model or load_model(
            model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads,
            n_batch=n_batch, verbose=verbose,
        )
        self._fresh = np.random.default_rng()
        self.stats = {"requests": 0, "samples": 0, "prompt_tokens": 0, "generated_tokens": 0,
                      "seconds": 0.0, "too_long": 0}

    def close(self) -> None:
        if self.stats["requests"]:
            s = self.stats
            log.info("local prover: %d requests, %d samples, %d tokens generated, %.1f s (%.1f tok/s)",
                     s["requests"], s["samples"], s["generated_tokens"], s["seconds"],
                     s["generated_tokens"] / max(s["seconds"], 1e-9))
        self._model.close()

    def __enter__(self) -> "LlamaCppProver":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def models(self) -> list[str]:
        return [f"{self.model} ({os.path.basename(self.model_path)}, llama.cpp)"]

    def _rng(self, i: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng(self._fresh.integers(2**63))
        return np.random.default_rng([int(self.seed) & 0xFFFFFFFF, i])

    def _sample(self, prompt_tokens: list[int], max_new: int, i: int) -> dict:
        rng = self._rng(i)
        gen: list[int] = []
        lps: list[float] = []
        offsets: list[int] = []
        text = ""
        logits = self._model.feed(prompt_tokens)
        while len(gen) < max_new:
            tok, lp = sample_token(logits, self.temperature, self.top_p, rng)
            if tok == self._model.eos:
                break
            offsets.append(len(text))
            gen.append(tok)
            lps.append(lp)
            text = self._model.detokenize(gen, prompt_tokens)
            cut = min((text.find(s) for s in STOP if s in text), default=-1)
            if cut >= 0:
                text = text[:cut]
                break
            if self.early_stop and tactic_complete(text):
                break
            if len(gen) < max_new:
                logits = self._model.feed(prompt_tokens + gen)
        keep = sum(1 for o in offsets if o < len(text)) if text else 0
        self.stats["generated_tokens"] += len(gen)
        return {"text": text, "logprobs": {"tokens": [""] * keep, "token_logprobs": lps[:keep],
                                           "text_offset": offsets[:keep]}}

    def complete(self, prompt: str, n: int | None = None) -> list[dict]:
        """``n`` samples as completion choices in vLLM's response format."""
        t0 = time.monotonic()
        n = n or self.n
        tokens = self._model.tokenize(prompt)
        self.stats["requests"] += 1
        max_new = min(self.max_tokens, self._model.n_ctx - len(tokens))
        if max_new <= 0:
            self.stats["too_long"] += 1
            log.warning("prompt of %d tokens does not fit the %d-token context; no candidates",
                        len(tokens), self._model.n_ctx)
            return []
        self.stats["prompt_tokens"] += len(tokens)
        choices = [self._sample(tokens, max_new, i) for i in range(n)]
        self.stats["samples"] += n
        self.stats["seconds"] += time.monotonic() - t0
        return choices

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
