"""Turn goal-state text into vectors.

The default featurizer is TF-IDF over Lean identifier tokens followed by a
truncated SVD.  An identifier such as ``Nat.Prime.two_le`` contributes:

* the full name and each namespace prefix (``Nat``, ``Nat.Prime``,
  ``Nat.Prime.two_le``), so shared namespaces pull states together, and
* its parts split on dots and underscores (``Prime``, ``two``, ``le``).

Single-character names (``x``, ``h``) and pure numbers are dropped.

``view`` picks which part of the goal state is read (``state_view``):

* ``"full"`` (default): the whole pretty-printed state, hypotheses included;
* ``"typed"``: the hypotheses' types and the goals, without binder names
  (``hx : 0 < x`` reads as ``0 < x``) or ``case`` tags;
* ``"goal"``: only the goals after ``⊢``.

On LeanDojo about 87% of a state's characters are hypotheses, and binder
names such as ``hfp`` or ``hmin`` are mostly specific to one proof, so the
full view clusters goal states by proof rather than by area.  Other
featurizers, such as a sentence-embedding model, plug in through
``register_featurizer`` and ``make_featurizer``; none of them may need the
network at import time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Lean identifiers: letters (including unicode letters), digits, _, ', and dots
# between parts.  `⊢`, `:`, `=` and other notation are skipped.
_IDENT = re.compile(r"[^\W\d][\w']*(?:\.[^\W\d][\w']*)*", re.UNICODE)


def lean_tokens(text: str) -> list[str]:
    """Split a goal state into identifier tokens, keeping namespace prefixes."""
    out: list[str] = []
    for ident in _IDENT.findall(text):
        parts = ident.split(".")
        if len(ident) <= 1:
            continue
        # namespace prefixes and the full name
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            if len(prefix) > 1:
                out.append(prefix)
        # sub-parts split on dots and underscores
        if len(parts) > 1 or "_" in ident:
            for piece in re.split(r"[._]", ident):
                if len(piece) > 1 and not piece.isdigit():
                    out.append(piece.lower())
    return out


_CASE = re.compile(r"^case(\s|$)")
# a hypothesis line: one or more binder names, then " : " and the type
_BINDERS = re.compile(r"^(?:[^\s:]+ )*[^\s:]+ : ")
VIEWS = ("full", "typed", "goal")


def state_view(text: str, view: str = "full") -> str:
    """The part of a pretty-printed goal state that ``view`` reads (see the module docstring).

    Continuation lines (indented) stay with the hypothesis or goal they belong to.
    """
    if view == "full":
        return text
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; known: {VIEWS}")
    out: list[str] = []
    keep = False
    for line in text.split("\n"):
        if not line.strip():
            keep = False
            continue
        if line[0].isspace():  # continuation of the previous hypothesis or goal
            if keep:
                out.append(line)
            continue
        if line.startswith("⊢"):
            out.append(line[1:])
            keep = True
        elif _CASE.match(line):
            keep = False
        elif view == "typed":
            out.append(_BINDERS.sub("", line, count=1))
            keep = True
        else:
            keep = False
    return "\n".join(out)


@dataclass(frozen=True)
class ViewTokens:
    """Picklable analyzer: ``lean_tokens`` of one ``state_view``."""

    view: str = "full"

    def __call__(self, text: str) -> list[str]:
        return lean_tokens(state_view(text, self.view))


class Featurizer(Protocol):
    """Anything that maps a list of goal states to an ``(n, d)`` float array."""

    def fit_transform(self, texts: list[str]) -> np.ndarray: ...

    def transform(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class TfidfSvdFeaturizer:
    """TF-IDF over identifier tokens of one ``view`` of the state, then TruncatedSVD, then L2 row normalization."""

    n_components: int = 64
    min_df: int | float = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    seed: int = 0
    view: str = "full"

    def __post_init__(self) -> None:
        if self.view not in VIEWS:
            raise ValueError(f"unknown view {self.view!r}; known: {VIEWS}")
        self.vectorizer = TfidfVectorizer(
            analyzer=lean_tokens if self.view == "full" else ViewTokens(self.view),
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
        )
        self.svd: TruncatedSVD | None = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.fit_transform(texts)
        k = max(1, min(self.n_components, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed)
        Z = self.svd.fit_transform(X)
        return normalize(Z).astype(np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("call fit_transform first")
        return normalize(self.svd.transform(self.vectorizer.transform(texts))).astype(np.float32)

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)


@dataclass
class SentenceEmbeddingFeaturizer:
    """Hook for a sentence-embedding model (not a dependency).

    Needs ``sentence-transformers`` and a model already in the local cache;
    nothing here downloads at import time.  Embeddings are L2-normalized so
    k-means behaves like cosine clustering, as with the TF-IDF featurizer.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    device: str | None = None

    def _model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - optional
            raise ImportError("pip install sentence-transformers to use this featurizer") from e
        if not hasattr(self, "_m"):
            self._m = SentenceTransformer(self.model_name, device=self.device)
        return self._m

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - optional
        Z = self._model().encode(list(texts), batch_size=self.batch_size, show_progress_bar=False)
        return normalize(np.asarray(Z)).astype(np.float32)


_REGISTRY: dict[str, Callable[..., Featurizer]] = {
    "tfidf_svd": TfidfSvdFeaturizer,
    "sentence_embedding": SentenceEmbeddingFeaturizer,
}


def register_featurizer(name: str, factory: Callable[..., Featurizer]) -> None:
    """Add a featurizer so configs can pick it by name."""
    _REGISTRY[name] = factory


def make_featurizer(name: str = "tfidf_svd", **kwargs) -> Featurizer:
    try:
        return _REGISTRY[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown featurizer {name!r}; known: {sorted(_REGISTRY)}") from None
