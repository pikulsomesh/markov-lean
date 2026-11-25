"""Build the search's area prior from a saved Markov state model.

``markov-lean pilot`` saves the model under ``<out>/msm``; this module turns
that directory into the ``prior(state_text, candidates)`` callable that
``markov_lean.search`` takes.  In ``configs/search.yaml``::

    prior:
      factory: markov_lean.prior_loader:build_area_prior
      kwargs: {msm_dir: results/pilot2/msm, lam: 0.3}

Files read from ``msm_dir`` (written by ``MSMResult.save``):

* ``featurizer.pkl``: the fitted featurizer, so a new goal state lands in the
  same feature space the microstates were clustered in
* ``centers.npy``: k-means microstate centers, ``(n_micro, d)``
* ``memberships.npy``: PCCA+ memberships, ``(n_micro, n_macro)``, with NaN
  rows for microstates outside the connected set (read as "no information")
* ``coarse_T.npy``: coarse-grained transition matrix, ``(n_macro, n_macro)``

``featurizer.pkl`` is a pickle, and loading a pickle runs code from the file.
Load only a model directory you produced yourself.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .area_prior import AreaPrior, make_prior

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ("featurizer.pkl", "centers.npy", "memberships.npy")


@dataclass
class MSMArtifacts:
    featurizer: Any
    centers: np.ndarray
    memberships: np.ndarray
    coarse_T: np.ndarray | None
    report: dict


def _resolve(msm_dir: str | Path) -> Path:
    p = Path(msm_dir).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_msm_artifacts(msm_dir: str | Path) -> MSMArtifacts:
    """Read what the area prior needs from a saved model directory, checking shapes agree."""
    d = _resolve(msm_dir)
    missing = [f for f in REQUIRED if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{d} is missing {missing}. Rerun `markov-lean pilot` with the current code; "
            "older runs did not save the featurizer, centers and memberships.")
    with open(d / "featurizer.pkl", "rb") as f:
        featurizer = pickle.load(f)  # noqa: S301 - a file this project wrote; see module docstring
    centers = np.load(d / "centers.npy")
    memberships = np.load(d / "memberships.npy")
    coarse_T = np.load(d / "coarse_T.npy") if (d / "coarse_T.npy").exists() else None
    report = json.loads((d / "msm_report.json").read_text()) if (d / "msm_report.json").exists() else {}
    if centers.shape[0] != memberships.shape[0]:
        raise ValueError(f"centers has {centers.shape[0]} microstates, memberships {memberships.shape[0]}")
    if coarse_T is not None and coarse_T.shape != (memberships.shape[1],) * 2:
        raise ValueError(f"coarse_T is {coarse_T.shape}, expected {(memberships.shape[1],) * 2}")
    probe = np.asarray(featurizer.transform(["⊢ True"]))
    if probe.shape[1] != centers.shape[1]:
        raise ValueError(f"featurizer gives {probe.shape[1]} dims, centers have {centers.shape[1]}")
    return MSMArtifacts(featurizer, centers, memberships, coarse_T, report)


def build_area_prior(msm_dir: str | Path, lam: float = 0.3, use_coarse_T: bool = True,
                     **kwargs: Any) -> AreaPrior:
    """The ``prior(state_text, candidates)`` for ``markov_lean.search``, from a saved model.

    ``lam`` mixes the area scores into the LM's distribution (0 is LM-only).
    ``use_coarse_T`` steers toward the areas the model expects next; without it
    the target is flat and only the diversity and stuck-escape terms act.
    Other keyword arguments go to ``AreaPrior`` (``temperature``, ``k``,
    ``diversity_alpha``, ``window``, ``threshold``, ``escape_strength``).
    """
    art = load_msm_artifacts(msm_dir)
    coarse_T = art.coarse_T if use_coarse_T else None
    return make_prior(art.featurizer, art.centers, art.memberships, lam=lam, coarse_T=coarse_T, **kwargs)
