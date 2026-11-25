"""Full-scale analysis: fit the Markov state model on all of LeanDojo train, then project other datasets.

``markov-lean full --out results/full`` runs, in order:

1. **Keys.** Read only ``proof_id`` and ``step`` of the training file, keep
   proofs with at least ``data.min_steps`` steps (8, as in the pilot), and
   sort them.  Goal-state text is never held for the whole set at once.
2. **Featurizer.** Fit TF-IDF + SVD on a seeded sample of
   ``features.fit_proofs`` proofs (the pilot's scale), which is all the
   vocabulary and SVD need.
3. **Features.** Stream the training file in ``features.batch_rows`` batches,
   transform them in ``features.n_jobs`` worker processes, and write the rows
   into a float32 memmap on disk (``<out>/cache/features.npy``).  A rerun with
   the same data and settings reuses it.
4. **Model.** ``msm.run`` on the memmapped features with the ``msm`` settings
   (10 macrostates and lag 1, as the pilot), saved to ``<out>/msm`` in the
   layout ``prior_loader.build_area_prior`` reads.
5. **Projection.** Each dataset under ``project.datasets`` is placed on the
   fitted model and measured (``markov_lean.project``): per-dataset transition
   matrices, area-switch rates per proof, dwell times, occupancy, and
   bootstrap intervals over whole proofs, plus differences from a reference
   dataset.  Missing files are skipped with a note.
6. **Figures and notes.** PDFs under ``<out>/figures`` and ``<out>/RUN.md``
   with the headline table; ``full_report.json`` records counts, timings and
   peak memory.

``markov-lean project`` reruns step 5 alone against a saved model, for
example once the proof search has written proofs for miniF2F or PutnamBench.

No step makes an API call or needs a network connection.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from . import project as pj
from .featurize import make_featurizer

ROOT = Path(__file__).resolve().parents[2]
KEY = ["proof_id", "step"]


def _path(p: str | Path | None, base: Path = ROOT) -> Path | None:
    if p is None:
        return None
    p = Path(p).expanduser()
    return p if p.is_absolute() else base / p


def _cli_path(p: str | Path) -> Path:
    """A path given on the command line: relative to where the command runs, as ``--out`` is.

    Falls back to the repo root (how config paths resolve) when nothing is there, so
    running from the repo root behaves as before.
    """
    q = Path(p).expanduser()
    if q.is_absolute():
        return q
    here = Path.cwd() / q
    return here if here.exists() or not (ROOT / q).exists() else ROOT / q


def load_config(path: str | Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def peak_memory_gb() -> float:
    """Peak resident memory of this process so far (ru_maxrss is bytes on macOS, KiB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9


# --------------------------------------------------------------------------- reading the training file


def iter_batches(path: Path, columns: list[str], batch_rows: int) -> Iterator[pd.DataFrame]:
    """Row batches of a parquet file, reading only ``columns``."""
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    for b in f.iter_batches(batch_size=batch_rows, columns=columns):
        yield b.to_pandas()


def select_keys(keys: pd.DataFrame, min_steps: int = 1, max_steps: int | None = None,
                max_proofs: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Sorted unique (proof_id, step) rows of the proofs to keep.

    Proof length counts unique steps.  ``max_proofs`` draws proofs the way
    ``pilot.sample_proofs`` does (sorted ids, ``random.Random(seed).sample``),
    so the same seed picks the same proofs.
    """
    k = keys[KEY].copy()
    k["proof_id"] = k["proof_id"].astype(str)
    k["step"] = k["step"].astype("int64")
    k = k.drop_duplicates(KEY)
    sizes = k.groupby("proof_id").size()
    ok = sizes >= min_steps
    if max_steps is not None:
        ok &= sizes <= max_steps
    ids = sorted(sizes[ok].index)
    if max_proofs is not None and len(ids) > max_proofs:
        ids = sorted(random.Random(seed).sample(ids, int(max_proofs)))
    k = k[k["proof_id"].isin(set(ids))]
    return k.sort_values(KEY, kind="stable").reset_index(drop=True)


def read_keys(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=KEY)


def collect_texts(path: Path, keys: pd.DataFrame, batch_rows: int) -> list[str]:
    """``state_before`` for exactly the rows in ``keys``, in the order of ``keys``."""
    index = pd.MultiIndex.from_frame(keys[KEY])
    texts: list[str | None] = [None] * len(keys)
    for b in iter_batches(path, KEY + ["state_before"], batch_rows):
        b["proof_id"] = b["proof_id"].astype(str)
        pos = index.get_indexer(pd.MultiIndex.from_frame(b[KEY].astype({"step": "int64"})))
        for p, t in zip(pos, b["state_before"].tolist()):
            if p >= 0 and texts[p] is None:
                texts[p] = t if isinstance(t, str) else ""
    missing = sum(t is None for t in texts)
    if missing:
        raise ValueError(f"{missing} selected rows were not found in {path}")
    return texts  # type: ignore[return-value]


# --------------------------------------------------------------------------- parallel transform

_WORKER: Any = None


def _init_worker(blob: bytes) -> None:
    global _WORKER
    _WORKER = pickle.loads(blob)  # noqa: S301 - the featurizer this process just pickled


def _transform_in_worker(texts: list[str]) -> np.ndarray:
    return np.asarray(_WORKER.transform(texts), dtype=np.float32)


class ParallelTransform:
    """``featurizer.transform`` over chunks of texts in ``n_jobs`` processes, results in order.

    With ``n_jobs <= 1`` everything runs in this process.  At most
    ``2 * n_jobs`` chunks are in flight, so memory stays bounded however long
    the input stream is.  Use it as a context manager.
    """

    def __init__(self, featurizer: Any, n_jobs: int = 1, chunk_rows: int = 5000):
        self.featurizer = featurizer
        self.n_jobs = max(1, int(n_jobs))
        self.chunk_rows = max(1, int(chunk_rows))
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> "ParallelTransform":
        if self.n_jobs > 1:
            self._pool = ProcessPoolExecutor(self.n_jobs, initializer=_init_worker,
                                             initargs=(pickle.dumps(self.featurizer),))
        return self

    def __exit__(self, *exc) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def stream(self, items: Iterable[tuple[Any, list[str]]]) -> Iterator[tuple[Any, np.ndarray]]:
        """For each ``(tag, texts)`` yield ``(tag, features)`` in input order."""
        if self._pool is None:
            for tag, texts in items:
                yield tag, np.asarray(self.featurizer.transform(texts), dtype=np.float32)
            return
        pending: list[tuple[Any, Any]] = []
        for tag, texts in items:
            pending.append((tag, self._pool.submit(_transform_in_worker, texts)))
            while len(pending) >= 2 * self.n_jobs:
                t, fut = pending.pop(0)
                yield t, fut.result()
        for t, fut in pending:
            yield t, fut.result()

    def __call__(self, texts: list[str]) -> np.ndarray:
        chunks = [(i, texts[i:i + self.chunk_rows]) for i in range(0, len(texts), self.chunk_rows)]
        parts = [x for _, x in self.stream(chunks)]
        if not parts:
            return np.asarray(self.featurizer.transform([]), dtype=np.float32)
        return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------------- features for the training set


def fit_featurizer(path: Path, keys: pd.DataFrame, msm_section: dict, fit_proofs: int | None, seed: int,
                   batch_rows: int) -> tuple[Any, dict]:
    """Fit the configured featurizer on a seeded sample of whole proofs from ``keys``."""
    ids = sorted(keys["proof_id"].unique())
    if fit_proofs is not None and len(ids) > fit_proofs:
        ids = sorted(random.Random(seed + 7).sample(ids, int(fit_proofs)))
    sub = keys[keys["proof_id"].isin(set(ids))].reset_index(drop=True)
    texts = collect_texts(path, sub, batch_rows)
    name = msm_section.get("featurizer", "tfidf_svd")
    kw = dict(msm_section.get("featurizer_kwargs") or {})
    if name == "tfidf_svd":
        kw.setdefault("seed", seed)
    feat = make_featurizer(name, **kw)
    feat.fit_transform(texts)
    return feat, {"fit_proofs": len(ids), "fit_steps": len(sub)}


def featurize_stream(path: Path, keys: pd.DataFrame, featurizer: Any, out_path: Path, batch_rows: int,
                     n_jobs: int, chunk_rows: int = 5000, say: Callable = print) -> np.ndarray:
    """Write features for every row of ``keys`` (in ``keys`` order) into a float32 ``.npy`` memmap."""
    dim = int(np.asarray(featurizer.transform(["⊢ True"])).shape[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(len(keys), dim))
    index = pd.MultiIndex.from_frame(keys[KEY])
    claimed = np.zeros(len(keys), bool)
    done = 0
    t0 = time.perf_counter()

    def chunks():
        for b in iter_batches(path, KEY + ["state_before"], batch_rows):
            b["proof_id"] = b["proof_id"].astype(str)
            pos = index.get_indexer(pd.MultiIndex.from_frame(b[KEY].astype({"step": "int64"})))
            take = np.flatnonzero(pos >= 0)
            if not len(take):
                continue
            p = pos[take]
            _, first = np.unique(p, return_index=True)   # duplicate keys: first row wins
            take, p = take[first], p[first]
            fresh = ~claimed[p]
            take, p = take[fresh], p[fresh]
            claimed[p] = True
            texts = b["state_before"].fillna("").astype(str).to_numpy()[take].tolist()
            for i in range(0, len(p), chunk_rows):
                yield p[i:i + chunk_rows], texts[i:i + chunk_rows]

    with ParallelTransform(featurizer, n_jobs, chunk_rows) as pt:
        for p, feats in pt.stream(chunks()):
            X[p] = feats
            done += len(p)
            if done and (done // chunk_rows) % 20 == 0:
                rate = done / max(time.perf_counter() - t0, 1e-9)
                say(f"  features: {done:,}/{len(keys):,} rows ({rate:,.0f} rows/s)")
    X.flush()
    if not claimed.all():
        raise ValueError(f"{int((~claimed).sum())} selected rows were not found in {path}")
    del X
    return np.load(out_path, mmap_mode="r")


def _fingerprint(path: Path, keys: pd.DataFrame, cfg: dict) -> str:
    st = path.stat()
    h = hashlib.sha256()
    h.update(json.dumps({"path": str(path.resolve()), "size": st.st_size, "mtime": int(st.st_mtime),
                         "n_rows": len(keys), "n_proofs": int(keys["proof_id"].nunique()), "cfg": cfg},
                        sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def training_features(path: Path, keys: pd.DataFrame, config: dict, cache_dir: Path, n_jobs: int,
                      refit: bool = False, say: Callable = print) -> tuple[np.ndarray, Any, dict]:
    """Features and featurizer for the training rows, reusing ``cache_dir`` when it matches."""
    fcfg = config.get("features") or {}
    msm_section = config.get("msm") or {}
    seed = int((config.get("data") or {}).get("seed", 0))
    batch_rows = int(fcfg.get("batch_rows", 50000))
    chunk_rows = int(fcfg.get("chunk_rows", 5000))
    fit_proofs = fcfg.get("fit_proofs")
    key_cfg = {"featurizer": msm_section.get("featurizer", "tfidf_svd"),
               "featurizer_kwargs": msm_section.get("featurizer_kwargs"), "fit_proofs": fit_proofs, "seed": seed}
    fp = _fingerprint(path, keys, key_cfg)
    meta_p, feat_p, x_p = cache_dir / "meta.json", cache_dir / "featurizer.pkl", cache_dir / "features.npy"
    if not refit and meta_p.exists() and feat_p.exists() and x_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("fingerprint") == fp and meta.get("complete"):
            say(f"features: reusing {x_p} ({meta['n_rows']:,} rows)")
            with open(feat_p, "rb") as f:
                featurizer = pickle.load(f)  # noqa: S301 - written by this function
            return np.load(x_p, mmap_mode="r"), featurizer, {**meta, "reused": True}
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    featurizer, fit_info = fit_featurizer(path, keys, msm_section, fit_proofs, seed, batch_rows)
    t1 = time.perf_counter()
    say(f"featurizer: fit on {fit_info['fit_proofs']:,} proofs / {fit_info['fit_steps']:,} steps "
        f"in {t1 - t0:.0f}s")
    with open(feat_p, "wb") as f:
        pickle.dump(featurizer, f)
    meta = {"fingerprint": fp, "n_rows": len(keys), **fit_info, "complete": False}
    meta_p.write_text(json.dumps(meta, indent=2))
    X = featurize_stream(path, keys, featurizer, x_p, batch_rows, n_jobs, chunk_rows, say)
    t2 = time.perf_counter()
    meta.update({"complete": True, "dim": int(X.shape[1]), "seconds_fit": round(t1 - t0, 1),
                 "seconds_transform": round(t2 - t1, 1), "n_jobs": n_jobs})
    meta_p.write_text(json.dumps(meta, indent=2))
    say(f"features: {len(keys):,} rows x {X.shape[1]} dims in {t2 - t1:.0f}s with {n_jobs} processes")
    return X, featurizer, {**meta, "reused": False}


# --------------------------------------------------------------------------- projection


def model_from_result(res) -> pj.FittedModel:
    return pj.FittedModel(res.featurizer, res.centers, res.micro_to_macro, res.memberships, res.coarse_T,
                          res.lag, int(res.report["n_macrostates"]), res.report)


def _glob_steps(pattern: str) -> pd.DataFrame | None:
    """Concatenate every steps parquet matching ``pattern`` (small files, such as a search run's).

    The same theorem appears once per condition and seed, so each ``proof_id``
    is prefixed with its file's folder relative to the pattern's fixed part.
    """
    import glob as _g

    base = _path(pattern.split("*", 1)[0].rsplit("/", 1)[0] or ".")
    files = sorted(Path(f) for f in _g.glob(str(_path(pattern)), recursive=True))
    if not files:
        return None
    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=KEY + ["state_before"])
        tag = str(f.parent.relative_to(base)) if f.parent.is_relative_to(base) else str(f.parent)
        parts.append(df.assign(proof_id=tag + "/" + df["proof_id"].astype(str)))
    return pd.concat(parts, ignore_index=True)


def load_dataset(spec: dict) -> pd.DataFrame | None:
    """A dataset's steps (only the columns projection needs), filtered by its length limits.

    ``path`` may hold a ``*`` pattern, which reads every matching file (see ``_glob_steps``).
    """
    if "*" in str(spec["path"]):
        df = _glob_steps(str(spec["path"]))
        if df is None:
            return None
        df["step"] = df["step"].astype("int64")
        df = df.drop_duplicates(KEY)
        keep = select_keys(df, int(spec.get("min_steps", 1)), spec.get("max_steps"), spec.get("max_proofs"),
                           int(spec.get("seed", 0)))
        return keep.merge(df, on=KEY, how="left")
    path = _path(spec["path"])
    if path is None or not path.exists():
        return None
    keep = select_keys(read_keys(path), int(spec.get("min_steps", 1)), spec.get("max_steps"),
                       spec.get("max_proofs"), int(spec.get("seed", 0)))
    # texts only for the rows kept, read in batches, so a large file never sits in memory whole
    return keep.assign(state_before=collect_texts(path, keep, int(spec.get("batch_rows", 50000))))


def run_projection(datasets: list[dict], model: pj.FittedModel, out_dir: Path, pcfg: dict, n_jobs: int,
                   in_sample: tuple[str, pd.DataFrame] | None = None, say: Callable = print
                   ) -> tuple[list[pj.DatasetResult], list[dict]]:
    """Place and measure every dataset; returns (results, notes on skipped datasets)."""
    lag = int(pcfg.get("lag") or model.lag)
    n_boot = int(pcfg.get("bootstrap", 1000))
    seed = int(pcfg.get("seed", 0))
    batch_rows = int(pcfg.get("batch_rows", 50000))
    results: list[pj.DatasetResult] = []
    skipped: list[dict] = []
    if in_sample is not None:
        name, placed = in_sample
        results.append(pj.analyze(name, placed, model.n_macrostates, lag, n_boot, seed))
        say(f"project: {name} (in-sample) {results[-1].tables.n_proofs:,} proofs")
    with ParallelTransform(model.featurizer, n_jobs, int(pcfg.get("chunk_rows", 5000))) as pt:
        for spec in datasets:
            name = spec["name"]
            t0 = time.perf_counter()
            df = load_dataset(spec)
            if df is None:
                skipped.append({"dataset": name, "reason": f"{spec['path']} not found"})
                say(f"project: skip {name}: {spec['path']} not found")
                continue
            if df.empty:
                skipped.append({"dataset": name, "reason": "no proofs left after the length filter"})
                say(f"project: skip {name}: no proofs left after the length filter")
                continue
            placed = pj.place_states(df, model, transform=pt, batch_rows=batch_rows)
            r = pj.analyze(name, placed, model.n_macrostates, lag, n_boot, seed)
            results.append(r)
            rate = r.point["pooled_switch_rate"]
            say(f"project: {name} {r.tables.n_proofs:,} proofs, {len(placed):,} states"
                + (f", switch rate {float(rate):.3f}" if r.has_transitions else ", statements only (no moves)")
                + f" ({time.perf_counter() - t0:.0f}s)")
    return results, skipped


# --------------------------------------------------------------------------- figures and notes


def make_figures(out: Path, res_report: dict | None, msm_dir: Path, results: list[pj.DatasetResult],
                 say: Callable = print) -> list[str]:
    """Draw every figure that applies; a missing plotting library only skips figures."""
    try:
        from . import figures as fg
        from . import full_figures as ff
    except ImportError as exc:  # pragma: no cover - figures extra not installed
        say(f"figures: skipped ({exc})")
        return []
    figs: list[tuple[str, Any]] = []
    try:
        its = pd.read_csv(msm_dir / "implied_timescales.csv")
        if len(its):
            piv = its.pivot(index="lag", columns="index", values="timescale").sort_index()
            figs.append(("implied_timescales", fg.implied_timescales(piv.index.to_numpy(), piv.to_numpy())))
        ck = pd.read_csv(msm_dir / "ck_test.csv")
        if len(ck):
            lags = sorted(ck["lag"].unique())
            m = int(ck["from_set"].max()) + 1
            pred = np.zeros((len(lags), m, m))
            est = np.zeros_like(pred)
            for li, L in enumerate(lags):
                sub = ck[ck["lag"] == L]
                i, j = sub["from_set"].to_numpy(int), sub["to_set"].to_numpy(int)
                pred[li, i, j] = sub["predicted"].to_numpy()
                est[li, i, j] = sub["estimated"].to_numpy()
            figs.append(("chapman_kolmogorov", fg.chapman_kolmogorov(lags, pred, est)))
        coarse = np.load(msm_dir / "coarse_T.npy")
        figs.append(("model_transitions", fg.transition_heatmap(coarse, check_stochastic=False)))
        for r in results:
            if r.has_transitions:
                T = np.nan_to_num(r.point["transition_matrix"])
                figs.append((f"transitions_{r.name}", fg.transition_heatmap(T, check_stochastic=False)))
        if results:
            figs.append(("switch_rates", ff.switch_rates(results)))
            figs.append(("occupancy", ff.occupancy(results)))
            if any(r.has_transitions for r in results):
                figs.append(("dwell_times", ff.dwell_times(results)))
                figs.append(("switches_per_proof", ff.switches_by_length(results)))
        paths = fg.save_all(figs, out / "figures")
    except ImportError as exc:  # pragma: no cover - figures extra not installed
        say(f"figures: skipped ({exc})")
        return []
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except ImportError:  # pragma: no cover
            pass
    return [str(p) for p in paths]


def _fmt(x, lo, hi, digits=3) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    if lo is None or hi is None:
        return f"{x:.{digits}f}"
    return f"{x:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def write_run_notes(out: Path, report: dict, summary: dict) -> Path:
    """``RUN.md``: what was fit and the headline numbers, for a quick look after a run."""
    lines = ["# Full-scale run", ""]
    m = report.get("msm") or {}
    if m:
        lines += [f"Model: {report['data']['n_proofs']:,} training proofs, {report['data']['n_steps']:,} steps; "
                  f"{m.get('n_microstates')} microstates, lag {m.get('lag')}, {m.get('n_macrostates')} macrostates. "
                  f"Checks failed: {', '.join(m.get('checks_failed') or []) or 'none'}.", ""]
    lines += ["Switch rate is the share of a proof's moves (step t to t + lag) that change macrostate. "
              "Brackets are 95% bootstrap intervals over whole proofs.", "",
              "| dataset | proofs | steps | pooled switch rate | mean per-proof switch rate | mean dwell (steps) |",
              "|---|---:|---:|---|---|---|"]
    for name, d in (summary.get("datasets") or {}).items():
        lines.append(f"| {name} | {d['n_proofs']:,} | {d['n_steps']:,} | "
                     f"{_fmt(d['pooled_switch_rate'], d['pooled_switch_rate_lo'], d['pooled_switch_rate_hi'])} | "
                     f"{_fmt(d['mean_switch_rate'], d['mean_switch_rate_lo'], d['mean_switch_rate_hi'])} | "
                     f"{_fmt(d['mean_dwell'], d['mean_dwell_lo'], d['mean_dwell_hi'], 2)} |")
    skipped = report.get("skipped") or []
    if skipped:
        lines += ["", "Skipped: " + "; ".join(f"{s['dataset']} ({s['reason']})" for s in skipped) + "."]
    lines += ["", "Statement-only datasets have one goal state per theorem, so they show occupancy only. "
              "See results/README.md for every file."]
    p = out / "RUN.md"
    p.write_text("\n".join(lines) + "\n")
    return p


# --------------------------------------------------------------------------- entry points


def train_path(data: dict) -> Path:
    processed = _path(data.get("processed_dir", "data/processed"))
    return processed / f"{data.get('train', 'leandojo_train')}.parquet"


def run(config: dict, out_dir: str | Path, *, n_jobs: int | None = None, refit_features: bool = False,
        skip_projection: bool = False, verbose: bool = True) -> int:
    """Fit on the full training set, project the datasets, write results.  Returns 0 on success."""
    from .msm import run as run_msm
    from .pilot import msm_config

    say = print if verbose else (lambda *a, **k: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    data = config.get("data") or {}
    fcfg = config.get("features") or {}
    pcfg = config.get("project") or {}
    n_jobs = int(n_jobs if n_jobs is not None else fcfg.get("n_jobs", max(1, (os.cpu_count() or 2) // 2)))
    seed = int(data.get("seed", 0))

    path = train_path(data)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run ./scripts/fetch_data.sh first (it writes data/processed/).")
    keys = select_keys(read_keys(path), int(data.get("min_steps", 8)), data.get("max_steps"),
                       data.get("max_proofs"), seed)
    if keys.empty:
        raise ValueError("no proofs left after filtering; lower data.min_steps")
    lens = keys.groupby("proof_id").size()
    say(f"full: {len(lens):,} proofs, {len(keys):,} steps, median length {lens.median():g} "
        f"(min_steps {data.get('min_steps', 8)})")

    X, featurizer, feat_info = training_features(path, keys, config, out / "cache", n_jobs, refit_features, say)
    t_feat = time.perf_counter()
    res = run_msm(keys.assign(state_before=""), msm_config(config.get("msm")), features=X, featurizer=featurizer)
    res.save(out / "msm")
    t_msm = time.perf_counter()
    rep = res.report
    failed = [k for k, c in rep["checks"].items() if not c["passed"]]
    say(f"msm: {rep['n_microstates']} microstates, lag {rep['lag']}, {rep['n_macrostates']} macrostates; "
        f"checks failed: {failed or 'none'} ({t_msm - t_feat:.0f}s)")
    del X

    report: dict[str, Any] = {
        "data": {"train": str(path), "min_steps": int(data.get("min_steps", 8)), "n_proofs": int(len(lens)),
                 "n_steps": int(len(keys)), "median_proof_length": float(lens.median())},
        "features": {k: v for k, v in feat_info.items() if k != "fingerprint"},
        "msm": {"n_microstates": rep["n_microstates"], "lag": rep["lag"], "n_macrostates": rep["n_macrostates"],
                "all_checks_passed": rep["all_checks_passed"], "checks_failed": failed,
                "timescales_steps": rep["timescales_steps"][:rep["n_macrostates"]],
                "dir": str(out / "msm")},
        "n_jobs": n_jobs,
    }
    summary: dict = {}
    results: list[pj.DatasetResult] = []
    if not skip_projection:
        model = model_from_result(res)
        in_sample = (pcfg.get("train_name", "leandojo_train"), res.assignments)
        results, skipped = run_projection(list(pcfg.get("datasets") or []), model, out / "project", pcfg, n_jobs,
                                          in_sample, say)
        summary = pj.write_results(results, out / "project", pcfg.get("reference"),
                                   float(pcfg.get("level", 0.95)), model)
        report["skipped"] = skipped
        report["projected"] = [r.name for r in results]
    t_proj = time.perf_counter()
    report["figures"] = make_figures(out, rep, out / "msm", results, say)
    report["seconds"] = {"features": round(t_feat - t0, 1), "msm": round(t_msm - t_feat, 1),
                         "projection": round(t_proj - t_msm, 1), "total": round(time.perf_counter() - t0, 1)}
    report["peak_memory_gb"] = round(peak_memory_gb(), 2)
    (out / "full_report.json").write_text(json.dumps(report, indent=2, default=str))
    if summary:
        write_run_notes(out, report, summary)
    say(f"full: done in {report['seconds']['total']:.0f}s, peak memory {report['peak_memory_gb']} GB. "
        f"See {out / 'RUN.md'}")
    return 0


def run_project_only(config: dict, msm_dir: str | Path, out_dir: str | Path, *, extra: list[str] | None = None,
                     n_jobs: int | None = None, verbose: bool = True) -> int:
    """Project datasets onto a saved model (no refit).  ``extra`` adds ``name=path`` datasets."""
    say = print if verbose else (lambda *a, **k: None)
    pcfg = dict(config.get("project") or {})
    datasets = list(pcfg.get("datasets") or [])
    for item in extra or []:
        name, _, p = item.partition("=")
        if not p:
            raise ValueError(f"--dataset takes name=path, got {item!r}")
        datasets.append({"name": name, "path": str(_cli_path(p)), "min_steps": 1})
    msm_dir = _cli_path(msm_dir)
    model = pj.load_model(msm_dir)
    in_sample = None
    if (msm_dir / "assignments.parquet").exists() and pcfg.get("include_train", True):
        in_sample = (pcfg.get("train_name", "leandojo_train"), pd.read_parquet(msm_dir / "assignments.parquet"))
    n_jobs = int(n_jobs if n_jobs is not None else (config.get("features") or {}).get("n_jobs", 1))
    out = Path(out_dir)
    results, skipped = run_projection(datasets, model, out, pcfg, n_jobs, in_sample, say)
    summary = pj.write_results(results, out, pcfg.get("reference"), float(pcfg.get("level", 0.95)), model)
    figs = make_figures(out, None, msm_dir, results, say)
    (out / "project_report.json").write_text(json.dumps(
        {"msm_dir": str(msm_dir), "projected": [r.name for r in results], "skipped": skipped, "figures": figs,
         "peak_memory_gb": round(peak_memory_gb(), 2)}, indent=2))
    write_run_notes(out, {"skipped": skipped}, summary)
    return 0
