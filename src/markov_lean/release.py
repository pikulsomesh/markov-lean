"""Release tooling: export the labelled dataset and check a release folder.

``markov-lean export-dataset --run results/pilot3 --out release/`` reads a finished
pilot run (two passes, judge labels in place) and writes a self-contained folder:

    release/
      README.md          dataset card (Hugging Face YAML header + prose)
      SCHEMA.md          every column of every table, generated from ``TABLES``
      run_summary.json   model and agreement numbers from the run, no paths
      MANIFEST.json      files, sizes, row counts, columns and SHA-256
      SHA256SUMS         ``sha256sum -c`` / ``shasum -a 256 -c`` format
      data/steps.parquet   one row per goal state, with its Markov-model area
      data/labels.parquet  one row per judge label (and consensus), steps and whole proofs
      data/proofs.parquet  one row per proof

``markov-lean check-release release/`` checks that such a folder is complete and
consistent (files, checksums, row counts, columns, codes, keys, the card's YAML)
and that nothing in it looks like an API key or a local path.  It exits 1 on any
problem and prints one line per problem.

Nothing here uploads anything or calls an API.  Uploading to Hugging Face or
Zenodo is done by hand from the checked folder.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import msc

RELEASE_FORMAT = 1
DATA_DIR = "data"
KEY = ["proof_id", "step"]
PROOF_LEVEL_STEP = -1
CONSENSUS = "consensus"

# --------------------------------------------------------------------------- schema

# table -> list of (column, dtype, nullable, description).  The export writes
# exactly these columns in this order, SCHEMA.md is generated from it, and the
# check compares every parquet file with it.
TABLES: dict[str, list[tuple[str, str, bool, str]]] = {
    "steps": [
        ("proof_id", "string", False, "stable id `<source>:<full theorem name>`"),
        ("step", "int64", False, "0-based position of the tactic in the proof (LeanDojo's traced order)"),
        ("theorem", "string", False, "full Lean name of the theorem"),
        ("file_path", "string", False, "Mathlib file the theorem is in, relative to the Mathlib root; empty if unknown"),
        ("split", "string", False, "split in the source benchmark (`train`, `val`, `test`)"),
        ("tactic", "string", True, "tactic text as written; null in a release made with `--no-text`"),
        ("state_before", "string", True, "pretty-printed goal state before the tactic; null with `--no-text`"),
        ("state_after", "string", True, "goal state after the tactic (`no goals` when closed); null with `--no-text`"),
        ("microstate", "int64", True, "k-means microstate of `state_before`; null if the run has no assignment"),
        ("macrostate", "int64", True, "PCCA+ macrostate (the model's area); -1 outside the connected set"),
        ("membership", "float64", True, "PCCA+ membership of the assigned macrostate, 0 to 1"),
    ],
    "labels": [
        ("proof_id", "string", False, "as in `steps`"),
        ("step", "int64", False, "step the label is for; -1 for a whole-proof label"),
        ("judge", "string", False, "model id of the judge, or `consensus` (both judges gave the same code)"),
        ("msc", "string", False, "two-digit MSC 2020 top-level code, e.g. `26`"),
        ("confidence", "float64", True, "0 to 1 as the judge reported it; the mean of both for `consensus`"),
    ],
    "proofs": [
        ("proof_id", "string", False, "as in `steps`"),
        ("theorem", "string", False, "full Lean name"),
        ("file_path", "string", False, "as in `steps`"),
        ("mathlib_folder", "string", True, "top-level Mathlib folder from `file_path` (the baseline partition)"),
        ("split", "string", False, "as in `steps`"),
        ("n_steps", "int64", False, "tactic steps in the proof"),
        ("n_switches", "int64", True, "changes of macrostate between consecutive steps; null without assignments"),
        ("proof_msc", "string", True, "consensus whole-proof MSC code; null when the judges disagree or one is missing"),
        ("n_step_labels", "int64", False, "steps with a label from at least one judge"),
        ("n_consensus_steps", "int64", False, "steps where both judges gave the same code"),
    ],
}

FILES_REQUIRED = ["README.md", "SCHEMA.md", "run_summary.json", "MANIFEST.json", "SHA256SUMS"] + [
    f"{DATA_DIR}/{t}.parquet" for t in TABLES]

CARD_KEYS_REQUIRED = ["pretty_name", "license", "language", "task_categories", "size_categories", "configs"]

# --------------------------------------------------------------------------- leak patterns

SECRET_PATTERNS = {
    "Anthropic API key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    "AWS access key id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    "key assignment": re.compile(r"(?i)\b(?:ANTHROPIC_API_KEY|AWS_SECRET_ACCESS_KEY|HF_TOKEN|VLLM_API_KEY)\s*[=:]\s*\S{8,}"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
PATH_PATTERNS = {
    "home directory path": re.compile(r"(?<![\w.~])/(?:Users|home|root)/[^\s/\"'`]+"),
    "temp or mount path": re.compile(r"(?<![\w.])/(?:private/var|var/folders|tmp|mnt|Volumes|opt/homebrew)/[^\s\"'`]+"),
    "Windows user path": re.compile(r"\b[A-Za-z]:\\\\?Users\\\\?[^\s\\\\]+"),
    "home shorthand path": re.compile(r"(?<![\w])~/[\w.\-]+"),
}


def find_leaks(text: str, *, paths: bool = True) -> list[str]:
    """Kinds of secret (and, with ``paths``, local path) that occur in ``text``."""
    found = [name for name, rx in SECRET_PATTERNS.items() if rx.search(text)]
    if paths:
        found += [name for name, rx in PATH_PATTERNS.items() if rx.search(text)]
    return found


def frame_leaks(df: pd.DataFrame, *, paths: bool = True) -> list[str]:
    """``column: kind`` for every string column of ``df`` holding a secret or a local path."""
    out = []
    patterns = dict(SECRET_PATTERNS)
    if paths:
        patterns.update(PATH_PATTERNS)
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_string_dtype(s) or s.dtype == object):
            continue
        s = s.dropna().astype(str)
        if s.empty:
            continue
        for name, rx in patterns.items():
            hits = s.str.contains(rx, regex=True)
            if hits.any():
                out.append(f"{col}: {name} in {int(hits.sum())} rows (first at row {int(hits.idxmax())})")
    return out


def looks_like_path(value: str) -> bool:
    """An absolute or home-relative path, which a release must not carry."""
    v = value.strip()
    return bool(v.startswith(("/", "~/", "\\\\")) or re.match(r"^[A-Za-z]:[\\/]", v) or find_leaks(v))


def scrub(obj: Any) -> Any:
    """Drop every dict entry or list item whose string value looks like a path or a secret."""
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()
                if not (isinstance(v, str) and looks_like_path(v)) and k not in ("commands",)}
    if isinstance(obj, list):
        return [scrub(v) for v in obj if not (isinstance(v, str) and looks_like_path(v))]
    return obj


# --------------------------------------------------------------------------- helpers


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def mathlib_folder(file_path: Any) -> str | None:
    """``Mathlib/NumberTheory/Padics/X.lean`` -> ``NumberTheory`` (same rule as ``evaluate.folder_label``)."""
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    parts = [p for p in file_path.replace("\\", "/").split("/") if p and p != "."]
    if "Mathlib" not in parts:
        return None
    parts = parts[parts.index("Mathlib") + 1:]
    if not parts:
        return None
    head = parts[0]
    return head[:-5] if head.endswith(".lean") else head


def relative_file_path(file_path: Any) -> str:
    """Strip anything before the Mathlib root, so no local checkout path survives."""
    if not isinstance(file_path, str) or not file_path:
        return ""
    p = file_path.replace("\\", "/")
    if not (p.startswith("/") or p.startswith("~") or re.match(r"^[A-Za-z]:/", p)):
        return p
    parts = p.split("/")
    if "Mathlib" in parts:
        return "/".join(parts[parts.index("Mathlib"):])
    return ""  # an absolute path outside Mathlib says nothing useful and leaks the machine


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else None
    except (OSError, subprocess.SubprocessError):
        return None


def _typed(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Columns of ``table`` in order with their release dtypes (pandas nullable types)."""
    cols = TABLES[table]
    out = pd.DataFrame(index=df.index)
    for name, dtype, _, _ in cols:
        s = df[name] if name in df.columns else pd.Series(pd.NA, index=df.index)
        if dtype == "string":
            out[name] = s.astype("string")
        elif dtype == "int64":
            out[name] = pd.to_numeric(s, errors="coerce").astype("Int64")
        elif dtype == "float64":
            out[name] = pd.to_numeric(s, errors="coerce").astype("Float64")
        else:  # pragma: no cover - TABLES only uses the three above
            raise ValueError(dtype)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- reading a run


def find_run_files(run: Path, labels: Path | None = None) -> dict[str, Path | None]:
    """Where a pilot run keeps the inputs the export needs."""
    run = Path(run)

    def first(*cands: Path) -> Path | None:
        return next((c for c in cands if c.exists()), None)

    return {
        "steps": first(run / "judge" / "sample_steps.parquet", run / "steps.parquet"),
        "labels": labels if labels is not None else first(run / "judge" / "labels.parquet",
                                                           run / "judge" / "run" / "labels.parquet"),
        "assignments": first(run / "msm" / "assignments.parquet"),
        "pilot_report": first(run / "pilot_report.json"),
        "msm_report": first(run / "msm" / "msm_report.json"),
        "issues": first(run / "judge" / "run" / "issues.json", run / "judge" / "issues.json"),
    }


def _consensus(labels: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add consensus rows when exactly two judges are present (as ``judge.consensus`` defines it)."""
    judges = [j for j in dict.fromkeys(labels["judge"].astype(str)) if j != CONSENSUS]
    if CONSENSUS in set(labels["judge"].astype(str)) or len(judges) != 2:
        return labels, judges
    a = labels[labels.judge == judges[0]].drop_duplicates(KEY)
    b = labels[labels.judge == judges[1]].drop_duplicates(KEY)
    p = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    p = p[p.msc_a == p.msc_b]
    cons = pd.DataFrame({
        "proof_id": p.proof_id, "step": p.step, "judge": CONSENSUS, "msc": p.msc_a,
        "confidence": (p.confidence_a.astype(float) + p.confidence_b.astype(float)) / 2,
    })
    return pd.concat([labels, cons], ignore_index=True), judges


def build_tables(steps: pd.DataFrame, labels: pd.DataFrame, assignments: pd.DataFrame | None = None,
                 *, include_text: bool = True) -> tuple[dict[str, pd.DataFrame], dict]:
    """The three release tables from a run's steps, judge labels and (optional) MSM assignments."""
    steps = steps.copy()
    steps["proof_id"] = steps["proof_id"].astype(str)
    steps["step"] = steps["step"].astype("int64")
    steps = steps[steps["step"] >= 0].drop_duplicates(KEY).sort_values(KEY).reset_index(drop=True)
    steps["file_path"] = steps["file_path"].map(relative_file_path) if "file_path" in steps else ""
    if not include_text:
        for c in ("tactic", "state_before", "state_after"):
            steps[c] = pd.NA
    if assignments is not None and len(assignments):
        a = assignments[KEY + ["microstate", "macrostate", "membership"]].copy()
        a["proof_id"] = a["proof_id"].astype(str)
        a["step"] = a["step"].astype("int64")
        steps = steps.merge(a.drop_duplicates(KEY), on=KEY, how="left")
    ids = set(steps["proof_id"])

    lab = labels.copy()
    lab["proof_id"] = lab["proof_id"].astype(str)
    lab["step"] = lab["step"].astype("int64")
    lab["msc"] = lab["msc"].map(msc.normalize)
    stats = {"labels_in": int(len(labels)), "labels_invalid_code": int(lab["msc"].isna().sum())}
    lab = lab[lab["msc"].notna()]
    outside = ~lab["proof_id"].isin(ids)
    stats["labels_outside_sample"] = int(outside.sum())
    lab = lab[~outside]
    step_keys = set(zip(steps["proof_id"], steps["step"]))
    unknown_step = (lab["step"] >= 0) & ~pd.Series([k in step_keys for k in zip(lab["proof_id"], lab["step"])],
                                                    index=lab.index, dtype=bool)
    stats["labels_unknown_step"] = int(unknown_step.sum())
    lab = lab[~unknown_step]
    lab, judges = _consensus(lab)
    lab = lab.drop_duplicates(KEY + ["judge"]).sort_values(["proof_id", "step", "judge"])

    per_step = lab[(lab.step >= 0)]
    judged = per_step[per_step.judge != CONSENSUS].groupby("proof_id")["step"].nunique()
    cons_steps = per_step[per_step.judge == CONSENSUS].groupby("proof_id")["step"].nunique()
    proof_msc = lab[(lab.step == PROOF_LEVEL_STEP) & (lab.judge == CONSENSUS)].set_index("proof_id")["msc"]
    g = steps.groupby("proof_id", sort=True)
    proofs = g.agg(theorem=("theorem", "first"), file_path=("file_path", "first"), split=("split", "first"),
                   n_steps=("step", "size")).reset_index()
    if "macrostate" in steps and steps["macrostate"].notna().any():
        def switches(m: pd.Series) -> int:
            v = m.dropna().to_numpy()
            return int((v[1:] != v[:-1]).sum()) if len(v) > 1 else 0
        proofs["n_switches"] = proofs["proof_id"].map(g["macrostate"].apply(switches))
    proofs["mathlib_folder"] = proofs["file_path"].map(mathlib_folder)
    proofs["proof_msc"] = proofs["proof_id"].map(proof_msc)
    proofs["n_step_labels"] = proofs["proof_id"].map(judged).fillna(0)
    proofs["n_consensus_steps"] = proofs["proof_id"].map(cons_steps).fillna(0)

    tables = {"steps": _typed(steps, "steps"), "labels": _typed(lab, "labels"), "proofs": _typed(proofs, "proofs")}
    stats.update({"judges": judges, "labels_out": int(len(tables["labels"]))})
    return tables, stats


def run_summary(files: dict[str, Path | None], tables: dict[str, pd.DataFrame], stats: dict,
                include_text: bool) -> dict:
    """Numbers worth shipping with the data; every path and command line is dropped."""
    rep = json.loads(files["pilot_report"].read_text()) if files.get("pilot_report") else {}
    msm_rep = json.loads(files["msm_report"].read_text()) if files.get("msm_report") else {}
    issues = json.loads(files["issues"].read_text()) if files.get("issues") else []
    issue_counts: dict[str, int] = {}
    for it in issues if isinstance(issues, list) else []:
        k = str(it.get("issue", "unknown")) if isinstance(it, dict) else "unknown"
        issue_counts[k] = issue_counts.get(k, 0) + 1
    lab = tables["labels"]
    summary = {
        "release_format": RELEASE_FORMAT,
        "created": _dt.date.today().isoformat(),
        "code_commit": _git_commit(),
        "includes_text": include_text,
        "counts": {
            "proofs": int(len(tables["proofs"])),
            "steps": int(len(tables["steps"])),
            "labels": int(len(lab)),
            "labels_by_judge": {str(k): int(v) for k, v in lab["judge"].value_counts().sort_index().items()},
            "proofs_with_consensus_area": int(tables["proofs"]["proof_msc"].notna().sum()),
            "min_proof_steps": int(tables["proofs"]["n_steps"].min()) if len(tables["proofs"]) else 0,
            "median_proof_steps": float(tables["proofs"]["n_steps"].median()) if len(tables["proofs"]) else 0.0,
        },
        "judges": stats.get("judges", []),
        "label_filtering": {k: v for k, v in stats.items() if k.startswith("labels_")},
        "judge_issues": issue_counts,
        "data": rep.get("data"),
        "msm": rep.get("msm") or {k: msm_rep.get(k) for k in ("n_microstates", "lag", "n_macrostates")
                                 if k in msm_rep} or None,
        "msm_checks_failed": rep.get("msm_checks_failed"),
        "judge_agreement": rep.get("judge_agreement"),
        "eval": rep.get("eval"),
    }
    return scrub({k: v for k, v in summary.items() if v is not None})


# --------------------------------------------------------------------------- writing


def schema_markdown() -> str:
    lines = ["# Schema", "",
             "Every table is a Parquet file under `data/`. Types are Arrow types; a nullable column may hold nulls.",
             "Join `steps`, `labels` and `proofs` on `proof_id` (and `step`).", ""]
    for table, cols in TABLES.items():
        lines += [f"## `{table}`", "", "| column | type | nullable | notes |", "|---|---|---|---|"]
        lines += [f"| `{c}` | {t} | {'yes' if n else 'no'} | {d} |" for c, t, n, d in cols]
        lines.append("")
    lines += ["## Keys", "",
              "- `steps`: (`proof_id`, `step`) is unique.",
              "- `labels`: (`proof_id`, `step`, `judge`) is unique; every `step >= 0` row matches a `steps` row,",
              "  and every `proof_id` is in `proofs`.",
              "- `proofs`: `proof_id` is unique and `n_steps` equals its row count in `steps`.", ""]
    return "\n".join(lines)


def write_manifest(out: Path, summary: dict) -> dict:
    """MANIFEST.json over every file but itself and SHA256SUMS, then SHA256SUMS over every file but itself."""
    entries = []
    for p in sorted(out.rglob("*")):
        rel = p.relative_to(out).as_posix()
        if not p.is_file() or rel in ("MANIFEST.json", "SHA256SUMS"):
            continue
        e: dict[str, Any] = {"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p)}
        if rel.startswith(f"{DATA_DIR}/") and rel.endswith(".parquet"):
            import pyarrow.parquet as pq

            meta = pq.read_metadata(p)
            e["rows"] = int(meta.num_rows)
            e["columns"] = list(meta.schema.names)
        entries.append(e)
    manifest = {"release_format": RELEASE_FORMAT, "created": summary.get("created"),
                "code_commit": summary.get("code_commit"), "files": entries}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = []
    for p in sorted(out.rglob("*")):
        rel = p.relative_to(out).as_posix()
        if p.is_file() and rel != "SHA256SUMS":
            lines.append(f"{sha256(p)}  {rel}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return manifest


def export(run: Path | str, out: Path | str, *, labels: Path | str | None = None, include_text: bool = True,
           force: bool = False, say=print) -> int:
    """Write the release folder from a finished pilot run.  Returns 0, or 1 when the result fails the check."""
    from .release_card import render_card

    run, out = Path(run), Path(out)
    files = find_run_files(run, Path(labels) if labels else None)
    missing = [k for k in ("steps", "labels") if files[k] is None or not Path(files[k]).exists()]
    if missing:
        raise FileNotFoundError(
            f"{run} has no {' or '.join(missing)} file. Run both pilot passes first "
            "(judge labels go in <run>/judge/labels.parquet), or pass --labels.")
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"{out} is not empty; pass --force to replace it")
        for p in sorted(out.rglob("*"), reverse=True):
            p.unlink() if p.is_file() or p.is_symlink() else p.rmdir()
    (out / DATA_DIR).mkdir(parents=True, exist_ok=True)

    steps = pd.read_parquet(files["steps"])
    labels_df = pd.read_parquet(files["labels"])
    assignments = pd.read_parquet(files["assignments"]) if files["assignments"] else None
    if assignments is None:
        say("export: no msm/assignments.parquet in the run; macrostate columns will be null")
    tables, stats = build_tables(steps, labels_df, assignments, include_text=include_text)

    leaks = {t: frame_leaks(df) for t, df in tables.items()}
    leaks = {t: v for t, v in leaks.items() if v}
    if leaks:
        raise ValueError("refusing to export: " + "; ".join(f"{t}.{x}" for t, v in leaks.items() for x in v))

    for t, df in tables.items():
        df.to_parquet(out / DATA_DIR / f"{t}.parquet", index=False)
    summary = run_summary(files, tables, stats, include_text)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (out / "SCHEMA.md").write_text(schema_markdown())
    (out / "README.md").write_text(render_card(summary))
    write_manifest(out, summary)
    c = summary["counts"]
    say(f"export: {c['proofs']} proofs, {c['steps']} steps, {c['labels']} labels -> {out}")
    problems = check(out)
    for p in problems:
        say(f"check: {p}")
    return 1 if problems else 0


# --------------------------------------------------------------------------- checking


def _card_header(text: str) -> tuple[dict | None, str | None]:
    import yaml

    if not text.startswith("---\n"):
        return None, "README.md does not start with a YAML header (---)"
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, "README.md YAML header is not closed (---)"
    try:
        meta = yaml.safe_load(text[4:end])
    except yaml.YAMLError as exc:
        return None, f"README.md YAML header does not parse: {exc}"
    return (meta, None) if isinstance(meta, dict) else (None, "README.md YAML header is not a mapping")


def check(folder: Path | str, *, allow_text_paths: bool = False) -> list[str]:
    """Every problem found in a release folder; an empty list means it is complete."""
    root = Path(folder)
    problems: list[str] = []
    if not root.is_dir():
        return [f"{root} is not a folder"]
    for rel in FILES_REQUIRED:
        if not (root / rel).is_file():
            problems.append(f"missing {rel}")
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}

    # checksums
    sums: dict[str, str] = {}
    if (root / "SHA256SUMS").is_file():
        for n, line in enumerate((root / "SHA256SUMS").read_text().splitlines(), 1):
            m = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line.strip())
            if not m:
                problems.append(f"SHA256SUMS line {n} is malformed")
                continue
            sums[m.group(2)] = m.group(1)
        for rel, digest in sums.items():
            p = root / rel
            if not p.is_file():
                problems.append(f"SHA256SUMS lists {rel}, which is missing")
            elif sha256(p) != digest:
                problems.append(f"checksum mismatch for {rel}")
        for rel in sorted(present - set(sums) - {"SHA256SUMS"}):
            problems.append(f"{rel} is not in SHA256SUMS")

    # manifest
    manifest: dict = {}
    if (root / "MANIFEST.json").is_file():
        try:
            manifest = json.loads((root / "MANIFEST.json").read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"MANIFEST.json does not parse: {exc}")
    listed = {e.get("path"): e for e in manifest.get("files", []) if isinstance(e, dict)}
    for rel in sorted(present - set(listed) - {"MANIFEST.json", "SHA256SUMS"}):
        problems.append(f"{rel} is not in MANIFEST.json")
    for rel, e in listed.items():
        p = root / str(rel)
        if not p.is_file():
            problems.append(f"MANIFEST.json lists {rel}, which is missing")
            continue
        if e.get("sha256") != sha256(p):
            problems.append(f"MANIFEST.json checksum mismatch for {rel}")
        if e.get("bytes") != p.stat().st_size:
            problems.append(f"MANIFEST.json size mismatch for {rel}")

    # tables
    tables: dict[str, pd.DataFrame] = {}
    for t, cols in TABLES.items():
        rel = f"{DATA_DIR}/{t}.parquet"
        if not (root / rel).is_file():
            continue
        try:
            df = pd.read_parquet(root / rel)
        except Exception as exc:  # a corrupt file is a problem to report, not a crash
            problems.append(f"{rel} does not read: {exc}")
            continue
        tables[t] = df
        want = [c for c, _, _, _ in cols]
        if list(df.columns) != want:
            problems.append(f"{rel} columns {list(df.columns)} differ from the schema {want}")
            continue
        for c, dtype, nullable, _ in cols:
            s = df[c]
            ok = {"string": pd.api.types.is_string_dtype(s) or s.dtype == object or s.isna().all(),
                  "int64": pd.api.types.is_integer_dtype(s) or s.isna().all(),
                  "float64": pd.api.types.is_float_dtype(s) or s.isna().all()}[dtype]
            if not ok:
                problems.append(f"{rel} column {c} has type {s.dtype}, expected {dtype}")
            if not nullable and s.isna().any():
                problems.append(f"{rel} column {c} has {int(s.isna().sum())} nulls")
        e = listed.get(rel)
        if e is not None and e.get("rows") != len(df):
            problems.append(f"MANIFEST.json says {rel} has {e.get('rows')} rows, the file has {len(df)}")
        for leak in frame_leaks(df, paths=not allow_text_paths):
            problems.append(f"{rel} {leak}")
    if set(tables) == set(TABLES) and not any("columns" in p for p in problems):
        st, lab, pr = tables["steps"], tables["labels"], tables["proofs"]
        if st.duplicated(KEY).any():
            problems.append(f"steps has {int(st.duplicated(KEY).sum())} repeated (proof_id, step)")
        if lab.duplicated(KEY + ["judge"]).any():
            problems.append(f"labels has {int(lab.duplicated(KEY + ['judge']).sum())} repeated (proof_id, step, judge)")
        if pr["proof_id"].duplicated().any():
            problems.append("proofs has repeated proof_id")
        bad = lab["msc"].map(lambda c: msc.normalize(c) != c)
        if bad.any():
            problems.append(f"labels has {int(bad.sum())} codes that are not two-digit MSC 2020 codes")
        conf = lab["confidence"].dropna()
        if ((conf < 0) | (conf > 1)).any():
            problems.append("labels has confidence outside 0 to 1")
        keys = set(zip(st["proof_id"], st["step"]))
        step_rows = lab[lab["step"] >= 0]
        orphans = sum(k not in keys for k in zip(step_rows["proof_id"], step_rows["step"]))
        if orphans:
            problems.append(f"labels has {orphans} step labels with no matching steps row")
        if (lab["step"] < PROOF_LEVEL_STEP).any():
            problems.append("labels has steps below -1")
        if not set(lab["proof_id"]) <= set(pr["proof_id"]):
            problems.append("labels has proof_ids that are not in proofs")
        if set(st["proof_id"]) != set(pr["proof_id"]):
            problems.append("steps and proofs list different proof_ids")
        counts = st.groupby("proof_id").size()
        if not (pr.set_index("proof_id")["n_steps"].astype(int) == counts.reindex(pr["proof_id"]).values).all():
            problems.append("proofs.n_steps does not match the steps table")
        if len(lab) == 0:
            problems.append("labels is empty")

    # card
    if (root / "README.md").is_file():
        meta, err = _card_header((root / "README.md").read_text())
        if err:
            problems.append(err)
        else:
            for k in CARD_KEYS_REQUIRED:
                if k not in meta:
                    problems.append(f"README.md YAML header has no {k}")
            for cfg in meta.get("configs", []) or []:
                for df_ in (cfg.get("data_files") or []) if isinstance(cfg, dict) else []:
                    path = df_.get("path") if isinstance(df_, dict) else df_
                    if isinstance(path, str) and path not in present:
                        problems.append(f"README.md config {cfg.get('config_name')} points at {path}, which is missing")
        if "TODO" in (root / "README.md").read_text():
            problems.append("README.md still has TODO items for the author")

    # every text file: no secrets, no local paths
    for rel in sorted(present):
        if rel.endswith((".md", ".json", ".txt", ".yaml", ".yml", ".csv")) or rel == "SHA256SUMS":
            for leak in find_leaks((root / rel).read_text(errors="replace")):
                problems.append(f"{rel} contains a {leak}")
    extra = sorted(p for p in present if p.startswith(f"{DATA_DIR}/") and p not in FILES_REQUIRED)
    for rel in extra:
        problems.append(f"unexpected file {rel} (not part of the schema)")
    return problems


# --------------------------------------------------------------------------- CLI


def export_main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="markov-lean export-dataset",
                                description="Write the labelled dataset release folder from a finished pilot run.")
    p.add_argument("--run", required=True, help="pilot output folder, e.g. results/pilot3")
    p.add_argument("--out", default="release", help="release folder to write (default release/)")
    p.add_argument("--labels", default=None, help="judge labels parquet if not in <run>/judge/labels.parquet")
    p.add_argument("--no-text", action="store_true", help="leave tactic and goal-state text out")
    p.add_argument("--force", action="store_true", help="replace a non-empty --out folder")
    a = p.parse_args(list(argv) if argv is not None else None)
    try:
        return export(a.run, a.out, labels=a.labels, include_text=not a.no_text, force=a.force)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"export-dataset: {exc}", file=sys.stderr)
        return 2


def check_main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="markov-lean check-release",
                                description="Check that a release folder is complete and safe to upload.")
    p.add_argument("folder", nargs="?", default="release")
    p.add_argument("--allow-todo", action="store_true", help="do not count TODO items in README.md")
    a = p.parse_args(list(argv) if argv is not None else None)
    problems = check(a.folder)
    if a.allow_todo:
        problems = [x for x in problems if "TODO" not in x]
    for x in problems:
        print(x)
    print(f"{a.folder}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("export", "export-dataset"):
        return export_main(argv[1:])
    if argv and argv[0] in ("check", "check-release"):
        return check_main(argv[1:])
    print("usage: python -m markov_lean.release {export|check} ...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
