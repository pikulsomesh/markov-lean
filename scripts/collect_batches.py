"""Collect finished judge Message Batches without waiting and without spending anything.

Used by scripts/after_judges.sh. Reads <run>/batches.json and <run>/manifest.json (written by
`python -m markov_lean.judge submit`), checks every batch has ended, then writes
<run>/labels.parquet and <run>/issues.json through `markov_lean.judge`'s own collect step.

Only two Batches API calls are ever made, `retrieve` (status) and `results` (download), and both
are free. The client is wrapped so that any other call, such as `batches.create`, `batches.cancel`
or `messages.create`, raises before it reaches the API.

    python scripts/collect_batches.py --out results/pilot3/judge/run

Exit codes: 0 labels written; 2 missing files or no key; 3 some batch has not ended yet.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ALLOWED_BATCH_CALLS = ("retrieve", "results")


class SpendBlocked(RuntimeError):
    """Raised when anything other than reading a batch's status or results is attempted."""


class _ReadOnlyBatches:
    def __init__(self, batches: Any):
        self._batches = batches

    def __getattr__(self, name: str) -> Any:
        if name in ALLOWED_BATCH_CALLS:
            return getattr(self._batches, name)
        raise SpendBlocked(f"messages.batches.{name} is blocked: collecting only reads finished batches")


class _ReadOnlyMessages:
    def __init__(self, messages: Any):
        self.batches = _ReadOnlyBatches(messages.batches)

    def __getattr__(self, name: str) -> Any:
        raise SpendBlocked(f"messages.{name} is blocked: collecting makes no new requests")


class ReadOnlyClient:
    """Exposes only ``client.messages.batches.retrieve`` and ``.results``."""

    def __init__(self, client: Any):
        self.messages = _ReadOnlyMessages(client.messages)

    def __getattr__(self, name: str) -> Any:
        raise SpendBlocked(f"client.{name} is blocked: collecting only reads finished batches")


def _status(batch: Any) -> str:
    return batch["processing_status"] if isinstance(batch, dict) else batch.processing_status


def _load_factory(spec: str):
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def run(out: str | Path, client: Any = None, echo=print) -> int:
    from markov_lean import judge as J

    out = Path(out)
    for name in ("batches.json", "manifest.json"):
        if not (out / name).exists():
            echo(f"ERROR: {out / name} is missing. It is written when the batches are submitted; "
                 f"run this from the markov-lean folder on the machine that submitted them.")
            return 2
    batch_ids = json.loads((out / "batches.json").read_text())["batch_ids"]
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            echo("ERROR: ANTHROPIC_API_KEY is not set in this shell. Export it here (never in a file in this "
                 "repo) and rerun. Downloading finished batch results is free.")
            return 2
        client = J.make_client()
    client = ReadOnlyClient(client)

    not_done = []
    for bid in batch_ids:
        b = client.messages.batches.retrieve(bid)
        echo(J.batch_status_line(bid, b, 0).replace("[0 min] ", ""))
        if _status(b) != "ended":
            not_done.append(bid)
    if not_done:
        echo(f"{len(not_done)} of {len(batch_ids)} batches have not ended yet; rerun this later. Nothing was written.")
        return 3
    return J.main(["collect", "--out", str(out), "--no-wait"], client=client)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the judge run folder holding batches.json and manifest.json")
    ap.add_argument("--client-factory", default=os.environ.get("AFTER_JUDGES_CLIENT_FACTORY"),
                    help=argparse.SUPPRESS)  # tests only: module:function returning a fake client
    a = ap.parse_args(argv)
    client = _load_factory(a.client_factory)() if a.client_factory else None
    return run(a.out, client=client)


if __name__ == "__main__":
    sys.exit(main())
