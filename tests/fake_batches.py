"""A stand-in for the Anthropic client's Message Batches API, for tests of scripts/after_judges.sh.

`client()` reads the JSON file named by FAKE_BATCHES:
{"status": "ended", "results": [{"custom_id": ..., "proof_msc": ..., "steps": [{"step": .., "msc": ..}]}]}
Every call is appended to FAKE_BATCHES_CALLS (one name per line), so a test can check that nothing
but `retrieve` and `results` was used.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace


def _log(name: str) -> None:
    path = os.environ.get("FAKE_BATCHES_CALLS")
    if path:
        with open(path, "a") as f:
            f.write(name + "\n")


class _Batches:
    def __init__(self, spec: dict):
        self.spec = spec

    def retrieve(self, bid: str):
        _log("retrieve")
        n = len(self.spec["results"])
        done = self.spec.get("status", "ended") == "ended"
        counts = SimpleNamespace(processing=0 if done else n, succeeded=n if done else 0, errored=0, expired=0,
                                 canceled=0)
        return SimpleNamespace(id=bid, processing_status=self.spec.get("status", "ended"), request_counts=counts)

    def results(self, bid: str):
        _log("results")
        for r in self.spec["results"]:
            if r.get("batch", bid) != bid:
                continue
            text = json.dumps({"proof_msc": r["proof_msc"], "proof_confidence": 0.9, "steps": r["steps"]})
            msg = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])
            yield SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=msg))

    def create(self, **kw):  # must never be reached through the read-only wrapper
        _log("create")
        raise AssertionError("batches.create called")

    def cancel(self, bid: str):
        _log("cancel")
        raise AssertionError("batches.cancel called")


def client():
    spec = json.loads(open(os.environ["FAKE_BATCHES"]).read())
    messages = SimpleNamespace(batches=_Batches(spec), create=lambda **kw: _log("messages.create"))
    return SimpleNamespace(messages=messages)
