"""Judge-context control: does a judge's area coherence come from reading the steps in order?

``python -m markov_lean.context_control label|compare|status|export``

The API judges see a whole proof in one request and are told that bookkeeping goals take the area
of the surrounding argument. So the low switch rate of their step labels (0.115 per move against
0.162 for a within-proof shuffle) could partly be the judge smoothing along the order it reads,
not coherence in the proofs. This control labels the same proofs twice with the local Ollama judge
(``markov_lean.ollama_judge``, no API call, no cost):

* **ordered**: the proof exactly as the API judges see it;
* **shuffled**: the same steps, the same goals and tactics, the same theorem, statement and final
  state, but shown in a random order with random step ids, so neither the order nor the numbers
  say where a step sits in the proof. Answers are mapped back to the true step numbers.

``compare`` then measures switching in the true step order under both conditions, on the steps
labelled in both. Each condition is compared with its own within-proof shuffle expectation (exact,
from ``switches.expected_shuffle_rate``), so a change in the area mix does not pass for a change in
order. If the judge's coherence came from smoothing along the order it read, the shuffled labels
lose it: their switch rate rises toward the shuffle expectation (ratio toward 1). If it comes from
the steps themselves, the ratio stays where the ordered labels put it.

The local judge agrees only weakly with the API judges (Cohen's kappa about 0.3), so this tests
the effect for a 7B judge, not for Claude itself; the same control with the API judges, about $3
on the same proofs, is the stronger version.

Output folder (default ``results/context_control/<model>``): ``sample.json``, ``progress.jsonl``
(one line per proof with both answers and the permutation; this makes the run resumable),
``labels.parquet`` (judges ``<local judge>#ordered`` and ``#shuffled``, true step numbers),
``summary.json``, and from ``compare`` ``comparison.json``, ``proofs.csv`` and ``REPORT.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import judge as J
from . import ollama_judge as oj
from . import prompts as P
from .switches import KEY, Sequences, expected_shuffle_rate, per_proof_stats

CONDITIONS = ("ordered", "shuffled")
DEFAULT_PROOFS = 300
DEFAULT_RUN = "results/pilot3"
RETRY_ON_RESUME = oj.RETRY_ON_RESUME
ID_RANGE = (100, 1000)  # shown ids for shuffled steps: random three-digit numbers, so they carry no position


def default_out(model: str, prompt: str = P.DEFAULT_VERSION) -> Path:
    name = model.replace(":", "_").replace("/", "_")
    if prompt != P.DEFAULT_VERSION:
        name += "_" + prompt
    return Path("results") / "context_control" / name


def condition_judge(model: str, prompt: str, condition: str) -> str:
    return f"{oj.judge_name(model, prompt)}#{condition}"


# --------------------------------------------------------------------------- the shuffled view


def _rng(proof_id: str, seed: int) -> random.Random:
    h = hashlib.sha256(f"{seed}:{proof_id}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def shuffled_view(proof: J.Proof, seed: int = 0) -> tuple[J.Proof, dict[int, int], list[int]]:
    """The proof with its shown steps in a random order and random ids.

    Returns the new proof, the map from shown id to the true step number, and the order (true step
    numbers in the order shown). The order is never the original one when there are two or more
    steps. Everything else the prompt shows stays the same: theorem, statement, every step's tactic
    and goal, and the state after the proof's true last shown step (the prompt prints the state after
    the last step in the list, so it is moved onto the step now shown last).
    """
    steps = list(proof.steps)
    n = len(steps)
    rng = _rng(proof.proof_id, seed)
    order = list(range(n))
    if n >= 2:
        while order == list(range(n)):
            rng.shuffle(order)
    ids = rng.sample(range(*ID_RANGE), n)
    final_state = steps[-1].state_after if steps else ""
    shown: list[J.Step] = []
    back: dict[int, int] = {}
    for k, (i, new_id) in enumerate(zip(order, ids)):
        st = steps[i]
        shown.append(replace(st, step=new_id, state_after=final_state if k == n - 1 else ""))
        back[new_id] = st.step
    return replace(proof, steps=shown), back, [steps[i].step for i in order]


def _rename(rows: list[dict], judge: str, back: dict[int, int] | None = None) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r, judge=judge)
        if back is not None and int(r["step"]) != J.PROOF_LEVEL_STEP:
            r["step"] = back[int(r["step"])]
        out.append(r)
    return out


def label_both(client: oj.OllamaClient, proof: J.Proof, s: oj.LocalSettings, seed: int = 0,
               sleep: Callable[[float], None] = time.sleep, only: tuple[str, ...] = CONDITIONS) -> dict:
    """Label one proof in each condition in ``only``; rows carry the true step numbers."""
    rec: dict = {"proof_id": proof.proof_id}
    if "ordered" in only:
        res = oj.label_proof(client, proof, s, sleep=sleep)
        rec["ordered"] = dict(res.to_json(), rows=_rename(res.rows, condition_judge(s.model, s.prompt, "ordered")))
    if "shuffled" in only:
        view, back, order = shuffled_view(proof, seed)
        res = oj.label_proof(client, view, s, sleep=sleep)
        rec["shuffled"] = dict(res.to_json(),
                               rows=_rename(res.rows, condition_judge(s.model, s.prompt, "shuffled"), back))
        rec["shown_order"] = order
    return rec


# --------------------------------------------------------------------------- the run


def read_progress(path: Path) -> dict[str, dict]:
    """Per proof, the last finished answer of each condition (a cut-off last line is ignored)."""
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        cur = done.setdefault(str(rec["proof_id"]), {"proof_id": str(rec["proof_id"])})
        for c in CONDITIONS:
            if c in rec:
                cur[c] = rec[c]
        if "shown_order" in rec:
            cur["shown_order"] = rec["shown_order"]
    return done


def _todo(rec: dict | None, retry_failed: bool) -> tuple[str, ...]:
    need = []
    for c in CONDITIONS:
        r = (rec or {}).get(c)
        if r is None or r["status"] in RETRY_ON_RESUME or (retry_failed and r["status"] != "ok"):
            need.append(c)
    return tuple(need)


def plan(steps: pd.DataFrame, cfg: J.JudgeConfig, labels: pd.DataFrame | None, n_proofs: int,
         seed: int) -> list[J.Proof]:
    """``n_proofs`` proofs with two or more shown steps, from the API judges' proofs when their labels
    are given, stratified by their proof label (or Mathlib folder), in interleaved order."""
    ordered = oj.plan_sample(steps, cfg, labels, "auto", None, seed)
    return [p for p in ordered if len(p.steps) >= 2][:n_proofs]


def run_label(steps: pd.DataFrame, out: str | Path, settings: oj.LocalSettings, *, client: oj.OllamaClient,
              cfg: J.JudgeConfig, labels: pd.DataFrame | None = None, n_proofs: int = DEFAULT_PROOFS,
              seed: int = 0, retry_failed: bool = False, limit_proofs: int | None = None,
              echo: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep,
              progress_every: int = 5) -> int:
    """Label the sample in both conditions; 0 when every proof has both answers, 1 otherwise."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    version = client.version()
    if not oj.has_model(client.models(), settings.model):
        raise oj.OllamaError(f"model {settings.model!r} is not pulled; run: ollama pull {settings.model}")
    if not oj.supports_schema(version):
        settings.schema = False
        echo(f"Ollama {version} predates structured outputs (0.5); using JSON mode plus repair.")

    steps = steps.assign(proof_id=steps["proof_id"].astype(str))
    proofs = plan(steps, cfg, labels, n_proofs, seed)
    by_id = {p.proof_id: p for p in proofs}
    ids = [p.proof_id for p in proofs]
    sp = out / oj.SAMPLE
    if sp.exists():
        old = json.loads(sp.read_text())
        if old.get("model") != settings.model or old.get("prompt") != settings.prompt:
            raise oj.OllamaError(f"{out} holds labels from {old.get('model')} with prompt {old.get('prompt')}; "
                                 "use another --out")
    sp.write_text(json.dumps({"model": settings.model, "prompt": settings.prompt, "n_proofs": n_proofs,
                              "seed": seed, "proof_ids": ids,
                              "settings": {k: v for k, v in vars(settings).items()}}))

    done = read_progress(out / oj.PROGRESS)
    todo = [(i, _todo(done.get(i), retry_failed)) for i in ids]
    todo = [(i, need) for i, need in todo if need]
    if limit_proofs is not None:
        todo = todo[:limit_proofs]
    calls = sum(len(need) for _, need in todo)
    echo(f"Context control, {oj.judge_name(settings.model, settings.prompt)} (Ollama {version}): "
         f"{len(ids)} proofs, {len(ids) - len(todo)} done, {len(todo)} to label now ({calls} requests).")

    t0 = time.monotonic()
    stopped = False
    try:
        with open(out / oj.PROGRESS, "a") as log:
            for k, (pid, need) in enumerate(todo, 1):
                rec = label_both(client, by_id[pid], settings, seed, sleep=sleep, only=need)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if k % progress_every == 0 or k == len(todo):
                    per = (time.monotonic() - t0) / k
                    echo(f"[{k}/{len(todo)}] {per:.1f} s per proof (both conditions), "
                         f"about {per * (len(todo) - k) / 60:.0f} min left")
                if any(rec[c]["status"] == "server_error" for c in need):
                    echo(f"Ollama kept failing on {pid}. Stopping; run again to resume.")
                    stopped = True
                    break
    except KeyboardInterrupt:  # pragma: no cover - manual stop
        echo("Stopped. Run the same command again to resume.")
        stopped = True
    export(out, echo=echo)
    left = [i for i in ids if _todo(read_progress(out / oj.PROGRESS).get(i), False)]
    return 1 if stopped or left else 0


def export(out: str | Path, echo: Callable[[str], None] = print) -> dict:
    """labels.parquet and summary.json from the progress log."""
    out = Path(out)
    done = read_progress(out / oj.PROGRESS)
    rows = [r for rec in done.values() for c in CONDITIONS for r in (rec.get(c) or {}).get("rows", [])]
    labels = J.labels_frame(rows)
    labels.to_parquet(out / "labels.parquet", index=False)
    sample = json.loads((out / oj.SAMPLE).read_text()) if (out / oj.SAMPLE).exists() else {}
    status: dict[str, dict[str, int]] = {c: {} for c in CONDITIONS}
    secs: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    for rec in done.values():
        for c in CONDITIONS:
            if c in rec:
                status[c][rec[c]["status"]] = status[c].get(rec[c]["status"], 0) + 1
                if rec[c].get("seconds"):
                    secs[c].append(rec[c]["seconds"])
    both = sum(all(rec.get(c, {}).get("status") == "ok" for c in CONDITIONS) for rec in done.values())
    summary = {"model": sample.get("model"), "prompt": sample.get("prompt"),
               "proofs_in_sample": len(sample.get("proof_ids", [])), "proofs_ok_in_both": int(both),
               "status": status,
               "mean_seconds_per_request": {c: round(sum(v) / len(v), 2) if v else None for c, v in secs.items()},
               "labels": int(len(labels))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    echo(f"{len(labels)} labels written to {out / 'labels.parquet'}; {both} proofs ok in both conditions")
    return summary


# --------------------------------------------------------------------------- compare


def _cond_areas(labels: pd.DataFrame, judge: str) -> pd.DataFrame:
    lab = labels[(labels["judge"].astype(str) == judge) & (labels["step"] >= 0)]
    lab = lab.assign(proof_id=lab["proof_id"].astype(str), step=lab["step"].astype("int64"))
    return lab.drop_duplicates(KEY)[KEY].assign(
        area=lab.drop_duplicates(KEY)["msc"].astype(str).str.zfill(2).values).reset_index(drop=True)


def proof_stats(areas: pd.DataFrame) -> pd.DataFrame:
    """Per proof: labelled steps, switches, switch rate, its exact shuffle expectation, has_switch,
    share off the main area."""
    seq = Sequences(areas)
    st = per_proof_stats(seq, seq.y[None, :])
    return pd.DataFrame({"proof_id": seq.proofs, "length": seq.length,
                         "n_switches": st["n_switches"][0], "switch_rate": st["switch_rate"][0],
                         "shuffle_rate": expected_shuffle_rate(seq), "has_switch": st["has_switch"][0],
                         "off_mode": st["off_mode"][0]})


def _summ(df: pd.DataFrame) -> dict:
    rate, shuf = float(df["switch_rate"].mean()), float(df["shuffle_rate"].mean())
    return {"proofs": int(len(df)), "steps": int(df["length"].sum()), "switch_rate": rate,
            "shuffle_rate": shuf, "ratio": rate / shuf if shuf > 0 else float("nan"),
            "has_switch": float(df["has_switch"].mean()), "off_mode": float(df["off_mode"].mean())}


def _boot(fn: Callable[[np.ndarray], float], n: int, B: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
    draws = draws[np.isfinite(draws)]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if len(draws) else [float("nan")] * 2


def _kappa(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    from sklearn.metrics import cohen_kappa_score

    m = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    if m.empty:
        return {"n": 0, "kappa": float("nan"), "raw_agreement": float("nan")}
    x, y = m["area_a"].astype(str), m["area_b"].astype(str)
    with warnings.catch_warnings():  # one shared label leaves kappa undefined (nan), which is the answer
        warnings.simplefilter("ignore")
        k = float(cohen_kappa_score(x, y))
    return {"n": int(len(m)), "kappa": k, "raw_agreement": float((x == y).mean())}


def compare(out: str | Path, api_labels: str | Path | None = None, n_boot: int = 2000, seed: int = 0,
            echo: Callable[[str], None] = print) -> dict:
    """Switching in true step order, ordered vs shuffled presentation, on steps labelled in both."""
    out = Path(out)
    labels = pd.read_parquet(out / "labels.parquet")
    sample = json.loads((out / oj.SAMPLE).read_text())
    model, prompt = sample["model"], sample.get("prompt", P.DEFAULT_VERSION)
    done = read_progress(out / oj.PROGRESS)
    ok = {p for p, rec in done.items() if all(rec.get(c, {}).get("status") == "ok" for c in CONDITIONS)}

    a = {c: _cond_areas(labels, condition_judge(model, prompt, c)) for c in CONDITIONS}
    a = {c: v[v["proof_id"].isin(ok)] for c, v in a.items()}
    common = a["ordered"][KEY].merge(a["shuffled"][KEY], on=KEY)
    a = {c: v.merge(common, on=KEY) for c, v in a.items()}
    stats = {c: proof_stats(v) for c, v in a.items()}
    merged = stats["ordered"].merge(stats["shuffled"], on="proof_id", suffixes=("_ordered", "_shuffled"))
    n = len(merged)

    def ratio(c: str) -> Callable[[np.ndarray], float]:
        r, s = merged[f"switch_rate_{c}"].to_numpy(), merged[f"shuffle_rate_{c}"].to_numpy()
        return lambda i: r[i].mean() / s[i].mean() if s[i].mean() > 0 else np.nan

    ro, rs = ratio("ordered"), ratio("shuffled")
    gap = {c: (merged[f"shuffle_rate_{c}"] - merged[f"switch_rate_{c}"]).to_numpy() for c in CONDITIONS}
    res: dict = {"judge": oj.judge_name(model, prompt), "proofs_ok_in_both": len(ok), "proofs_compared": n,
                 "steps_compared": int(len(common)), "conditions": {}}
    for c in CONDITIONS:
        s = _summ(stats[c][stats[c]["proof_id"].isin(merged["proof_id"])])
        s["ratio_ci"] = _boot(ro if c == "ordered" else rs, n, n_boot, seed) if n else [float("nan")] * 2
        res["conditions"][c] = s
    if n:
        idx = np.arange(n)
        res["ratio_difference"] = float(rs(idx) - ro(idx))
        res["ratio_difference_ci"] = _boot(lambda i: rs(i) - ro(i), n, n_boot, seed + 1)
        res["gap_difference"] = float((gap["ordered"] - gap["shuffled"]).mean())
        res["gap_difference_ci"] = _boot(lambda i: float((gap["ordered"][i] - gap["shuffled"][i]).mean()),
                                         n, n_boot, seed + 2)
    res["step_agreement_ordered_vs_shuffled"] = _kappa(a["ordered"], a["shuffled"])
    po = labels[labels["step"] == J.PROOF_LEVEL_STEP]
    pa = {c: po[po["judge"].astype(str) == condition_judge(model, prompt, c)][["proof_id", "msc"]]
          .rename(columns={"msc": "area"}).assign(step=J.PROOF_LEVEL_STEP) for c in CONDITIONS}
    res["proof_agreement_ordered_vs_shuffled"] = _kappa(pa["ordered"], pa["shuffled"])

    if api_labels is not None and Path(api_labels).exists():
        from .bridges import judge_areas

        api, used = judge_areas(pd.read_parquet(api_labels))
        api = api.assign(proof_id=api["proof_id"].astype(str), step=api["step"].astype("int64"))
        api = api[api["proof_id"].isin(merged["proof_id"])]
        res["api"] = {"labels": used, "agreement_with_ordered": _kappa(a["ordered"], api),
                      "agreement_with_shuffled": _kappa(a["shuffled"], api)}
        if len(api):
            res["api"]["switching_same_proofs"] = _summ(proof_stats(api))

    merged.to_csv(out / "proofs.csv", index=False)
    (out / "comparison.json").write_text(json.dumps(res, indent=1))
    (out / "REPORT.md").write_text(report(res))
    echo(report(res))
    return res


def _f(x, d=3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def report(res: dict) -> str:
    o, s = res["conditions"]["ordered"], res["conditions"]["shuffled"]
    lines = [
        f"# Judge-context control: {res['judge']}",
        "",
        f"{res['proofs_compared']} proofs and {res['steps_compared']} steps labelled in both conditions "
        f"({res['proofs_ok_in_both']} proofs answered cleanly in both). Switching is measured in the true "
        "step order; each ratio is the mean switch rate over the mean exact within-proof shuffle rate.",
        "",
        "| condition | switches per move | shuffle | ratio [95% CI] | proofs with a switch | share off main area |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in (("steps in order", o), ("steps shuffled", s)):
        lines.append(f"| {name} | {_f(c['switch_rate'])} | {_f(c['shuffle_rate'])} | {_f(c['ratio'], 2)} "
                     f"[{_f(c['ratio_ci'][0], 2)}, {_f(c['ratio_ci'][1], 2)}] | {_f(c['has_switch'])} "
                     f"| {_f(c['off_mode'])} |")
    api = res.get("api", {}).get("switching_same_proofs")
    if api:
        lines.append(f"| API judges ({res['api']['labels']}), same proofs | {_f(api['switch_rate'])} | "
                     f"{_f(api['shuffle_rate'])} | {_f(api['ratio'], 2)} | {_f(api['has_switch'])} | "
                     f"{_f(api['off_mode'])} |")
    if "ratio_difference" in res:
        lo, hi = res["ratio_difference_ci"]
        lines += ["", f"Shuffled minus ordered ratio: {_f(res['ratio_difference'], 2)} [{_f(lo, 2)}, {_f(hi, 2)}] "
                      "(bootstrap over proofs)."]
    k = res["step_agreement_ordered_vs_shuffled"]
    kp = res["proof_agreement_ordered_vs_shuffled"]
    lines += ["", f"Step labels, ordered vs shuffled: kappa {_f(k['kappa'], 2)}, raw agreement "
                  f"{_f(k['raw_agreement'], 2)} on {k['n']} steps. Proof labels: kappa {_f(kp['kappa'], 2)} "
                  f"on {kp['n']} proofs."]
    if "api" in res:
        ao, as_ = res["api"]["agreement_with_ordered"], res["api"]["agreement_with_shuffled"]
        lines.append(f"Agreement with the API judges' step labels: ordered kappa {_f(ao['kappa'], 2)} "
                     f"({ao['n']} steps), shuffled kappa {_f(as_['kappa'], 2)} ({as_['n']} steps).")
    lines += ["", "Reading: if the judge's coherence came from smoothing along the order it read, the shuffled "
                  "ratio rises toward 1. A shuffled ratio that stays near the ordered one says the coherence "
                  "is in the steps, for this judge. The local judge agrees only weakly with the API judges "
                  "(kappa about 0.3), so this is weaker evidence than the same control with the API judges.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- status and CLI


def complete(out: str | Path) -> bool:
    """Every proof in the sample has an answer in both conditions."""
    out = Path(out)
    if not (out / oj.SAMPLE).exists():
        return False
    ids = json.loads((out / oj.SAMPLE).read_text()).get("proof_ids", [])
    done = read_progress(out / oj.PROGRESS)
    return bool(ids) and not any(_todo(done.get(i), False) for i in ids)


def status(out: str | Path) -> str:
    out = Path(out)
    if not (out / oj.SAMPLE).exists():
        return f"no context-control run in {out} yet"
    sample = json.loads((out / oj.SAMPLE).read_text())
    done = read_progress(out / oj.PROGRESS)
    ids = sample.get("proof_ids", [])
    finished = sum(1 for i in ids if not _todo(done.get(i), False))
    msg = f"context control: {finished}/{len(ids)} proofs labelled in both conditions ({out})"
    if (out / "comparison.json").exists():
        c = json.loads((out / "comparison.json").read_text())["conditions"]
        msg += (f"; ratio ordered {_f(c['ordered']['ratio'], 2)}, shuffled {_f(c['shuffled']['ratio'], 2)}")
    return msg


def main(argv: list[str] | None = None, client: oj.OllamaClient | None = None,
         sleep: Callable[[float], None] = time.sleep) -> int:
    ap = argparse.ArgumentParser(prog="python -m markov_lean.context_control",
                                 description="Judge-context control with the local Ollama judge (no API calls).")
    ap.add_argument("--config", default=str(J.DEFAULT_CONFIG), help="judge config (prompt limits)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label", help="label the sample with steps in order and shuffled (resumable)")
    lb.add_argument("--run", default=DEFAULT_RUN, help=f"pilot run folder (default {DEFAULT_RUN})")
    lb.add_argument("--steps", help="steps table (default <run>/judge/sample_steps.parquet)")
    lb.add_argument("--labels", help="API judges' labels, to sample their proofs (default <run>/judge/labels.parquet)")
    lb.add_argument("--proofs", type=int, default=DEFAULT_PROOFS, help=f"proofs in the sample (default {DEFAULT_PROOFS})")
    lb.add_argument("--host", default=oj.DEFAULT_HOST)
    lb.add_argument("--model", default=oj.DEFAULT_MODEL)
    lb.add_argument("--prompt", choices=list(P.PROMPTS), default=P.DEFAULT_VERSION,
                    help="v1, the API judges' prompt (default), or v2")
    lb.add_argument("--num-ctx", type=int, default=oj.LocalSettings.num_ctx)
    lb.add_argument("--timeout", type=float, default=600.0)
    lb.add_argument("--seed", type=int, default=0)
    lb.add_argument("--limit-proofs", type=int)
    lb.add_argument("--retry-failed", action="store_true")
    lb.add_argument("--out")
    cp = sub.add_parser("compare", help="switching in true order, ordered vs shuffled")
    cp.add_argument("--out", required=True)
    cp.add_argument("--run", default=DEFAULT_RUN)
    cp.add_argument("--labels", help="API judges' labels (default <run>/judge/labels.parquet)")
    cp.add_argument("--n-boot", type=int, default=2000)
    cp.add_argument("--partial", action="store_true", help="compare before every proof is labelled")
    for name in ("status", "export"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name == "status":
            p.add_argument("--check", action="store_true", help="exit 1 unless labelling is complete")
    args = ap.parse_args(argv)

    if args.cmd == "status":
        print(status(args.out))
        return 0 if not args.check or complete(args.out) else 1
    if args.cmd == "export":
        export(args.out)
        return 0
    if args.cmd == "compare":
        if not args.partial and not complete(args.out):
            print(f"labelling in {args.out} is not finished; resume it, or pass --partial", file=sys.stderr)
            return 1
        lp = args.labels or str(Path(args.run) / "judge" / "labels.parquet")
        compare(args.out, lp, n_boot=args.n_boot)
        return 0

    steps_path = args.steps or str(Path(args.run) / "judge" / "sample_steps.parquet")
    if not Path(steps_path).exists():
        ap.error(f"no steps table at {steps_path}")
    lp = args.labels or str(Path(args.run) / "judge" / "labels.parquet")
    labels = J._read_table(lp) if Path(lp).exists() else None
    cfg = J.load_config(args.config)
    client = client or oj.OllamaClient(args.host, args.timeout)
    settings = oj.LocalSettings(model=args.model, num_ctx=args.num_ctx, prompt=args.prompt, seed=args.seed)
    out = args.out or default_out(args.model, args.prompt)
    try:
        return run_label(J._read_table(steps_path), out, settings, client=client, cfg=cfg, labels=labels,
                         n_proofs=args.proofs, seed=args.seed, retry_failed=args.retry_failed,
                         limit_proofs=args.limit_proofs, sleep=sleep)
    except oj.OllamaError as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
