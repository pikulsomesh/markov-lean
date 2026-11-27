"""Cross-area projects in the paper (free labels now, judge labels after Dec 1) and scripts/after_case_judges.sh.

No API calls: judge labels are written by hand, and the script only collects runs that have batches.json.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("seaborn")

from markov_lean import case_study as CS  # noqa: E402
from markov_lean import paper as P  # noqa: E402
from markov_lean.data import STEP_COLUMNS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "after_case_judges.sh"

# The Mac's exact extractions (Nov 27): proofs, steps, modules, premise_area switches/move, % switch, dwell.
KNOWN = {"sphere_packing": (784, 6159, 64, 0.560, 0.510, 1.5, 0.580),
         "pnt": (1644, 26791, 92, 0.497, 0.600, 1.8, 0.517),
         "carleson": (2153, 32510, 117, 0.500, 0.560, 1.7, 0.525)}


def _summary(d: Path, proofs, steps, modules, rate, has, dwell, own, own_key="premise_own_area"):
    d.mkdir(parents=True, exist_ok=True)
    sw = {"premise_area": {"switches_per_move": rate, "share_proofs_with_switch": has, "mean_dwell": dwell},
          own_key: {"switches_per_move": own, "share_proofs_with_switch": has, "mean_dwell": dwell}}
    (d / "summary.json").write_text(json.dumps({"case": d.name, "proofs": proofs, "steps": steps,
                                                "modules": modules, "switches": sw}))


def _case_with_judges(d: Path, n_proofs: int, p_switch: float, seed: int):
    """A case folder with a steps table and consensus judge labels; areas switch with probability p_switch."""
    rng = np.random.default_rng(seed)
    rows, labs = [], []
    for i in range(n_proofs):
        area = int(rng.integers(3))
        for s in range(int(rng.integers(4, 10))):
            if s and rng.random() < p_switch:
                area = (area + 1 + int(rng.integers(2))) % 3
            rows.append({c: None for c in STEP_COLUMNS} | {"proof_id": f"{d.name}:t{i}", "step": s,
                                                           "source": d.name, "split": d.name,
                                                           "theorem": f"t{i}", "file_path": f"{d.name}/F.lean"})
            labs.append({"proof_id": f"{d.name}:t{i}", "step": s, "judge": "consensus",
                         "msc": ["05", "11", "26"][area], "confidence": 0.9})
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[STEP_COLUMNS].to_parquet(d / "steps.parquet", index=False)
    (d / "api").mkdir(exist_ok=True)
    pd.DataFrame(labs).to_parquet(d / "api" / "labels.parquet", index=False)


@pytest.fixture
def paper_dir(tmp_path):
    d = tmp_path / "paper"
    shutil.copytree(ROOT / "paper" / "figures" / "preview", d / "figures" / "preview")
    return d


def _macros(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}  %", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def test_preview_table_matches_code_and_the_paper_uses_it():
    prev = (ROOT / "paper" / "figures" / "preview" / "table_projects.tex").read_text()
    assert prev == P.HEADER + " (preview).\n" + P.table_projects_preview_tex()
    assert "table_projects" in P.TABLES
    text = (ROOT / "paper" / "sections" / "projects.tex").read_text()
    assert r"\resultstable{table_projects}" in text
    assert r"\input{sections/projects}" in (ROOT / "paper" / "sections" / "appendix.tex").read_text()
    assert set(re.findall(r"\\(res[A-Z]\w*)", text)) <= set(P.MACROS)


def test_free_labels_fill_the_table_and_judge_columns_wait(tmp_path, paper_dir):
    res = tmp_path / "results"
    for name, vals in KNOWN.items():
        _summary(res / "cases" / name, *vals)
    items = P.run(res, paper_dir, verbose=False)
    tab = (paper_dir / "figures" / "table_projects.tex").read_text()
    lines = [ln.strip().removesuffix(r"\\").strip() for ln in tab.splitlines()
             if "&" in ln and not ln.strip().startswith(("&", "Project"))]
    row = {ln.split("&")[0].strip(): [c.strip() for c in ln.split("&")[1:]] for ln in lines}
    assert row[r"Sphere packing (dim.\ 8)"][:6] == ["784", "6{,}159", "0.560", "51.0", "1.5", "0.580"]
    assert row["PNT+"][:6] == ["1{,}644", "26{,}791", "0.497", "60.0", "1.8", "0.517"]
    assert row["Carleson (control)"][:6] == ["2{,}153", "32{,}510", "0.500", "56.0", "1.7", "0.525"]
    assert all(c == r"\tbd" for c in row["PFR"])                          # no results/pfr
    assert all(c == r"\tbd" for r in row.values() for c in r[6:])         # judge columns wait on the labels
    m = _macros(paper_dir / "figures" / "results_macros.tex")
    assert m["resCaseSphereModules"] == "64" and m["resCasePNTPremRate"] == "0.497"
    assert m["resCaseCarlesonPremHas"] == "56.0" and m["resCaseCarlesonJudgeRate"] == r"\tbd"
    assert m["resCaseSphereVsCtlRate"] == r"\tbd"
    assert next(i for i in items if i.name == "table_projects").status == "real"


def test_no_cases_keeps_the_preview(tmp_path, paper_dir):
    P.run(tmp_path / "none", paper_dir, verbose=False)
    assert not (paper_dir / "figures" / "table_projects.tex").exists()


def test_compare_against_the_control(tmp_path):
    _case_with_judges(tmp_path / "a", 120, 0.6, 1)
    _case_with_judges(tmp_path / "ctl", 120, 0.1, 2)
    from markov_lean import switches as S

    for n in ("a", "ctl"):
        d = tmp_path / n
        S.run(d, d / "switches", steps_path=d / "steps.parquet", labels_path=d / "api" / "labels.parquet",
              sources=["judge"], nulls=["shuffle"], n_perm=19, n_boot=20, markov_group="none",
              pooled_within="all", figures=False, verbose=False)
    r = CS.compare({"a": tmp_path / "a", "ctl": tmp_path / "ctl", "gone": tmp_path / "gone"}, "ctl",
                   n_boot=200, n_perm=200)
    assert r["missing"] == ["gone"] and set(r["cases"]) == {"a", "ctl"}
    d = r["vs_control"]["a"]["switch_rate"]
    assert d["difference"] > 0.3 and d["ci95"][0] > 0 and d["p_two_sided"] < 0.01
    a = r["cases"]["a"]
    assert a["ratio"] == pytest.approx(a["switch_rate"] / a["shuffle"])
    assert "ctl" not in r["vs_control"]


sh_only = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _sh(*args, env=None):
    base = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    e = {**base, "PY": sys.executable, **(env or {})}
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, env=e, capture_output=True, text=True,
                          timeout=900)


@sh_only
def test_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


@sh_only
def test_script_end_to_end(tmp_path, paper_dir):
    """Labels in for two cases, one case not submitted: switch tests, comparison, paper; nothing submitted."""
    res = tmp_path / "results"
    for name, vals in KNOWN.items():
        _summary(res / "cases" / name, *vals)
    _case_with_judges(res / "cases" / "sphere_packing", 80, 0.6, 3)
    _case_with_judges(res / "cases" / "carleson", 80, 0.1, 4)
    pnt = res / "cases" / "pnt"
    pd.read_parquet(res / "cases" / "carleson" / "steps.parquet").to_parquet(pnt / "steps.parquet")
    env = {"RESULTS": str(res), "PAPER": str(paper_dir), "SKIP_PDF": "1", "GUARD_LOCK": "", "PAPER_ARGS": "",
           "N_PERM": "19", "N_BOOT": "20", "ANTHROPIC_API_KEY": "should-not-be-used"}
    p = _sh(env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "pnt: not submitted yet" in p.stdout and "still waiting on: pnt" in p.stdout
    assert "batches.create" not in p.stdout + p.stderr
    c = json.loads((res / "cases" / "compare.json").read_text())
    assert c["control"] == "carleson" and set(c["cases"]) == {"sphere_packing", "carleson"}
    assert c["vs_control"]["sphere_packing"]["switch_rate"]["difference"] > 0
    m = _macros(paper_dir / "figures" / "results_macros.tex")
    assert m["resCaseSphereJudgeRate"] != r"\tbd" and m["resCaseSphereVsCtlRateP"] != r"\tbd"
    assert m["resCasePNTJudgeRate"] == r"\tbd"
    assert (res / "cases" / "sphere_packing" / "api" / "agreement.txt").exists()
    st = _sh("--status", env=env)
    assert "sphere_packing: labels in; switch test done" in st.stdout
    assert "pnt: not submitted; no switch test yet" in st.stdout
    # a rerun skips what is done
    p2 = _sh(env=env)
    assert p2.returncode == 0 and "switch test on judge labels" not in p2.stdout
