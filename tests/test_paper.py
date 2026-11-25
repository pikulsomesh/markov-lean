"""`markov-lean paper` on small hand-made result folders (no model fits, no API calls)."""

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("seaborn")

from markov_lean import cli  # noqa: E402
from markov_lean import paper as P  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def paper_dir(tmp_path):
    """A paper folder with the repo's preview figures, so no preview has to be redrawn."""
    d = tmp_path / "paper"
    shutil.copytree(REPO / "paper" / "figures" / "preview", d / "figures" / "preview")
    return d


def _write_msm(d: Path, m: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "msm_report.json").write_text(json.dumps({
        "lag": 1, "n_macrostates": m, "n_microstates": 40, "timescales_steps": [6.25, 3.1, 1.4],
        "checks": {"chapman_kolmogorov": {"passed": False}, "spectral_gap": {"passed": True}},
    }))
    rows = [(lag, k, t * (1 - 0.3 * np.exp(-lag))) for lag in (1, 2, 3) for k, t in enumerate([6.0, 3.0, 1.5], 1)]
    pd.DataFrame(rows, columns=["lag", "index", "timescale"]).to_csv(d / "implied_timescales.csv", index=False)
    T = np.full((m, m), 0.1 / (m - 1)) + np.eye(m) * (0.9 - 0.1 / (m - 1))
    np.save(d / "coarse_T.npy", T)
    ck = [(lag, i, j, np.linalg.matrix_power(T, lag)[i, j], np.linalg.matrix_power(T, lag)[i, j] + 0.01)
          for lag in (1, 2) for i in range(m) for j in range(m)]
    pd.DataFrame(ck, columns=["lag", "from_set", "to_set", "predicted", "estimated"]).to_csv(d / "ck_test.csv",
                                                                                            index=False)
    a = []
    for pid in range(30):
        area = int(rng.integers(m))
        for step in range(8):
            if rng.random() < 0.2:
                area = int(rng.integers(m))
            a.append((f"leandojo:t{pid}", step, area * 5, area, 0.9))
    pd.DataFrame(a, columns=["proof_id", "step", "microstate", "macrostate", "membership"]).to_parquet(
        d / "assignments.parquet", index=False)
    return pd.read_parquet(d / "assignments.parquet")


def _write_search(root: Path, gain: float = 0.25):
    for bench in ("minif2f", "putnam"):
        for cond in ("lm", "full", "random_l0.3"):
            for seed in (0, 1):
                d = root / bench / cond / f"seed{seed}"
                d.mkdir(parents=True)
                lines = []
                for i in range(8):
                    p = 0.25 + (gain if cond == "full" else 0.0)
                    solved = (i / 8) < p
                    lines.append(json.dumps({"proof_id": f"{bench}:p{i}", "theorem": f"p{i}", "source": bench,
                                             "split": "test", "solved": solved,
                                             "stop_reason": "solved" if solved else "budget",
                                             "expansions": 10, "elapsed": 1.0, "nodes": 5, "n_tactics": 2,
                                             "proof": ""}))
                (d / "summary.jsonl").write_text("\n".join(lines) + "\n")


def make_results(root: Path, *, search: bool = True) -> Path:
    pilot = root / "pilot3"
    assign = _write_msm(pilot / "msm")
    (pilot / "judge").mkdir(parents=True)
    lab = assign[["proof_id", "step"]].assign(judge="consensus", msc=assign["macrostate"].map(
        {0: "05", 1: "11", 2: "26"}), confidence=0.9)
    lab.to_parquet(pilot / "judge" / "labels.parquet", index=False)
    (pilot / "pilot_report.json").write_text(json.dumps({
        "data": {"source": "leandojo", "n_proofs": 8423, "n_steps": 101234},
        "msm": {"n_microstates": 40, "lag": 1, "n_macrostates": 3},
        "judge_agreement": {"judges": ["judge-a", "judge-b"], "step": {"kappa": 0.612, "agreement": 0.734}},
        "eval": {"n_labelled_steps": 240, "macrostates": {"nmi": 0.3141, "ari": 0.2}, "folder_baseline":
                 {"nmi": 0.11, "ari": 0.05}, "gate1": {"passed": True}},
    }))
    # an older unlabelled pilot that must not be picked over pilot3
    (root / "pilot2").mkdir()
    (root / "pilot2" / "pilot_report.json").write_text(json.dumps({"data": {"source": "leandojo", "n_proofs": 1,
                                                                            "n_steps": 1}}))
    synth = root / "pilot_synthetic"
    synth.mkdir()
    (synth / "pilot_report.json").write_text(json.dumps({
        "data": {"source": "synthetic", "n_proofs": 200, "n_steps": 3000},
        "eval": {"macrostates": {"nmi": 0.87, "ari": 0.8}}}))
    rob = pilot / "robustness"
    (rob / "figures").mkdir(parents=True)
    pd.DataFrame({"grouping": ["macrostate", "folder", "random"], "nmi": [0.31, 0.11, 0.004],
                  "ari": [0.2, 0.05, -0.001]}).to_csv(rob / "baselines.csv", index=False)
    (rob / "robustness.json").write_text(json.dumps({"judge_ceiling": {"nmi": 0.55, "kappa": 0.6}}))
    shutil.copyfile(REPO / "paper" / "figures" / "preview" / "area_flow.pdf", rob / "figures" / "baselines_nmi.pdf")
    (root / "bridges").mkdir()
    shutil.copyfile(REPO / "paper" / "figures" / "preview" / "area_flow.pdf", root / "bridges" / "bridges.pdf")
    if search:
        _write_search(root / "search_eval")
    return root


def macros(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}  %", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


# --------------------------------------------------------------------------- tests


def test_no_results_keeps_every_preview(tmp_path, paper_dir):
    items = P.run(tmp_path / "results", paper_dir, verbose=False)
    assert {i.status for i in items} == {"preview"}
    figs = paper_dir / "figures"
    assert not list(figs.glob("*.pdf")) and not list(figs.glob("*.tex"))
    prev = macros(figs / "preview" / "results_macros.tex")
    assert set(prev) == set(P.MACROS) and set(prev.values()) == {r"\tbd"}
    for name in P.TABLES:
        assert r"\tbd" in (figs / "preview" / f"{name}.tex").read_text()


def test_every_input_fills_figures_tables_and_numbers(tmp_path, paper_dir):
    results = make_results(tmp_path / "results")
    items = {(i.kind, i.name): i for i in P.run(results, paper_dir, verbose=False)}
    figs = paper_dir / "figures"
    for name in P.FIGURES:
        assert items[("figure", name)].status == "real", name
        assert (figs / f"{name}.pdf").stat().st_size > 1000
    assert items[("extra", "bridges")].status == "real" and (figs / "bridges.pdf").exists()
    assert (figs / "robustness_baselines_nmi.pdf").exists()

    v = macros(figs / "results_macros.tex")
    assert v["resMacroNMI"] == "0.31" and v["resFolderNMI"] == "0.11"
    assert v["resNumProofs"] == "8{,}423" and v["resNumSteps"] == "101{,}234"
    assert v["resLag"] == "1" and v["resNumMacrostates"] == "3" and v["resSlowestTimescale"] == "6.2"
    assert v["resChecksFailed"] == "1" and v["resGateOne"] == "passed"
    assert v["resJudgeKappa"] == "0.61" and v["resJudgeAgreement"] == "73"
    assert v["resRandomNMI"] == "0.00" and v["resRandomARI"] == "0.00"
    assert v["resJudgeCeilingNMI"] == "0.55" and v["resSyntheticNMI"] == "0.87"
    assert v["resSearchSeeds"] == "2"
    assert v["resSearchBaseMiniF"] == "25.0" and v["resSearchAreaMiniF"] == "50.0"
    assert v["resSearchGainMiniF"] == "$+25.0$~points"
    assert v["resSearchGainProofNet"] == r"\tbd"  # never run, so the text keeps its dash
    assert int(v["resFlowsShown"]) >= 1 and 0 < int(v["resFlowCoverage"]) <= 100

    data = (figs / "table_data.tex").read_text()
    assert "8{,}423 & 101{,}234" in data and "200 & 3{,}000" in data
    assert re.search(r"PutnamBench\s+& 8 ", data)  # counted from the search runs
    areas = (figs / "table_areas.tex").read_text()
    assert "($m = 3$)" in areas and "0.31 & 0.20" in areas and "0.87 & 0.80" in areas
    search = (figs / "table_search.tex").read_text()
    assert "miniF2F & PutnamBench" in search
    lines = [ln for ln in search.splitlines() if "&" in ln and "Prior" not in ln]
    assert lines[0].startswith(r"  LM only") and "25.0 & 25.0" in lines[0]
    assert "Area prior" in lines[1] and "50.0 & 50.0" in lines[1]
    assert "Random scores" in lines[2]


def test_discovery_prefers_labelled_pilot_and_full_model(tmp_path):
    results = make_results(tmp_path / "results", search=False)
    src = P.discover(results)
    assert src.pilot.name == "pilot3" and src.synthetic.name == "pilot_synthetic"
    assert src.msm == results / "pilot3" / "msm" and src.search is None
    full = results / "full"
    _write_msm(full / "msm", m=4, seed=1)
    (full / "full_report.json").write_text(json.dumps({"data": {"n_proofs": 20000, "n_steps": 300000}}))
    src = P.discover(results)
    assert src.full == full and src.msm == full / "msm"
    b = P.build(src)
    assert b.macros["resNumProofs"][0] == "20{,}000" and b.macros["resNumMacrostates"][0] == "4"
    # a flag overrides the search
    assert P.discover(results, msm=results / "pilot3" / "msm").msm == results / "pilot3" / "msm"


def test_partial_results_and_stale_tables(tmp_path, paper_dir):
    results = make_results(tmp_path / "results")
    P.run(results, paper_dir, verbose=False)
    assert (paper_dir / "figures" / "table_areas.tex").exists()
    only_search = tmp_path / "only_search"
    _write_search(only_search / "search_eval")
    for f in (paper_dir / "figures").glob("*.pdf"):
        f.unlink()
    items = {(i.kind, i.name): i.status for i in P.run(only_search, paper_dir, verbose=False)}
    assert items[("figure", "search_ablation")] == "real"
    assert items[("figure", "implied_timescales")] == "preview"
    assert items[("table", "table_areas")] == "preview"
    # the old generated table no longer has a run behind it, so the preview shows again
    assert not (paper_dir / "figures" / "table_areas.tex").exists()
    assert items[("table", "table_data")] == "real"  # PutnamBench count from the runs
    v = macros(paper_dir / "figures" / "results_macros.tex")
    assert v["resMacroNMI"] == r"\tbd" and v["resSearchGainPutnam"] == "$+25.0$~points"


def test_earlier_figure_is_reported_not_deleted(tmp_path, paper_dir):
    (paper_dir / "figures" / "transition_matrix.pdf").write_bytes(b"%PDF-1.4 old")
    items = {i.name: i.status for i in P.run(tmp_path / "none", paper_dir, verbose=False) if i.kind == "figure"}
    assert items["transition_matrix"] == "earlier"
    assert (paper_dir / "figures" / "transition_matrix.pdf").exists()


def test_dry_run_writes_nothing(tmp_path, paper_dir):
    results = make_results(tmp_path / "results")
    before = sorted(p.name for p in (paper_dir / "figures").rglob("*"))
    items = P.run(results, paper_dir, dry_run=True, verbose=False)
    assert sum(i.status == "real" for i in items if i.kind == "figure") == len(P.FIGURES)
    assert sorted(p.name for p in (paper_dir / "figures").rglob("*")) == before


def test_broken_input_stays_preview(tmp_path, paper_dir, capsys):
    results = make_results(tmp_path / "results", search=False)
    (results / "pilot3" / "msm" / "ck_test.csv").write_text("not,a,ck\n1,2,3\n")
    items = {i.name: i.status for i in P.run(results, paper_dir, verbose=True) if i.kind == "figure"}
    assert items["chapman_kolmogorov"] == "preview"
    assert items["judge_confusion"] == "real"
    assert "could not read the msm input" in capsys.readouterr().out
    with pytest.raises(KeyError):
        P.run(results, paper_dir, strict=True, verbose=False)


def test_number_formats():
    assert P.tex_num(-0.25) == "$-$0.25" and P.tex_num(-0.001) == "0.00" and P.tex_num(float("nan")) == r"\tbd"
    assert P.tex_int(1234567) == "1{,}234{,}567"
    assert P.tex_signed(-1.25) == "$-1.2$~points" and P.tex_signed(None) == r"\tbd"


def test_consecutive_flows_skip_gaps():
    df = pd.DataFrame({"proof_id": ["a"] * 4 + ["b"] * 2, "step": [0, 1, 3, 4, 0, 1],
                       "msc": ["05", "11", "26", "26", "11", "05"]})
    flows = P.consecutive_flows(df, "msc")
    assert set(map(tuple, flows[["from_area", "to_area"]].to_numpy())) == {("05", "11"), ("11", "05")}
    assert flows["count"].sum() == 2  # 11 -> 26 spans a missing step and 26 -> 26 is no change


def test_cli_paper(tmp_path, paper_dir):
    results = make_results(tmp_path / "results", search=False)
    assert cli.main(["paper", "--results", str(results), "--paper", str(paper_dir)]) == 0
    assert (paper_dir / "figures" / "judge_confusion.pdf").exists()


def test_paper_text_uses_only_known_macros_and_tables():
    used, tables = set(), set()
    for tex in (REPO / "paper" / "sections").glob("*.tex"):
        text = tex.read_text()
        used |= set(re.findall(r"\\(res[A-Z]\w*)", text))
        tables |= set(re.findall(r"\\resultstable\{(\w+)\}", text))
    assert used and used <= set(P.MACROS), used - set(P.MACROS)
    assert tables == set(P.TABLES)
    prev = REPO / "paper" / "figures" / "preview"
    for name in P.TABLES:
        assert (prev / f"{name}.tex").read_text().startswith(P.HEADER)
    # the committed preview files match what the code writes
    assert (prev / "results_macros.tex").read_text() == P.macros_tex({})
    assert (prev / "table_search.tex").read_text() == P.HEADER + " (preview).\n" + P.table_search_preview_tex()


def test_real_synthetic_pilot_feeds_the_paper(tmp_path, paper_dir):
    """The pilot's own output on synthetic data, read as the synthetic row and the model figures."""
    from markov_lean import pilot

    out = tmp_path / "results" / "pilot_synth"
    cfg = {"data": {"source": "synthetic", "synthetic": {"n_proofs": 120, "seed": 2}},
           "msm": {"n_microstates": 20, "lag": 1, "n_macrostates": 4, "error_samples": 0,
                   "featurizer_kwargs": {"n_components": 8}},
           "eval": {"topk": [1]}}
    pilot.run(cfg, out, verbose=False)
    items = {(i.kind, i.name): i.status
             for i in P.run(tmp_path / "results", paper_dir, msm=out / "msm", verbose=False)}
    assert items[("figure", "implied_timescales")] == "real"
    assert items[("figure", "chapman_kolmogorov")] == "real"
    assert items[("number", "resSyntheticNMI")] == "real"
    assert items[("number", "resMacroNMI")] == "preview"  # synthetic labels never stand in for the judges
