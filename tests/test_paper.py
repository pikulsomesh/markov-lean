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


def make_results(root: Path, *, search: bool = True, switches: bool = True) -> Path:
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
    if switches:  # `markov-lean switches` on the pilot itself: judge labels and macrostates
        from markov_lean import switches as S

        S.run(pilot, n_perm=19, n_boot=20, verbose=False)
    (root / "bridges").mkdir()
    shutil.copyfile(REPO / "paper" / "figures" / "preview" / "area_flow.pdf", root / "bridges" / "bridges.pdf")
    if search:
        _write_search(root / "search_eval")
        rep = root / "search_eval" / "report"  # what `markov-lean search-report` leaves next to the runs
        rep.mkdir()
        (rep / "paired.tex").write_text("\\begin{tabular}{l}\\end{tabular}\n")
        shutil.copyfile(REPO / "paper" / "figures" / "preview" / "area_flow.pdf", rep / "fig_budget.pdf")
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
    assert items[("extra", "bridges.pdf")].status == "real" and (figs / "bridges.pdf").exists()
    assert (figs / "search_paired.tex").exists() and (figs / "search_budget.pdf").exists()
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
    assert "resSearchGainProofNet" not in v  # the paper uses PutnamBench, so no ProofNet macros unless it runs
    assert int(v["resFlowsShown"]) >= 1 and 0 < int(v["resFlowCoverage"]) <= 100
    head = json.loads((results / "pilot3" / "switches" / "switches.json").read_text())["headline"]
    assert v["resSwitchJudgeProofs"] == "30" and v["resSwitchMacroProofs"] == "30"
    assert v["resSwitchJudgeHas"] == f"{100 * head['judge']['has_switch']:.1f}"
    assert v["resSwitchMacroRate"] == f"{head['macrostate']['switch_rate']:.3f}"
    assert v["resSwitchMacroRateShuffleP"] == f"{head['macrostate']['p_switch_rate_shuffle']:.3f}"
    assert v["resSwitchJudgeHasLo"] == f"{100 * head['judge']['has_switch_ci95'][0]:.1f}"

    data = (figs / "table_data.tex").read_text()
    assert "8{,}423 & 101{,}234" in data and "200 & 3{,}000" in data
    assert re.search(r"PutnamBench\s+& 8 ", data)  # counted from the search runs
    areas = (figs / "table_areas.tex").read_text()
    assert "($m = 3$)" in areas and r"0.31 & \tbd & 0.20" in areas and "0.87 & -- & 0.80" in areas
    search = (figs / "table_search.tex").read_text()
    assert "miniF2F & PutnamBench" in search
    lines = [ln for ln in search.splitlines() if "&" in ln and "Prior" not in ln]
    assert lines[0].startswith(r"  LM only") and "25.0 & 25.0" in lines[0]
    assert "Area prior" in lines[1] and "50.0 & 50.0" in lines[1]
    assert "Random scores" in lines[2]
    switches = (figs / "table_switches.tex").read_text()
    assert switches.startswith(P.HEADER) and "Judge MSC" in switches and "Macrostate" in switches
    assert r"\tbd" not in _block(switches, "Judge MSC") + _block(switches, "Macrostate")
    assert _block(switches, "Premise MSC") == "\n".join(P._switch_preview_rows("premise"))  # no premise run


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


# --------------------------------------------------------------------------- table 2 baselines


def _write_robustness_full(rob: Path) -> None:
    """What `markov-lean robustness` writes: every grouping with AMI and intervals, and the permutation nulls."""
    rob.mkdir(parents=True, exist_ok=True)
    rows = [("macrostate", 0.312, 0.21, 0.205), ("microstate", 0.40, 0.18, 0.06), ("folder", 0.11, 0.09, 0.05),
            ("folder2", 0.15, 0.10, 0.04), ("namespace", 0.13, 0.08, 0.03), ("namespace2", 0.17, 0.07, 0.02),
            ("kmeans_steps", 0.24, 0.16, 0.12), ("kmeans_microstates", 0.26, 0.17, 0.13),
            ("proof_label", 0.70, 0.66, 0.61), ("random", 0.004, -0.001, -0.002)]
    t = pd.DataFrame(rows, columns=["grouping", "nmi", "ami", "ari"])
    t["n_groups"] = 3
    t["nmi_lo"], t["nmi_hi"] = t["nmi"] - 0.02, t["nmi"] + 0.03
    t["ari_lo"], t["ari_hi"] = t["ari"] - 0.01, t["ari"] + 0.01
    t.to_csv(rob / "baselines.csv", index=False)
    (rob / "robustness.json").write_text(json.dumps({
        "judge_ceiling": {"nmi": 0.55, "kappa": 0.6},
        "baselines": {"null": {
            "observed": {"nmi": 0.312, "ari": 0.205}, "n_permutations": 1000,
            "row": {"nmi_mean": 0.0021, "nmi_q95": 0.004, "p_nmi": 0.000999, "ari_mean": 0.0001, "p_ari": 0.000999},
            "proof": {"nmi_mean": 0.081, "nmi_q95": 0.1, "p_nmi": 0.012, "ari_mean": 0.031, "p_ari": 0.02},
        }}}))


def _write_significance(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "significance.json").write_text(json.dumps({
        "partitions": {"msm": {"nmi": {"value": 0.3141, "ci95": [0.28, 0.35]},
                               "ari": {"value": 0.2, "ci95": [0.17, 0.23]}},
                       "folder": {"nmi": {"value": 0.11, "ci95": [0.09, 0.13]},
                                  "ari": {"value": 0.05, "ci95": [0.04, 0.06]}}},
        "shuffle_nulls": {"msm": {"nmi": 0.3141,
                                  "within_proof": {"null_mean": 0.29, "null_sd": 0.01, "z": 2.4, "p": 0.034},
                                  "global": {"null_mean": 0.003, "null_sd": 0.001, "z": 300.0, "p": 0.001}}},
    }))


def _ci(v: str, interval: str) -> str:
    return r"\begin{tabular}[t]{@{}c@{}}" + v + r"\\[-1pt]{\scriptsize[" + interval + r"]}\end{tabular}"


def _area_rows(path: Path) -> dict[str, list[str]]:
    out = {}
    for line in path.read_text().splitlines():
        if "&" in line and "multicolumn" not in line and "Labeling" not in line:
            cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
            out[cells[0]] = cells[1:]
    return out


def test_table_areas_lists_every_robustness_baseline_and_null(tmp_path, paper_dir):
    results = make_results(tmp_path / "results", search=False)
    _write_robustness_full(results / "pilot3" / "robustness")
    _write_significance(results / "pilot3" / "significance")
    assert P.discover(results).significance == results / "pilot3" / "significance"
    P.run(results, paper_dir, verbose=False)
    figs = paper_dir / "figures"
    rows = _area_rows(figs / "table_areas.tex")
    # the macrostates come from the robustness table (same steps as every baseline), with its intervals
    assert rows["Macrostates ($m = 3$)"] == [_ci("0.31", "0.29, 0.34"), "0.21",
                                              _ci("0.20", "0.19, 0.21")]
    assert rows["Mathlib folder"][0] == _ci("0.11", "0.09, 0.14") and rows["Mathlib folder"][1] == "0.09"
    for label in ("Mathlib folder, 2 levels", "Theorem namespace", "Theorem namespace, 2 levels",
                  "$k$-means on goal states", "$k$-means on microstates", "Microstates",
                  "Judges' proof label", "Random partition"):
        assert r"\tbd" not in " ".join(rows[label]), label
    assert rows["$k$-means on goal states"][1] == "0.16" and rows["Random partition"][1] == "0.00"
    assert rows[r"\quad steps shuffled"] == ["0.00", "--", "0.00"]
    assert rows[r"\quad proofs swapped"] == ["0.08", "--", "0.03"]
    assert rows[r"\quad shuffled within each proof"] == ["0.29", "--", "--"]
    assert r"\tbd" not in (figs / "table_areas.tex").read_text()

    v = macros(figs / "results_macros.tex")
    assert v["resMacroNMI"] == "0.31" and v["resMacroAMI"] == "0.21"  # headline NMI stays the pilot's
    assert v["resMacroNMILo"] == "0.29" and v["resMacroNMIHi"] == "0.34"
    assert v["resFolderAMI"] == "0.09" and v["resNamespaceNMI"] == "0.13" and v["resNamespaceAMI"] == "0.08"
    assert v["resKmeansNMI"] == "0.24" and v["resKmeansAMI"] == "0.16"
    assert v["resNullRowNMI"] == "0.00" and v["resNullRowP"] == "0.001"
    assert v["resNullProofNMI"] == "0.08" and v["resNullProofP"] == "0.012"
    assert v["resNullWithinNMI"] == "0.29" and v["resNullWithinP"] == "0.034"


def test_table_areas_from_significance_alone(tmp_path, paper_dir):
    """No robustness run: the pilot's rows get the significance intervals, the rest stay red."""
    results = make_results(tmp_path / "results", search=False)
    shutil.rmtree(results / "pilot3" / "robustness")
    _write_significance(results / "significance")
    P.run(results, paper_dir, verbose=False)
    rows = _area_rows(paper_dir / "figures" / "table_areas.tex")
    assert rows["Macrostates ($m = 3$)"] == [_ci("0.31", "0.28, 0.35"), r"\tbd",
                                              _ci("0.20", "0.17, 0.23")]
    assert rows["Mathlib folder"][0] == _ci("0.11", "0.09, 0.13")
    assert rows["Theorem namespace"] == [r"\tbd"] * 3 and rows["$k$-means on goal states"] == [r"\tbd"] * 3
    assert rows[r"\quad steps shuffled"] == ["0.00", "--", r"\tbd"]  # the significance global shuffle
    assert rows[r"\quad proofs swapped"] == [r"\tbd", "--", r"\tbd"]
    assert rows[r"\quad shuffled within each proof"] == ["0.29", "--", "--"]
    v = macros(paper_dir / "figures" / "results_macros.tex")
    assert v["resNullRowP"] == "0.001" and v["resNullProofP"] == r"\tbd" and v["resMacroNMILo"] == "0.28"


def test_table_areas_preview_matches_code():
    prev = REPO / "paper" / "figures" / "preview" / "table_areas.tex"
    assert prev.read_text() == P.HEADER + " (preview).\n" + P.table_areas_tex({})
    rows = _area_rows(prev)
    assert len(rows) == len(P.AREA_ROWS) + len(P.NULL_ROWS) + 1
    # cells no run can fill print a plain dash; everything else waits in red
    assert rows["Synthetic, planted areas"] == [r"\tbd", "--", r"\tbd"]
    assert all(r"\tbd" in c for c in rows["Macrostates ($m = 10$)"])


def test_proofnet_macros_only_when_it_runs():
    assert not any("ProofNet" in m for m in P.MACROS)
    assert P.macro_suffix("proofnet") == "ProofNet" and P.macro_suffix("putnam") == "Putnam"


def test_tex_ci():
    assert P.tex_ci(0.312, 0.29, 0.34) == _ci("0.31", "0.29, 0.34")
    assert P.tex_ci(0.312) == "0.31" and P.tex_ci(float("nan"), 0.1, 0.2) == r"\tbd"
    assert P.tex_ci(-0.004, -0.01, 0.003) == _ci("0.00", "$-$0.01, 0.00")


def _switch_run(d: Path, sources=("macrostate",), table: bool = True, figure: bool = True) -> Path:
    """A hand-made `markov-lean switches` output folder."""
    d.mkdir(parents=True)
    head = {s: {"n_proofs": 8423, "has_switch": 0.0646, "has_switch_ci95": [0.0594, 0.0700], "switch_rate": 0.01166,
                "mean_dwell": 14.53, "off_mode": 0.01425, "switch_rate_shuffle": 0.02154,
                "p_switch_rate_shuffle": 0.001998, "has_switch_markov": 0.1137, "p_has_switch_markov": 0.001998,
                "has_switch_pooled": 0.5653, "p_has_switch_pooled": 0.001998} for s in sources}
    (d / "switches.json").write_text(json.dumps({"headline": head}))
    if table:
        (d / "table_switches.tex").write_text("\\begin{tabular}{l}\n  Macrostate \\\\\n\\end{tabular}\n")
    if figure:
        shutil.copyfile(REPO / "paper" / "figures" / "preview" / "switches.pdf", d / "switches.pdf")
    return d


def test_switches_discovery_and_override(tmp_path):
    results = make_results(tmp_path / "results", search=False)
    assert P.discover(results).switches == results / "pilot3" / "switches"  # judge labels as well as macrostates
    other = _switch_run(tmp_path / "real-switches" / "train_main")
    assert P.discover(results, switches=other).switches == other
    bare = tmp_path / "bare"
    _switch_run(bare / "switches")
    assert P.discover(bare).switches == bare / "switches"
    assert P.discover(tmp_path / "nothing").switches is None


def test_switches_fill_the_paper(tmp_path, paper_dir):
    run = _switch_run(tmp_path / "real-switches" / "train_main")
    items = {(i.kind, i.name): i for i in P.run(tmp_path / "none", paper_dir, switches=run, verbose=False)}
    figs = paper_dir / "figures"
    assert items[("figure", "switches")].status == "real" and (figs / "switches.pdf").exists()
    assert items[("table", "table_switches")].status == "real"
    assert "Macrostate" in (figs / "table_switches.tex").read_text()
    v = macros(figs / "results_macros.tex")
    assert v["resSwitchMacroProofs"] == "8{,}423" and v["resSwitchMacroHas"] == "6.5"
    assert v["resSwitchMacroHasLo"] == "5.9" and v["resSwitchMacroHasHi"] == "7.0"
    assert v["resSwitchMacroRate"] == "0.012" and v["resSwitchMacroDwell"] == "14.5"
    assert v["resSwitchMacroOffMode"] == "1.4" and v["resSwitchMacroHasPooled"] == "56.5"
    assert v["resSwitchMacroRateShuffleP"] == "0.002"
    # no judge labels and no noise floor in this run: those stay red dashes
    assert v["resSwitchJudgeHas"] == r"\tbd" and v["resSwitchMacroOffModeNoise"] == r"\tbd"
    # every other item is still a preview
    assert items[("figure", "implied_timescales")].status == "preview"


def test_switches_json_alone_rebuilds_the_table_and_keeps_the_figure_preview(tmp_path, paper_dir):
    results = make_results(tmp_path / "results", search=False)
    sw = results / "pilot3" / "switches"
    (sw / "table_switches.tex").unlink()
    (sw / "switches.pdf").unlink()
    items = {(i.kind, i.name): i.status for i in P.run(results, paper_dir, verbose=False)}
    assert items[("table", "table_switches")] == "real"
    assert "Judge MSC" in (paper_dir / "figures" / "table_switches.tex").read_text()
    assert items[("figure", "switches")] == "preview"


def test_switches_stale_table_goes_back_to_preview(tmp_path, paper_dir):
    run = _switch_run(tmp_path / "sw")
    P.run(tmp_path / "none", paper_dir, switches=run, verbose=False)
    assert (paper_dir / "figures" / "table_switches.tex").exists()
    P.run(tmp_path / "none", paper_dir, verbose=False)
    assert not (paper_dir / "figures" / "table_switches.tex").exists()


def test_table_switches_preview_matches_code():
    from markov_lean import switches as S

    prev = (REPO / "paper" / "figures" / "preview" / "table_switches.tex").read_text()
    assert prev == P.HEADER + " (preview).\n" + P.table_switches_preview_tex()
    rows = [ln for ln in prev.splitlines() if ln.count("&") == 2 + len(S.NULLS)]
    assert len(rows) == 1 + len(P.SWITCH_SOURCES) * len(S.STATS)  # header + one row per source and statistic
    assert all(name in P.MACROS for name in ("resSwitchJudgeHas", "resSwitchMacroRateShuffleP"))


def test_cli_paper_takes_every_input_flag(tmp_path, paper_dir):
    assert cli.PAPER_INPUTS == P.INPUTS
    run = _switch_run(tmp_path / "sw")
    assert cli.main(["paper", "--results", str(tmp_path / "none"), "--paper", str(paper_dir),
                     "--switches", str(run)]) == 0
    assert (paper_dir / "figures" / "table_switches.tex").exists()


def _accuracy_run(d: Path) -> Path:
    """A real `judge-accuracy` report on the committed gold set: haiku always right, sonnet always 26."""
    from markov_lean import gold as G
    from markov_lean import judge_accuracy as JA

    gold = G.read_gold()
    scored = gold[gold.status != "excluded"]
    fam = G.family_codes(scored)
    right = [m if isinstance(m, str) and m else sorted(f)[0] for m, f in zip(scored.msc, fam)]
    labels = pd.DataFrame({"proof_id": list(scored.id) * 2, "step": -1,
                           "judge": ["claude-haiku-5-5"] * len(scored) + ["claude-sonnet-5-5"] * len(scored),
                           "msc": right + ["26"] * len(scored)})
    rep = JA.report(labels, gold, n_boot=20)
    JA.write(rep, d, figures=False)
    return d


def test_accuracy_fills_the_gold_macros(tmp_path, paper_dir):
    run = _accuracy_run(tmp_path / "judge_accuracy")
    assert P.discover(tmp_path).accuracy == run
    P.run(tmp_path / "none", paper_dir, accuracy=run, verbose=False)
    v = macros(paper_dir / "figures" / "results_macros.tex")
    rep = json.loads((run / "judge_accuracy.json").read_text())
    st = rep["gold"]["status"]
    assert v["resGoldStrict"] == P.tex_int(st["strict"]) and v["resGoldFamily"] == P.tex_int(st["family"])
    assert v["resGoldProblems"] == P.tex_int(st["strict"] + st["family"])
    assert v["resGoldAccHaiku"] == "100.0" and v["resGoldFamAccHaiku"] == "100.0"
    sonnet = rep["judges"]["claude-sonnet-5-5"]["strict"]["accuracy"]
    assert v["resGoldAccSonnet"] == P.tex_num(100 * sonnet, 1)
    assert v["resGoldAccSonnetLo"] != r"\tbd" and v["resGoldMajority"] != r"\tbd"


def test_accuracy_missing_leaves_red_dashes(tmp_path, paper_dir):
    P.run(tmp_path / "none", paper_dir, verbose=False)
    v = macros(paper_dir / "figures" / "preview" / "results_macros.tex")
    assert all(v[k] == r"\tbd" for k in ("resGoldProblems", "resGoldAccHaiku", "resGoldFamAccSonnet"))


# --------------------------------------------------------------------------- premise-folder rows


def _block(tex: str, label: str) -> str:
    """The rows of one source in the switching table, from its label to the next rule."""
    lines = tex.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip().startswith(label + " &"))
    j = next(k for k in range(i + 1, len(lines)) if "rule" in lines[k])
    return "\n".join(lines[i:j])


def _premise_run(root: Path, min_steps: int = 1) -> Path:
    """A real `markov-lean premise-labels` output, as it lands under results/premise-switches."""
    from markov_lean import premise_areas as pa
    from test_premise_areas import synthetic_steps

    steps, idx = synthetic_steps(n_proofs=40)
    root.mkdir(parents=True, exist_ok=True)
    steps.to_parquet(root / "steps.parquet")
    idx.to_parquet(root / "idx.parquet")
    out = root / "premise-switches" / ("proofs_8plus" if min_steps > 1 else "all_proofs")
    pa.run([root / "steps.parquet"], out, index_paths=[root / "idx.parquet"], n_perm=10, n_boot=20,
           min_switch_steps=min_steps, figures=False, verbose=False)
    return out / "switches" / "train"


def test_premise_discovery_prefers_long_proofs(tmp_path):
    results = tmp_path / "results"
    assert P.discover(results).premise is None
    allp = _premise_run(results)
    assert P.discover(results).premise == allp
    long = _premise_run(results, min_steps=4)
    assert P.discover(results).premise == long
    assert P.discover(results, premise=allp).premise == allp


def test_premise_rows_without_a_switches_run(tmp_path, paper_dir):
    pytest.importorskip("seaborn")
    results = tmp_path / "results"
    run = _premise_run(results, min_steps=4)
    items = {(i.kind, i.name): i for i in P.run(results, paper_dir, verbose=False)}
    figs = paper_dir / "figures"
    tex = (figs / "table_switches.tex").read_text()
    assert items[("table", "table_switches")].status == "real"
    # judge and macrostate rows wait on their run; the premise rows are real and match the run's own table
    for src, label in (("judge", "Judge MSC"), ("macrostate", "Macrostate")):
        assert _block(tex, label) == "\n".join(P._switch_preview_rows(src))
    prem = _block(tex, "Premise MSC")
    assert r"\tbd" not in prem
    assert prem in (run / "table_switches.tex").read_text()
    assert tex.count(r"\midrule") == 3 and "Premise (carried)" not in tex  # the main source only
    # the figure is redrawn from the premise run, not the preview
    assert items[("figure", "switches")].status == "real" and items[("figure", "switches")].note == "premise"
    assert (figs / "switches.pdf").read_bytes()[:4] == b"%PDF"
    v = macros(figs / "results_macros.tex")
    head = json.loads((run / "switches.json").read_text())["headline"]["premise"]
    assert v["resSwitchPremiseProofs"] == P.tex_int(head["n_proofs"])
    assert v["resSwitchPremiseHas"] == f"{100 * head['has_switch']:.1f}"
    assert v["resSwitchPremiseOffModeNoise"] == f"{100 * head['off_mode_noise']:.1f}"
    cov = json.loads((run.parents[1] / "coverage.json").read_text())
    share = next(e["share_labelled"] for e in cov["by_split"] if e["split"] == "all" and e["source"] == "premise")
    assert v["resSwitchPremiseCoverage"] == f"{100 * share:.1f}"
    assert v["resSwitchPremiseMinSteps"] == "4"
    assert v["resSwitchJudgeHas"] == r"\tbd" and v["resSwitchMacroHas"] == r"\tbd"


def test_premise_rows_join_the_pilot_run(tmp_path, paper_dir):
    pytest.importorskip("seaborn")
    results = make_results(tmp_path / "results", search=False)
    _premise_run(results)
    items = {(i.kind, i.name): i for i in P.run(results, paper_dir, verbose=False)}
    tex = (paper_dir / "figures" / "table_switches.tex").read_text()
    labels = [ln.split("&")[0].strip() for ln in tex.splitlines() if "Proofs with a switch" in ln]
    assert labels == ["Judge MSC", "Macrostate", "Premise MSC"]
    assert r"\tbd" not in tex
    assert items[("figure", "switches")].note == "judge, macrostate, premise"
    v = macros(paper_dir / "figures" / "results_macros.tex")
    assert v["resSwitchJudgeProofs"] == "30" and v["resSwitchPremiseMinSteps"] == "1"
    assert v["resSwitchPremiseHas"] != r"\tbd"


def test_old_switches_run_keeps_its_own_table_and_figure(tmp_path, paper_dir):
    """A switches run that kept only its table and figure cannot take premise rows; it is used as it was."""
    results = tmp_path / "results"
    _premise_run(results)
    old = _switch_run(tmp_path / "real-switches" / "train_main")
    items = {(i.kind, i.name): i for i in P.run(results, paper_dir, switches=old, verbose=False)}
    figs = paper_dir / "figures"
    assert "Macrostate" in (figs / "table_switches.tex").read_text()
    assert (figs / "switches.pdf").read_bytes() == (old / "switches.pdf").read_bytes()
    assert items[("figure", "switches")].note == "macrostate"
    v = macros(figs / "results_macros.tex")
    assert v["resSwitchMacroHas"] == "6.5" and v["resSwitchPremiseHas"] != r"\tbd"


def test_premise_run_without_premise_source_is_a_warning(tmp_path, paper_dir):
    bad = tmp_path / "results" / "premise-switches" / "all_proofs" / "switches" / "train"
    bad.mkdir(parents=True)
    (bad / "switches.json").write_text(json.dumps({"headline": {"premise_strict": {"n_proofs": 3}}, "sources": {}}))
    b = P.build(P.discover(tmp_path / "results"))
    assert "premise" in b.failures and "table_switches" in b.tables and not b.tables["table_switches"][2]
    assert not any(k.startswith("resSwitchPremise") for k in b.macros)


def test_switch_preview_has_three_sources():
    tex = P.table_switches_preview_tex()
    assert [ln.split("&")[0].strip() for ln in tex.splitlines() if "Proofs with a switch" in ln] == \
        ["Judge MSC", "Macrostate", "Premise MSC"]
    assert _block(tex, "Macrostate").count("--") == 4 and "--" not in _block(tex, "Premise MSC")
