"""Case studies for the paper: a few readable proofs that clearly cross MSC areas.

``markov-lean cases --run results/pilot3 --out paper/figures`` reads a pilot
run's judged sample and writes two files the appendix picks up:

* ``case_studies.pdf``: one strip per proof, a cell per step coloured by its
  area, with the tactic under each step and the switching tactics in bold;
* ``case_studies_table.tex``: a ``tabular`` listing theorem, Mathlib file, area
  path and switching tactics.

A step's area is the judges' label for its goal (``labels`` in docs/schema.md).
With two judges a step is *confident* when both give the same MSC code and the
lower of their two confidences is at least ``min_confidence``.  A *confident
switch* at step ``t`` is a pair of adjacent steps ``t`` and ``t + 1``, both
confident, with different areas; the tactic at ``t`` is the one that moved the
goal from one area to the other.

Proofs are eligible when every step was labelled, they are between
``min_steps`` and ``max_steps`` long, every tactic is at most
``max_tactic_chars`` characters, the judges agree on at least
``min_agreement`` of the steps and there are at least ``min_switches``
confident switches.  Eligible proofs are ranked by confident switches (more
first), then judge agreement (higher first), then length (shorter first), and
picked greedily so that no two cases come from the same Mathlib file or show
only area moves already shown, unless there are too few proofs to keep that up.

Nothing here calls an API.  ``synthetic_case_inputs`` builds a stand-in run
from ``markov_lean.synthetic`` with two simulated judges, for the paper's
layout preview and the tests.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import msc

KEY = ["proof_id", "step"]
FIGURE_NAME = "case_studies"
TABLE_NAME = "case_studies_table"
UNCERTAIN = "?"


# --------------------------------------------------------------------------- config


@dataclass
class CaseConfig:
    n_cases: int = 5
    min_confidence: float = 0.7
    min_agreement: float = 0.8
    min_switches: int = 1
    min_steps: int = 4
    max_steps: int = 14
    max_tactic_chars: int = 60
    diverse: bool = True
    judges: list[str] | None = None  # two judge ids; default: the first two in the labels

    @classmethod
    def from_dict(cls, d: dict | None) -> "CaseConfig":
        d = dict(d or {})
        unknown = sorted(set(d) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown case-study settings: {unknown}")
        return cls(**d)


# --------------------------------------------------------------------------- inputs


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"msc": str})
    raise ValueError(f"unsupported table format: {path}")


def _first(paths: Sequence[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def load_run(run: str | Path, steps_path: str | Path | None = None,
             labels_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The judged steps and their labels from a ``markov-lean pilot`` output folder.

    Looks for ``judge/sample_steps.parquet`` and ``judge/labels.parquet`` (or the
    ``judge/run/labels.parquet`` that ``collect`` writes before it is copied).
    """
    run = Path(run)
    sp = Path(steps_path) if steps_path else _first([run / "judge" / "sample_steps.parquet",
                                                    run / "sample_steps.parquet"])
    lp = Path(labels_path) if labels_path else _first([run / "judge" / "labels.parquet",
                                                     run / "judge" / "run" / "labels.parquet",
                                                     run / "labels.parquet"])
    if sp is None or not sp.exists():
        raise FileNotFoundError(f"no judged steps under {run}; run `markov-lean pilot --out {run}` first")
    if lp is None or not lp.exists():
        raise FileNotFoundError(f"no judge labels under {run}; collect the judge batches first "
                                f"(see {run / 'judge' / 'NEXT.md'})")
    steps = _read_table(sp)
    labels = _read_table(lp)
    for name, df, cols in (("steps", steps, ["proof_id", "step", "theorem", "file_path", "tactic"]),
                           ("labels", labels, ["proof_id", "step", "judge", "msc", "confidence"])):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name} table is missing columns {missing}")
    return steps, labels


# --------------------------------------------------------------------------- per step


def step_areas(labels: pd.DataFrame, min_confidence: float, judges: list[str] | None = None) -> pd.DataFrame:
    """One row per labelled step: ``area``, ``agree``, ``confidence`` and ``confident``.

    Two judges: ``area`` is the shared code where they agree (else the
    uncertain marker), ``confidence`` the lower of the two.  One judge, or a
    ready-made ``consensus`` judge: its own labels.
    """
    lab = labels[labels["step"].astype("int64") >= 0].copy()
    lab["proof_id"] = lab["proof_id"].astype(str)
    lab["step"] = lab["step"].astype("int64")
    lab["msc"] = lab["msc"].map(msc.normalize)
    lab = lab.dropna(subset=["msc"])
    present = list(dict.fromkeys(lab["judge"].astype(str)))
    if judges is None:
        others = [j for j in present if j != "consensus"]
        judges = others[:2] if len(others) >= 2 else (others or present)[:1]
    if len(judges) >= 2:
        a = lab[lab["judge"] == judges[0]].drop_duplicates(KEY)
        b = lab[lab["judge"] == judges[1]].drop_duplicates(KEY)
        if a.empty or b.empty:
            raise ValueError(f"judges {judges[:2]} not both in the labels (found {present})")
        p = a[KEY + ["msc", "confidence"]].merge(b[KEY + ["msc", "confidence"]], on=KEY, suffixes=("_a", "_b"))
        out = p[KEY].copy()
        out["agree"] = (p["msc_a"] == p["msc_b"]).to_numpy()
        out["area"] = np.where(out["agree"], p["msc_a"], UNCERTAIN)
        out["confidence"] = p[["confidence_a", "confidence_b"]].min(axis=1).to_numpy()
    else:
        one = lab[lab["judge"] == judges[0]].drop_duplicates(KEY)
        out = one[KEY].copy()
        out["agree"] = True
        out["area"] = one["msc"].to_numpy()
        out["confidence"] = one["confidence"].to_numpy()
    out["confidence"] = out["confidence"].astype(float).fillna(0.0)
    out["confident"] = out["agree"] & (out["confidence"] >= min_confidence)
    return out.sort_values(KEY).reset_index(drop=True)


def case_steps(steps: pd.DataFrame, areas: pd.DataFrame) -> pd.DataFrame:
    """Every step of every proof with its area (or ``?``) and whether a confident switch starts there."""
    s = steps[KEY + ["theorem", "file_path", "tactic"]].copy()
    s["proof_id"] = s["proof_id"].astype(str)
    s["step"] = s["step"].astype("int64")
    s = s.drop_duplicates(KEY).merge(areas, on=KEY, how="left")
    s["labelled"] = s["area"].notna()
    s["area"] = s["area"].fillna(UNCERTAIN)
    s["agree"] = s["agree"].fillna(False).astype(bool)
    s["confident"] = s["confident"].fillna(False).astype(bool)
    s["confidence"] = s["confidence"].fillna(0.0)
    s["file_path"] = s["file_path"].fillna("").astype(str)
    s["tactic"] = s["tactic"].fillna("").astype(str)
    s = s.sort_values(KEY).reset_index(drop=True)
    g = s.groupby("proof_id", sort=False)
    nxt_area, nxt_conf, nxt_step = g["area"].shift(-1), g["confident"].shift(-1), g["step"].shift(-1)
    s["switch"] = (s["confident"] & nxt_conf.fillna(False).astype(bool) & (nxt_step == s["step"] + 1)
                   & (nxt_area != s["area"]))
    s["switch_to"] = np.where(s["switch"], nxt_area, None)
    return s


def _area_path(rows: pd.DataFrame) -> list[str]:
    """Areas of the confident steps in order, repeats collapsed."""
    path: list[str] = []
    for a in rows.loc[rows["confident"], "area"]:
        if not path or path[-1] != a:
            path.append(a)
    return path


def proof_table(cs: pd.DataFrame, cfg: CaseConfig) -> pd.DataFrame:
    """One row per proof with the ranking fields and why it is or isn't eligible."""
    rows = []
    for pid, g in cs.groupby("proof_id", sort=False):
        n = len(g)
        sw = g[g["switch"]]
        rec = {
            "proof_id": pid,
            "theorem": str(g["theorem"].iloc[0]),
            "file_path": str(g["file_path"].iloc[0]),
            "n_steps": n,
            "n_labelled": int(g["labelled"].sum()),
            "agreement": float(g["agree"].mean()),
            "mean_confidence": float(g["confidence"].mean()),
            "n_switches": int(len(sw)),
            "switch_steps": [int(x) for x in sw["step"]],
            "switch_pairs": [(str(a), str(b)) for a, b in zip(sw["area"], sw["switch_to"])],
            "switch_tactics": [str(t) for t in sw["tactic"]],
            "area_path": _area_path(g),
            "max_tactic_chars": int(g["tactic"].str.len().max()),
        }
        why = []
        if rec["n_labelled"] < n:
            why.append("unlabelled steps")
        if not cfg.min_steps <= n <= cfg.max_steps:
            why.append("length")
        if rec["max_tactic_chars"] > cfg.max_tactic_chars:
            why.append("long tactic")
        if rec["agreement"] < cfg.min_agreement:
            why.append("judges disagree")
        if rec["n_switches"] < cfg.min_switches:
            why.append("no confident switch")
        rec["eligible"] = not why
        rec["excluded_because"] = ", ".join(why)
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["eligible", "n_switches", "agreement", "n_steps", "proof_id"],
                           ascending=[False, False, False, True, True], kind="stable").reset_index(drop=True)


def pick(proofs: pd.DataFrame, cfg: CaseConfig) -> pd.DataFrame:
    """The top ``n_cases`` eligible proofs, kept varied when ``cfg.diverse``."""
    if proofs.empty:
        return proofs
    ranked = proofs[proofs["eligible"]]
    if not cfg.diverse:
        return ranked.head(cfg.n_cases).reset_index(drop=True)
    chosen: list[int] = []
    files: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for idx, r in ranked.iterrows():
        if len(chosen) == cfg.n_cases:
            break
        new_pairs = set(r["switch_pairs"]) - pairs
        if (r["file_path"] and r["file_path"] in files) or not new_pairs:
            continue
        chosen.append(idx)
        files.add(r["file_path"])
        pairs |= set(r["switch_pairs"])
    for idx in ranked.index:  # too few varied proofs: fill by rank
        if len(chosen) == cfg.n_cases:
            break
        if idx not in chosen:
            chosen.append(idx)
    return ranked.loc[sorted(chosen, key=list(ranked.index).index)].reset_index(drop=True)


@dataclass
class Cases:
    picked: pd.DataFrame          # proof_table rows of the chosen proofs, in rank order
    steps: pd.DataFrame           # case_steps rows of the chosen proofs
    proofs: pd.DataFrame          # proof_table for every proof (for the report)
    config: CaseConfig = field(default_factory=CaseConfig)

    def summary(self) -> dict:
        p = self.proofs
        return {
            "config": asdict(self.config),
            "n_proofs": int(len(p)),
            "n_eligible": int(p["eligible"].sum()) if len(p) else 0,
            "excluded": (p.loc[~p["eligible"], "excluded_because"].str.split(", ").explode()
                         .value_counts().to_dict() if len(p) else {}),
            "picked": [
                {k: r[k] for k in ("proof_id", "theorem", "file_path", "n_steps", "n_switches", "agreement",
                                   "area_path", "switch_steps", "switch_tactics")}
                for _, r in self.picked.iterrows()
            ],
        }


def select(steps: pd.DataFrame, labels: pd.DataFrame, cfg: CaseConfig | None = None) -> Cases:
    cfg = cfg or CaseConfig()
    cs = case_steps(steps, step_areas(labels, cfg.min_confidence, cfg.judges))
    proofs = proof_table(cs, cfg)
    picked = pick(proofs, cfg)
    ids = list(picked["proof_id"]) if len(picked) else []
    chosen = cs[cs["proof_id"].isin(ids)].copy()
    chosen["proof_id"] = pd.Categorical(chosen["proof_id"], categories=ids, ordered=True)
    chosen = chosen.sort_values(KEY).reset_index(drop=True)
    chosen["proof_id"] = chosen["proof_id"].astype(str)
    return Cases(picked=picked, steps=chosen, proofs=proofs, config=cfg)


# --------------------------------------------------------------------------- figure


def short_name(theorem: str, limit: int = 28) -> str:
    """The theorem's last two name components, trimmed to ``limit`` characters."""
    parts = str(theorem).split(".")
    s = ".".join(parts[-2:]) if len(parts) > 1 else parts[0]
    return s if len(s) <= limit else "…" + s[-(limit - 1):]


def tactic_head(tactic: str, limit: int = 10) -> str:
    """The tactic's first word (``rw``, ``simp``, ``exact``), for the label under a cell."""
    t = str(tactic).strip().split()
    head = t[0] if t else ""
    return head if len(head) <= limit else head[: limit - 1] + "…"


def area_colors(areas: Sequence[str]) -> dict[str, str]:
    """A colour per area, in order of first appearance; the uncertain marker is gray."""
    from .figures import OTHER, color_for

    order = [a for a in dict.fromkeys(areas) if a != UNCERTAIN]
    colors = {a: color_for(i) for i, a in enumerate(order)}
    colors[UNCERTAIN] = OTHER
    return colors


def case_strips(steps: pd.DataFrame, *, max_steps: int | None = None, area_names: dict | None = None,
                width: float | None = None, row_height: float = 0.62):
    """One strip per proof: a cell per step coloured by area, the tactic's first word under it.

    ``steps`` is :func:`case_steps` output for the chosen proofs, in the order
    they should appear.  Cells where a confident switch starts are outlined
    and their tactic is bold.  Unconfident steps are gray with a ``?``.
    """
    from matplotlib.patches import Rectangle

    from .figures import TEXT, TEXT_SECONDARY, TEXT_WIDTH, _new_ax, _themed

    @_themed
    def draw():
        ids = list(dict.fromkeys(steps["proof_id"]))
        if not ids:
            raise ValueError("no case-study proofs to draw")
        longest = int(steps.groupby("proof_id").size().max())
        ncol = max(longest, max_steps or 0)
        colors = area_colors(steps["area"])
        names = area_names or {}
        w = width or TEXT_WIDTH
        fig, ax = _new_ax(None, (w, row_height * len(ids) + 0.45))
        label_w = 2.2  # cell units kept on the left for the theorem name
        cell, gap = 1.0, 0.08
        for r, pid in enumerate(ids):
            g = steps[steps["proof_id"] == pid].sort_values("step")
            y = -r
            ax.text(-0.25, y + 0.5, short_name(g["theorem"].iloc[0]), ha="right", va="center",
                    fontsize="small", color=TEXT)
            for i, s in enumerate(g.itertuples(index=False)):
                x = i * cell
                face = colors.get(s.area, colors[UNCERTAIN])
                ax.add_patch(Rectangle((x + gap / 2, y + 0.15), cell - gap, 0.7, facecolor=face, lw=0,
                                       alpha=1.0 if s.confident else 0.35))
                if s.switch:
                    ax.add_patch(Rectangle((x + gap / 2, y + 0.15), cell - gap, 0.7, fill=False,
                                           edgecolor=TEXT, lw=1.0))
                ax.text(x + cell / 2, y + 0.5, s.area if s.area != UNCERTAIN else "?", ha="center",
                        va="center", fontsize=5.5, color="white" if s.confident and _dark(face) else TEXT)
                ax.text(x + cell / 2, y + 0.07, tactic_head(s.tactic), ha="center", va="top", fontsize=4.5,
                        color=TEXT if s.switch else TEXT_SECONDARY,
                        fontweight="bold" if s.switch else "normal")
        ax.set_xlim(-label_w, ncol * cell + 0.1)
        ax.set_ylim(-len(ids) + 1 - 0.15, 1.0)
        ax.axis("off")
        shown = [a for a in colors if a != UNCERTAIN and a in set(steps["area"])]
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=colors[a], lw=0, label=f"{a} {names.get(a, _title(a))}") for a in shown]
        if (steps["area"] == UNCERTAIN).any() or (~steps["confident"]).any():
            handles.append(Patch(facecolor=colors[UNCERTAIN], alpha=0.35, lw=0, label="judges unsure"))
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=min(len(handles), 4),
                  fontsize=5.5, handlelength=1.0, columnspacing=1.0, borderaxespad=0.1, frameon=False)
        ax.text(ncol * cell, 1.0, "step →", ha="right", va="bottom", fontsize=5, color=TEXT_SECONDARY)
        fig.n_cases = len(ids)
        return fig

    return draw()


def _dark(color: str) -> bool:
    """Whether white text reads on ``color`` (relative luminance under 0.4)."""
    from matplotlib.colors import to_rgb

    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(color)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] < 0.4


def _title(code: str, limit: int = 24) -> str:
    try:
        t = msc.title(code)
    except KeyError:
        return ""
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------- table

# Lean's Unicode, for pdflatex.  Anything else outside ASCII becomes "?".
_UNICODE_TEX = {
    "→": r"$\to$", "←": r"$\leftarrow$", "↔": r"$\leftrightarrow$", "⊢": r"$\vdash$", "∀": r"$\forall$",
    "∃": r"$\exists$", "¬": r"$\neg$", "∧": r"$\wedge$", "∨": r"$\vee$", "≤": r"$\le$", "≥": r"$\ge$",
    "≠": r"$\ne$", "∈": r"$\in$", "∉": r"$\notin$", "⊆": r"$\subseteq$", "⊂": r"$\subset$", "∪": r"$\cup$",
    "∩": r"$\cap$", "∘": r"$\circ$", "⟨": r"$\langle$", "⟩": r"$\rangle$", "⁻¹": r"$^{-1}$", "‖": r"$\|$",
    "∑": r"$\sum$", "∏": r"$\prod$", "∫": r"$\int$", "ℝ": r"$\mathbb{R}$", "ℕ": r"$\mathbb{N}$",
    "ℤ": r"$\mathbb{Z}$", "ℚ": r"$\mathbb{Q}$", "ℂ": r"$\mathbb{C}$", "·": r"$\cdot$", "•": r"$\bullet$",
    "∞": r"$\infty$", "≃": r"$\simeq$", "≅": r"$\cong$", "λ": r"$\lambda$", "α": r"$\alpha$",
    "β": r"$\beta$", "γ": r"$\gamma$", "δ": r"$\delta$", "ε": r"$\varepsilon$", "π": r"$\pi$",
    "σ": r"$\sigma$", "φ": r"$\varphi$", "ψ": r"$\psi$", "ω": r"$\omega$", "μ": r"$\mu$", "↑": r"$\uparrow$",
    "…": r"\ldots{}", "²": r"$^2$", "₀": r"$_0$", "₁": r"$_1$", "₂": r"$_2$",
}
_TEX_SPECIAL = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
                "%": r"\%", "_": r"\_", "^": r"\^{}", "~": r"\~{}", "<": r"\textless{}", ">": r"\textgreater{}",
                "|": r"\textbar{}"}


def tex_escape(text: str, breaks: str = "") -> str:
    """Escape ``text`` for pdflatex; characters in ``breaks`` are followed by a break point."""
    out = []
    i = 0
    text = str(text)
    while i < len(text):
        if text[i:i + 2] in _UNICODE_TEX:
            out.append(_UNICODE_TEX[text[i:i + 2]])
            i += 2
            continue
        c = text[i]
        if c in _TEX_SPECIAL:
            out.append(_TEX_SPECIAL[c])
        elif c in _UNICODE_TEX:
            out.append(_UNICODE_TEX[c])
        elif ord(c) > 127:
            out.append("?")
        elif c in "\n\t":
            out.append(" ")
        else:
            out.append(c)
        if c in breaks:
            out.append(r"\allowbreak{}")
        i += 1
    return "".join(out)


def _tt(text: str) -> str:
    return r"\texttt{" + tex_escape(text, breaks="._/") + "}"


def latex_table(picked: pd.DataFrame) -> str:
    """A booktabs ``tabular`` of the chosen proofs, for ``\\input`` inside a table float."""
    lines = [
        "% Written by `markov-lean cases`; do not edit by hand.",
        r"\begin{tabular}{@{}p{0.24\linewidth}p{0.25\linewidth}p{0.12\linewidth}p{0.31\linewidth}@{}}",
        r"\toprule",
        r"Theorem & Mathlib file & Area path & Switching tactic (step) \\",
        r"\midrule",
    ]
    for _, r in picked.iterrows():
        path = r"\,$\to$\,".join(tex_escape(a) for a in r["area_path"])
        tactics = r" \newline ".join(
            f"{_tt(t)} ({s})" for s, t in zip(r["switch_steps"], r["switch_tactics"]))
        fp = r["file_path"] or "--"
        lines.append(f"{_tt(r['theorem'])} & {_tt(fp) if r['file_path'] else fp} & {path} & {tactics} \\\\")
    if picked.empty:
        lines.append(r"\multicolumn{4}{c}{no proof met the selection rule} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- synthetic stand-in

_PREVIEW_TACTICS = (
    "intro x hx", "simp", "rw [Nat.add_comm]", "exact h", "apply le_trans", "norm_num", "constructor",
    "obtain ⟨k, hk⟩ := h", "linarith", "ring", "refine ⟨_, _⟩", "field_simp", "nlinarith [sq_nonneg x]",
    "exact_mod_cast hk", "positivity", "induction n with n ih", "cases h", "gcongr", "omega", "use 2",
)


def synthetic_case_inputs(n_proofs: int = 120, seed: int = 0,
                          judges: tuple[str, str] = ("judge-a", "judge-b"),
                          noise: float = 0.08) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A synthetic judged sample: steps from ``markov_lean.synthetic`` and two noisy simulated judges.

    For the layout preview and tests only.  Nothing here is a result.
    """
    from .synthetic import generate

    steps, _ = generate(n_proofs=n_proofs, min_steps=4, max_steps=16, p_stay=0.75, seed=seed)
    rng = np.random.default_rng(seed + 1)
    steps = steps.copy()
    steps["tactic"] = rng.choice(np.array(_PREVIEW_TACTICS, dtype=object), size=len(steps))
    codes = sorted(set(steps["true_area"]))
    rows = []
    for j, judge in enumerate(judges):
        r = np.random.default_rng(seed + 10 + j)
        wrong = r.random(len(steps)) < noise
        lab = np.where(wrong, r.choice(codes, size=len(steps)), steps["true_area"].to_numpy())
        conf = np.clip(r.beta(8, 2, size=len(steps)) - 0.25 * wrong, 0, 1)
        rows.append(pd.DataFrame({"proof_id": steps["proof_id"].to_numpy(), "step": steps["step"].to_numpy(),
                                  "judge": judge, "msc": lab, "confidence": conf}))
    return steps, pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- entry point


def write_outputs(cases: Cases, out: str | Path, figure_name: str = FIGURE_NAME, table_name: str = TABLE_NAME,
                  stamp: str | None = None, summary: bool = True) -> dict[str, Path]:
    """Write the table, a JSON summary of the picks and the figure into ``out``."""
    import json

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"table": out / f"{table_name}.tex"}
    paths["table"].write_text(latex_table(cases.picked))
    if summary:
        paths["summary"] = out / f"{figure_name}.json"
        paths["summary"].write_text(json.dumps(cases.summary(), indent=2, default=str))
    if len(cases.picked):
        from .figures import paper_rc, save_pdf

        fig = case_strips(cases.steps, max_steps=cases.config.max_steps)
        if stamp:
            import matplotlib

            with matplotlib.rc_context(paper_rc()):
                fig.text(0.995, 0.005, stamp, ha="right", va="bottom", fontsize=5, color="#999999")
        paths["figure"] = save_pdf(fig, out / f"{figure_name}.pdf")
    return paths


def build_parser(p: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    p = p or argparse.ArgumentParser(prog="markov-lean cases", description=__doc__.splitlines()[0])
    p.add_argument("--run", help="pilot output folder with judge/sample_steps.parquet and judge/labels.parquet")
    p.add_argument("--steps", help="steps table instead of <run>/judge/sample_steps.parquet")
    p.add_argument("--labels", help="labels table instead of <run>/judge/labels.parquet")
    p.add_argument("--out", default="paper/figures", help="folder for case_studies.pdf and case_studies_table.tex")
    p.add_argument("--synthetic", action="store_true",
                   help="use a synthetic stand-in run (layout preview, not a result)")
    p.add_argument("--n", dest="n_cases", type=int, default=CaseConfig.n_cases)
    p.add_argument("--min-confidence", type=float, default=CaseConfig.min_confidence)
    p.add_argument("--min-agreement", type=float, default=CaseConfig.min_agreement)
    p.add_argument("--min-steps", type=int, default=CaseConfig.min_steps)
    p.add_argument("--max-steps", type=int, default=CaseConfig.max_steps)
    p.add_argument("--max-tactic-chars", type=int, default=CaseConfig.max_tactic_chars)
    p.add_argument("--judges", nargs=2, metavar=("A", "B"), help="judge ids to compare (default: first two)")
    p.add_argument("--no-diverse", action="store_true", help="take the top proofs even if they repeat a file")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = CaseConfig(n_cases=args.n_cases, min_confidence=args.min_confidence, min_agreement=args.min_agreement,
                     min_steps=args.min_steps, max_steps=args.max_steps, max_tactic_chars=args.max_tactic_chars,
                     judges=list(args.judges) if args.judges else None, diverse=not args.no_diverse)
    if args.synthetic:
        steps, labels = synthetic_case_inputs()
    elif args.run or (args.steps and args.labels):
        steps, labels = load_run(args.run or ".", args.steps, args.labels)
    else:
        print("give --run <pilot output folder>, or --steps and --labels, or --synthetic", file=sys.stderr)
        return 2
    cases = select(steps, labels, cfg)
    s = cases.summary()
    print(f"cases: {s['n_eligible']} of {s['n_proofs']} proofs eligible; picked {len(s['picked'])}")
    if s["excluded"]:
        print("  excluded: " + ", ".join(f"{k} {v}" for k, v in s["excluded"].items()))
    for c in s["picked"]:
        print(f"  {c['theorem']}: {' -> '.join(c['area_path'])}, {c['n_switches']} switch(es), "
              f"agreement {c['agreement']:.2f}, {c['n_steps']} steps")
    paths = write_outputs(cases, args.out, stamp="synthetic preview" if args.synthetic else None)
    for path in paths.values():
        print(path)
    return 0 if len(cases.picked) else 1


if __name__ == "__main__":
    sys.exit(main())
