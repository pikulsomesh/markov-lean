"""Smoke test for scripts/preview_figures.py (synthetic layout preview)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("seaborn")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preview_figures.py"


def load():
    spec = importlib.util.spec_from_file_location("preview_figures", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preview_renders_every_paper_figure(tmp_path):
    mod = load()
    mod.main(["--out", str(tmp_path)])
    names = {p.stem for p in tmp_path.glob("*.pdf")}
    paper = (SCRIPT.parents[1] / "paper" / "sections")
    wanted = set()
    for tex in paper.glob("*.tex"):
        for line in tex.read_text().splitlines():
            if "\\maybefigure{" in line:
                wanted.add(line.split("\\maybefigure{")[1].split("}")[0])
    assert wanted and wanted <= names


def test_every_preview_figure_is_stamped():
    mod = load()
    for _, fig in mod.make_all(np.random.default_rng(1)):
        assert any(t.get_text() == mod.STAMP for t in fig.texts)
