# Paper

ICML 2027 submission draft. Abstracts are due Jan 14, 2027 and full papers Jan 22, 2027.

## Build

```bash
cd paper
make          # main.pdf, with latexmk if installed, else pdflatex + bibtex
make style    # says whether the ICML style or the article fallback will be used
```

`main.pdf` is the latest build, committed so the draft can be read on GitHub. Rebuild and commit it when the text changes. It needs a TeX distribution with latexmk: BasicTeX on the Mac (about 300 MB, versus about 9 GB for full MacTeX), or `texlive-latex-extra`, `texlive-fonts-recommended` and `latexmk` on Debian/Ubuntu.

On the Mac:

```
brew install --cask basictex && eval "$(/usr/libexec/path_helper)"
sudo tlmgr update --self
sudo tlmgr install latexmk booktabs microtype caption eso-pic forloop printlen newtx float courier helvetic
make
```

`courier` and `helvetic` are needed: without `courier`, `\texttt` fails with a missing `pcrr7t` font.

`main.tex` follows the official ICML 2027 `example_paper.tex`. It falls back to the plain article class (natbib, `plainnat`) only if `icml2027.sty` is missing.

**Style files:** the ICML 2027 author kit is in this folder: `icml2027.sty`, `icml2027.bst`, and the `algorithm.sty`, `algorithmic.sty` and `fancyhdr.sty` it ships with. Leave them as they are; ICML checks submissions against the official style. The kit's `example_paper.tex` and `example_paper.bib` aren't copied in, since `main.tex` already follows them.

**ICML 2027 rules that shape this draft** (from the 2027 author instructions):

- The main body is at most 8 pages (9 in the camera-ready). References, the appendix and the required statements don't count.
- Submissions are double-blind. `\usepackage{icml2027}` gives the anonymous version, and `main.tex` uses it by default. Use `[preprint]` for an arXiv version and `[accepted]` for the camera-ready. Don't link a non-anonymized repo, and leave out acknowledgements until the camera-ready.
- An **AI Use Statement** and an **Impact Statement** are required. They are unnumbered sections before the references, in `sections/statements.tex`.
- The abstract is one paragraph, 4 to 6 sentences.
- Protect capitals in `refs.bib` with braces, e.g. `{Markov}`.
- Figures are drawn at the width they print, 3.25 in for one column and 6.75 in across the page, with 8 pt sans-serif text (7 pt tick labels), and the `.tex` includes them at exactly `\columnwidth` or `\textwidth`. Two panels side by side go in a `figure*`, each `\columnwidth` wide. Including a figure at any other width rescales its text.
- Captions go under figures, and figures carry no titles of their own. The figure functions draw none by default; the confusion matrix's NMI and the flow plot's coverage are attached to the Figure for the caption.
- The kit asks for Type-1 fonts and warns against Type 3. matplotlib can't write Type 1 without LaTeX, so `figures.save_pdf` embeds TrueType and never Type 3. Check the built paper with `pdffonts main.pdf` before submitting.
- An OpenReview profile is needed by Dec 31, 2026. The author list freezes at the Jan 14, 2027 abstract deadline.

## Layout

| path | what it holds |
|---|---|
| `main.tex` | preamble, title block, section order |
| `macros.tex` | notation, `\todo{}`, `\tbd` (a red dash for a number that waits on a run) and `\maybefigure{}` (uses `figures/<name>.pdf`, else the synthetic preview, else a placeholder box) |
| `sections/*.tex` | one file per section: abstract, intro, related, method, experiments, search, discussion, appendix, statements |
| `refs.bib` | references, each checked against a primary source (see the comment above each entry) |
| `figures/` | rendered figure PDFs (written by the analysis, not by hand) |
| `figures/preview/` | the same figures drawn from made-up numbers by `scripts/preview_figures.py`, for judging layout only |

## Figures

Each figure is a function in `src/markov_lean/figures.py` that takes arrays or DataFrames from the tables in `docs/schema.md` and returns a matplotlib Figure. Save it with `figures.save_pdf(fig, "paper/figures/<name>.pdf")` under the name the section expects:

| file in `figures/` | function | inputs |
|---|---|---|
| `implied_timescales.pdf` | `implied_timescales(lags, timescales, lower, upper)` | timescales per lag from the MSM |
| `chapman_kolmogorov.pdf` | `chapman_kolmogorov(lags, predicted, estimated, estimated_err)` | `T(tau)^k` and `T(k tau)` per lag, `(n_lags, m, m)` |
| `transition_matrix.pdf` | `transition_heatmap(T, labels)` | macrostate transition matrix |
| `judge_confusion.pdf` | `judge_cluster_confusion(macrostate, msc)` | `assignments.macrostate` and `labels.msc` merged on `proof_id, step`; NMI is `fig.nmi`, for the caption |
| `area_flow.pdf` | `area_flow(area_transitions(assignments))` | consecutive steps' areas within each proof |
| `search_ablation.pdf` | `search_ablation(results, group_col="benchmark")` | long-form: one row per method, benchmark and seed |
| `switches.pdf`, `table_switches.tex` | `markov-lean switches` (`markov_lean.switches.draw`), copied in by `markov-lean paper` | a switches run's `switches.json`, `switches.pdf` and `table_switches.tex`; see `docs/switches.md` |
| `case_studies.pdf`, `case_studies_table.tex` | `markov-lean cases --run results/pilot3 --out paper/figures` (`markov_lean.cases`) | a pilot run's `judge/sample_steps.parquet` and `judge/labels.parquet`; the table is a `tabular` the appendix reads with `\maybetable{}` |

Figures use seaborn's `ticks` style at `paper` scale with a sans-serif font: Arial, falling back to Helvetica, Liberation Sans or DejaVu Sans where Arial isn't installed. Default sizes fit the ICML column (3.25 in); `figures.TEXT_WIDTH` (6.75 in) is the full width. Every figure is drawn inside `figures.paper_rc()`, so global matplotlib settings are left alone; call `figures.use_paper_style()` to get the same look in a notebook.

For `search_ablation`, pass one row per run when there are several seeds, and seaborn draws the mean with a 95% CI. Pass `err_col` only for errors you computed yourself, with one row per bar.

Install the plotting extra (matplotlib and seaborn) with `pip install -e ".[figures]"`.

### Synthetic preview

Until the runs are in, `python scripts/preview_figures.py` draws every figure from made-up numbers into `figures/preview/`, so the PDF shows its final layout. These are not results: each one carries a small "synthetic preview" stamp, and the paper prints "synthetic preview, not a result" in red under it. A real figure saved to `figures/<name>.pdf` replaces the preview automatically. Delete `figures/preview/` before submitting, then rebuild and check the PDF has no red notes left.

Tables hold red dashes (`\tbd`) wherever a number waits on a run.

## Citations

Only cite works whose entry is in `refs.bib`. Every entry there was checked against a page that was actually read (arXiv, the proceedings site or the publisher); the source is in a comment above it. Anything not yet checked is listed under "TODO" at the bottom of `refs.bib`, not invented.
