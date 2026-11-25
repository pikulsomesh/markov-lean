# Paper

ICML 2027 submission draft. Abstracts are due Jan 14, 2027 and full papers Jan 22, 2027.

## Build

```bash
cd paper
make          # main.pdf, with latexmk if installed, else pdflatex + bibtex
make style    # says whether the ICML style or the article fallback will be used
```

`main.tex` uses the official ICML style when `icml2027.sty` and `icml2027.bst` are in this folder, and falls back to the plain article class (natbib, `plainnat`) when they aren't. Nothing else needs to change when the style files arrive.

**Style files:** the ICML 2027 author kit is posted on the ICML 2027 author instructions page as `icml2027.zip`, but this cloud environment can't reach media.icml.cc, so the kit is not in the repo yet. On the Mac Mini, download the zip from that page, put `icml2027.sty`, `icml2027.bst` and any other files it ships into `paper/`, and run `make style` to check. The ICML template loads `hyperref` before the style; `main.tex` already does.

## Layout

| path | what it holds |
|---|---|
| `main.tex` | preamble, title block, section order |
| `macros.tex` | notation, `\todo{}`, and `\maybefigure{}` (draws a placeholder box until the PDF exists) |
| `sections/*.tex` | one file per section: abstract, intro, related, method, experiments, search, discussion, appendix |
| `refs.bib` | references, each checked against a primary source (see the comment above each entry) |
| `figures/` | rendered figure PDFs (written by the analysis, not by hand) |

## Figures

Each figure is a function in `src/markov_lean/figures.py` that takes arrays or DataFrames from the tables in `docs/schema.md` and returns a matplotlib Figure. Save it with `figures.save_pdf(fig, "paper/figures/<name>.pdf")` under the name the section expects:

| file in `figures/` | function | inputs |
|---|---|---|
| `implied_timescales.pdf` | `implied_timescales(lags, timescales, lower, upper)` | timescales per lag from the MSM |
| `chapman_kolmogorov.pdf` | `chapman_kolmogorov(lags, predicted, estimated, estimated_err)` | `T(tau)^k` and `T(k tau)` per lag, `(n_lags, m, m)` |
| `transition_matrix.pdf` | `transition_heatmap(T, labels)` | macrostate transition matrix |
| `judge_confusion.pdf` | `judge_cluster_confusion(macrostate, msc)` | `assignments.macrostate` and `labels.msc` merged on `proof_id, step`; NMI goes in the title |
| `area_flow.pdf` | `area_flow(area_transitions(assignments))` | consecutive steps' areas within each proof |
| `search_ablation.pdf` | `search_ablation(results, group_col="benchmark", err_col=...)` | one row per method and benchmark |

Install the plotting extra with `pip install -e ".[figures]"`.

## Citations

Only cite works whose entry is in `refs.bib`. Every entry there was checked against a page that was actually read (arXiv, the proceedings site or the publisher); the source is in a comment above it. Anything not yet checked is listed under "TODO" at the bottom of `refs.bib`, not invented.
