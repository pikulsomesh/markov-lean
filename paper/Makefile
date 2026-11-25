# Build the paper. Uses latexmk when present, else pdflatex + bibtex.
# `make` builds main.pdf; `make clean` removes build output.

MAIN := main
LATEXMK := $(shell command -v latexmk 2>/dev/null)

.PHONY: all pdf clean style

all: pdf

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex macros.tex refs.bib $(wildcard sections/*.tex) $(wildcard figures/*.pdf)
ifdef LATEXMK
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex
else
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
endif

# Reports which build the paper will use.
style:
	@if [ -f icml2027.sty ]; then echo "ICML 2027 style"; else echo "article fallback (icml2027.sty not found)"; fi

clean:
	-latexmk -C $(MAIN).tex 2>/dev/null
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).pdf sections/*.aux
