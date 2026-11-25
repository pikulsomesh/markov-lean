# Release and reproduction shortcuts. Nothing here uploads anything or calls an API.
#   make release            write release/ from the pilot run (RUN=results/pilot3)
#   make release-check      check release/ is complete, consistent and free of keys and local paths
#   make reproduce          scripts/reproduce.sh with cached judge labels (see REPRODUCE.md)
#   make reproduce-plan     print every stage and command without running anything

PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
RUN ?= results/pilot3
RELEASE ?= release

.PHONY: release release-check reproduce reproduce-plan paper-assets

release:
	$(PY) -m markov_lean.cli export-dataset --run $(RUN) --out $(RELEASE) --force

release-check:
	$(PY) -m markov_lean.cli check-release $(RELEASE)

paper-assets:
	$(PY) scripts/paper_assets.py --pilot $(RUN)

reproduce:
	PY=$(PY) bash scripts/reproduce.sh --pilot $(RUN) --release $(RELEASE)

reproduce-plan:
	PY=$(PY) bash scripts/reproduce.sh --dry-run --pilot $(RUN) --release $(RELEASE)
