# Team entry points. Run from the repo root inside the Python 3.12 environment (see requirements.txt).
# Notebooks are generated: edit the builders in tools/, never the .ipynb. Run the long ones one at a time.
CROPS ?= strawberry blueberry raspberry tomato bell_pepper cucumber hydroponics
EXEC = jupyter nbconvert --to notebook --execute --inplace

.PHONY: help setup growth plague normalize normalize-safe clean-outputs

help:
	@echo "make setup           enable the git filter that strips notebook outputs (once per clone)"
	@echo "make growth          rebuild + execute growth.ipynb       (~1.5 min with a warm cache)"
	@echo "make plague          rebuild + execute plague.ipynb       (~15 min)"
	@echo "make normalize       master notebook, crops in sequence   (~45 min; CROPS='tomato cucumber' to pick, add 'hydroponics')"
	@echo "make normalize-safe  one notebook per crop, one at a time (lowest RAM; outputs normalization/normalize_<crop>.ipynb)"
	@echo "make clean-outputs   strip outputs from every notebook in place"

setup:
	git config filter.stripnb.clean "python tools/strip_outputs.py"
	git config filter.stripnb.smudge cat

growth:
	python tools/build_growth_nb.py && $(EXEC) --ExecutePreprocessor.timeout=1800 growth.ipynb

plague:
	python tools/build_plague_nb.py && $(EXEC) --ExecutePreprocessor.timeout=3600 plague.ipynb

normalize:
	python tools/build_normalized_nbs.py --crops $(CROPS) && $(EXEC) --ExecutePreprocessor.timeout=14400 normalization/normalize_all.ipynb

normalize-safe:
	for c in $(CROPS); do \
		python tools/build_normalized_nbs.py --crops $$c --out normalization/normalize_$$c.ipynb && \
		$(EXEC) --ExecutePreprocessor.timeout=7200 normalization/normalize_$$c.ipynb || exit 1; \
	done

clean-outputs:
	python tools/strip_outputs.py growth.ipynb plague.ipynb $(wildcard normalization/*.ipynb)
