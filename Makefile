# Full pipeline. DATA can be overridden: make all DATA=/path/to/data
DATA ?= data
PY ?= python

.PHONY: all data weather features leakage tune evaluate report market ablation figures test clean

all: data features leakage evaluate report market ablation figures

data:            ## download raw data (~1-2 h, rate-limited on purpose)
	$(PY) -m footpred.data.understat $(DATA)
	$(PY) -m footpred.data.wikipedia $(DATA)
	$(PY) -m footpred.data.parse_wikipedia $(DATA)
	$(PY) -m footpred.data.wikidata $(DATA)
	$(PY) -m footpred.data.ages $(DATA)

weather:         ## optional: hourly weather at each stadium (slow, Open-Meteo rate limits)
	$(PY) -m footpred.data.weather $(DATA)

features:        ## build the 220 leakage-free features (~3 min)
	$(PY) -m footpred.features.build $(DATA)

leakage:         ## automated leakage audit (~8 min)
	$(PY) -m footpred.evaluation.leak_tests $(DATA)

tune:            ## random search of LightGBM hyper-parameters on the tuning period (~40 min)
	$(PY) -m footpred.evaluation.walk_forward $(DATA) tune 60

evaluate:        ## walk-forward evaluation, Jul 2024 -> Oct 2026 (~25 min)
	$(PY) -m footpred.evaluation.walk_forward $(DATA) eval

report:          ## fair comparison with the market, betting simulation, CLV
	$(PY) -m footpred.evaluation.report $(DATA)

market:          ## market-efficiency study (favourite-longshot, outlier bookmaker, ...)
	$(PY) -m footpred.evaluation.market_study $(DATA)

ablation:        ## feature-family ablation on the tuning period (~12 min)
	$(PY) -m footpred.evaluation.ablation $(DATA)

figures:
	$(PY) scripts/make_figures.py $(DATA)

test:
	$(PY) -m pytest -q

clean:
	rm -rf results/*.csv results/*.txt docs/figures/*.png
