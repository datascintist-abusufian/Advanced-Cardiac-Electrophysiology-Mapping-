.PHONY: install prepare test smoke aggregate figures audit
install:
	python -m pip install -e ".[dev]"
prepare:
	python scripts/prepare_acdc.py --config configs/default.yaml
test:
	pytest
smoke:
	python scripts/create_synthetic_acdc.py
	python scripts/prepare_acdc.py --config configs/smoke.yaml
	python scripts/run_experiment.py --config configs/smoke.yaml --seed 42 --fold 0
aggregate:
	python scripts/aggregate_results.py --run-dir runs --output-dir runs/summary
figures:
	python scripts/make_figures.py --config configs/default.yaml
audit:
	python scripts/audit_manuscript_claims.py

