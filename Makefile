.PHONY: install test lint demo
install:
	pip install -e '.[xgb,dev]'
test:
	pytest
lint:
	ruff check . && ruff format --check .
demo:
	drive-failure synth --out data/synthetic.parquet
	drive-failure train --snapshots data/synthetic.parquet --model-out models/demo.joblib --report results_demo.md --synthetic
