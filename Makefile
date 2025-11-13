.PHONY: explain-no-shap  explain  etl-train-noclf  setup lint test format run-api run-app mlflow ingest-local duckdb-shell validate api ingest-jsonl ingest-csv etl-train

setup: ; poetry install
lint: ; poetry run ruff check . && poetry run black --check . && poetry run isort --check-only .
format: ; poetry run ruff check . --fix && poetry run black . && poetry run isort .
test: ; PYTHONPATH=src poetry run pytest

run-api: ; poetry run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
run-app: ; poetry run streamlit run app/streamlit_app.py
mlflow: ; poetry run mlflow ui --host 127.0.0.1 --port 5000

# src/ layout needs PYTHONPATH=src when invoking modules
ingest-local: ; PYTHONPATH=src poetry run python -m ebay_price.ingest.ingest_cli --mode local --input data/raw/sample.jsonl
validate: ; PYTHONPATH=src poetry run python ge/create_suite.py

# use Poetry's Python so duckdb is available
duckdb-shell: ; poetry run python -c 'import duckdb; con=duckdb.connect("data/artifacts/warehouse.duckdb"); print(con.execute("SELECT COUNT(*) FROM listings").fetchall())'
build-features: ; PYTHONPATH=src poetry run python -m ebay_price.features.build_features

train-regression: ; PYTHONPATH=src poetry run python -m ebay_price.modeling.train_baselines --task regression
train-classification: ; PYTHONPATH=src poetry run python -m ebay_price.modeling.train_baselines --task classification

coverage: ; PYTHONPATH=src poetry run pytest --cov=src --cov-report=term-missing

mlflow-log: ; PYTHONPATH=src poetry run python -m ebay_price.modeling.log_to_mlflow
mlflow-ui: ; poetry run mlflow ui --backend-store-uri $${MLFLOW_TRACKING_URI:-file:./data/artifacts/mlruns}
api:
	PYTHONPATH=src poetry run uvicorn api.app:app --reload --port 8000

streamlit: ; PYTHONPATH=src poetry run streamlit run app/streamlit_app.py
ingest-jsonl:
	PYTHONPATH=src poetry run python -m ebay_price.ingest.cli --ingest data/raw/synthetic.jsonl
ingest-csv:
	PYTHONPATH=src poetry run python -m ebay_price.ingest.cli --ingest data/raw/listings.csv
etl-train:
	PYTHONPATH=src poetry run python -m ebay_price.ingest.flow --path data/raw/synthetic.jsonl

etl-train-noclf:
	PYTHONPATH=src poetry run python -m ebay_price.ingest.flow --path data/raw/synthetic.jsonl --no-clf

explain:
	PYTHONPATH=src poetry run python -m ebay_price.explain.cli

explain-no-shap:
	PYTHONPATH=src poetry run python -m ebay_price.explain.cli --no-shap

validate-ci:
	PYTHONPATH=src poetry run python -m ebay_price.validation.ge_checks --fail-fast

prefect-orion:
	poetry run prefect server start

prefect-deploy:
	PYTHONPATH=src poetry run python -m ebay_price.flows.deploy

prefect-agent:
	poetry run prefect agent start -q default

ge-validate:
	PYTHONPATH=src poetry run python -m ebay_price.validation.ge_checks

ge-validate-ci:
	PYTHONPATH=src poetry run python -m ebay_price.validation.ge_checks --fail-fast
