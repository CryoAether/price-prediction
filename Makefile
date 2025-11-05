.PHONY: setup lint test format run-api run-app mlflow
setup: ; poetry install
lint: ; poetry run ruff check . && poetry run black --check . && poetry run isort --check-only .
format: ; poetry run ruff check . --fix && poetry run black . && poetry run isort .
test: ; poetry run pytest
run-api: ; poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
run-app: ; poetry run streamlit run app/streamlit_app.py
mlflow: ; poetry run mlflow ui --host 127.0.0.1 --port 5000
