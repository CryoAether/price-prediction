from __future__ import annotations

from prefect import flow, task

from ebay_price.eval.eda_report import main as run_eda
from ebay_price.validation.run_validation import validate_duckdb, validate_latest_snapshot


@task
def t_validate_snapshot():
    validate_latest_snapshot()


@task
def t_validate_duckdb():
    validate_duckdb()


@task
def t_run_eda():
    run_eda()


@flow(name="validation_and_eda")
def main():
    t_validate_snapshot()
    t_validate_duckdb()
    t_run_eda()


if __name__ == "__main__":
    main()
