from __future__ import annotations

from prefect import flow, task

from ebay_price.ingest.cli import ingest_file, refresh_features
from ebay_price.modeling.train_baselines import train_classification, train_regression


@task
def t_ingest(path: str) -> int:
    return ingest_file(path)


@task
def t_refresh_features() -> None:
    refresh_features()


@task
def t_train_regression() -> None:
    train_regression()


@task
def t_train_classification() -> None:
    train_classification()


@flow(name="eBay ETL + Train")
def etl_train(path: str | None = None, do_classification: bool = True) -> None:
    if path:
        t_ingest.submit(path)
    t_refresh_features.submit()
    t_train_regression.submit()
    if do_classification:
        t_train_classification.submit()


if __name__ == "__main__":
    # Example: poetry run python -m ebay_price.ingest.flow --path data/raw/listings.jsonl
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, default=None)
    p.add_argument("--no-clf", action="store_true")
    args = p.parse_args()
    etl_train(path=args.path, do_classification=not args.no_clf)
