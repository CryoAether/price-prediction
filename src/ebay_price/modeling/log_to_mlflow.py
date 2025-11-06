from __future__ import annotations

from pathlib import Path

from ebay_price.modeling.mlflow_utils import (
    configure_mlflow,
    log_classification_run,
    log_regression_run,
)

ART_DIR = Path("data/artifacts/models")


def main() -> None:
    configure_mlflow()
    ART_DIR.mkdir(parents=True, exist_ok=True)
    log_regression_run(ART_DIR)
    log_classification_run(ART_DIR)
    print("Logged artifacts and metrics to MLflow.")


if __name__ == "__main__":
    main()
