from __future__ import annotations

import json
from pathlib import Path

import mlflow

from ebay_price.config import settings


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)


def log_regression_run(art_dir: Path) -> None:
    metrics_path = art_dir / "reg_metrics.json"
    if not metrics_path.exists():
        print(f"[mlflow] regression metrics file not found: {metrics_path}")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    with mlflow.start_run(run_name="regression-baselines"):
        # Log metrics
        if isinstance(metrics.get("linear"), dict):
            mlflow.log_metrics(
                {
                    f"linear_{k}": float(v)
                    for k, v in metrics["linear"].items()
                    if isinstance(v, int | float)
                }
            )
        else:
            mlflow.set_tag("linear_status", str(metrics.get("linear")))

        if isinstance(metrics.get("lightgbm"), dict):
            mlflow.log_metrics(
                {
                    f"lgbm_{k}": float(v)
                    for k, v in metrics["lightgbm"].items()
                    if isinstance(v, int | float)
                }
            )
        else:
            mlflow.set_tag("lgbm_status", str(metrics.get("lightgbm")))

        # Log artifacts (models + metrics)
        if (art_dir / "reg_linear.joblib").exists():
            mlflow.log_artifact(str(art_dir / "reg_linear.joblib"), artifact_path="models")
        if (art_dir / "reg_lightgbm.joblib").exists():
            mlflow.log_artifact(str(art_dir / "reg_lightgbm.joblib"), artifact_path="models")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")


def log_classification_run(art_dir: Path) -> None:
    metrics_path = art_dir / "clf_metrics.json"
    if not metrics_path.exists():
        print(f"[mlflow] classification metrics file not found: {metrics_path}")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    with mlflow.start_run(run_name="classification-baselines"):
        # Metrics may be dicts or strings (skip messages)
        for model_name in ("logit", "lightgbm"):
            val = metrics.get(model_name)
            if isinstance(val, dict):
                mlflow.log_metrics(
                    {
                        f"{model_name}_{k}": float(v)
                        for k, v in val.items()
                        if isinstance(v, int | float)
                    }
                )
            else:
                mlflow.set_tag(f"{model_name}_status", str(val))

        # Log artifacts (if present)
        for fname in ("clf_logit.joblib", "clf_lightgbm.joblib"):
            f = art_dir / fname
            if f.exists():
                mlflow.log_artifact(str(f), artifact_path="models")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
