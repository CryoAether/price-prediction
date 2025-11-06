from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    # guard for zero targets to avoid division by zero in MAPE
    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "mape": mape}


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out: dict[str, Any] = {"accuracy": float(accuracy_score(y_true, y_pred))}
    # handle edge cases for AUC/AP when only one class present
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["avg_precision"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["avg_precision"] = float("nan")
    return out
