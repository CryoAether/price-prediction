from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from ebay_price.modeling.datasets import feature_target_split, load_train, train_val_split
from ebay_price.modeling.metrics import classification_metrics, regression_metrics

ARTIFACTS_DIR = Path("data/artifacts/models")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def train_regression(target: str = "final_price") -> None:
    df = load_train()
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not in training data.")
    X, y = feature_target_split(df, target)

    # Persist training feature columns for inference alignment
    (ARTIFACTS_DIR / "reg_feature_columns.json").write_text(json.dumps(X.columns, indent=2))

    X_tr, X_va, y_tr, y_va = train_val_split(X, y)

    # If the training set is tiny, tree models cannot fit; measure size safely
    n_train = getattr(y_tr, "shape", None)[0] if hasattr(y_tr, "shape") else len(y_tr)
    tiny_train = n_train < 2

    # Linear baseline (always train)
    lr = LinearRegression(n_jobs=None)
    lr.fit(X_tr, y_tr)
    yhat = lr.predict(X_va)
    m_lr = regression_metrics(y_va, yhat)

    metrics_out: dict[str, object] = {"linear": m_lr}

    # LightGBM baseline (only when training set has >= 2 rows)
    if not tiny_train:
        lgbm = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )
        lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
        yhat_l = lgbm.predict(X_va)
        m_lgbm = regression_metrics(y_va, yhat_l)
        metrics_out["lightgbm"] = m_lgbm
        joblib.dump(lgbm, ARTIFACTS_DIR / "reg_lightgbm.joblib")
    else:
        metrics_out["lightgbm"] = "skipped: training set < 2 rows"

    # Save artifacts
    joblib.dump(lr, ARTIFACTS_DIR / "reg_linear.joblib")
    (ARTIFACTS_DIR / "reg_metrics.json").write_text(json.dumps(metrics_out, indent=2))
    print("Regression metrics:", metrics_out)


def train_classification(target: str = "sold") -> None:
    df = load_train()
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not in training data.")

    # Ensure boolean 0/1
    if df.get_column(target).dtype != pl.Boolean:
        df = df.with_columns(pl.col(target).cast(pl.Boolean, strict=False))

    X, y = feature_target_split(df, target)

    # Persist training feature columns for inference alignment
    (ARTIFACTS_DIR / "clf_feature_columns.json").write_text(json.dumps(X.columns, indent=2))

    # Guard for single-class datasets
    y_np_all = y.to_pandas().values
    classes = np.unique(y_np_all)
    if classes.size < 2:
        # Write a metrics file so downstream steps don't special-case missing files
        (ARTIFACTS_DIR / "clf_metrics.json").write_text(
            json.dumps(
                {"logit": "skipped: one class", "lightgbm": "skipped: one class"},
                indent=2,
            )
        )
        print("[classification] Skipping training: only one class present in full dataset.")
        return

    # Stratified split for classification
    X_tr, X_va, y_tr, y_va = train_val_split(X, y, stratify=True)

    # Logistic Regression
    logit = LogisticRegression(max_iter=1000)
    logit.fit(X_tr, y_tr)
    prob = logit.predict_proba(X_va)[:, 1]
    m_logit = classification_metrics(y_va, prob)

    # LightGBM classifier
    lgbm = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    prob_l = lgbm.predict_proba(X_va)[:, 1]
    m_lgbm = classification_metrics(y_va, prob_l)

    # Save artifacts and metrics
    joblib.dump(logit, ARTIFACTS_DIR / "clf_logit.joblib")
    joblib.dump(lgbm, ARTIFACTS_DIR / "clf_lightgbm.joblib")
    (ARTIFACTS_DIR / "clf_metrics.json").write_text(
        json.dumps({"logit": m_logit, "lightgbm": m_lgbm}, indent=2)
    )
    print("Classification metrics:", {"logit": m_logit, "lightgbm": m_lgbm})


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["regression", "classification"], required=True)
    args = p.parse_args()

    if args.task == "regression":
        train_regression()
    else:
        train_classification()
