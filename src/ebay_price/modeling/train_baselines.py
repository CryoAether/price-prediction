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


def train_regression(target: str = "final_price"):
    df = load_train()
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not in training data.")
    X, y = feature_target_split(df, target)
    y_np = y.to_pandas().values
    classes = np.unique(y_np)
    if classes.size < 2:
        print("[classification] Skipping training: only one class present in data.")
        return
    X_tr, X_va, y_tr, y_va = train_val_split(X, y, stratify=True)

    # Linear baseline
    lr = LinearRegression(n_jobs=None)
    lr.fit(X_tr, y_tr)
    yhat = lr.predict(X_va)
    m_lr = regression_metrics(y_va, yhat)

    # LightGBM baseline
    lgbm = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    yhat_l = lgbm.predict(X_va)
    m_lgbm = regression_metrics(y_va, yhat_l)

    # Save artifacts
    joblib.dump(lr, ARTIFACTS_DIR / "reg_linear.joblib")
    joblib.dump(lgbm, ARTIFACTS_DIR / "reg_lightgbm.joblib")
    (ARTIFACTS_DIR / "reg_metrics.json").write_text(
        json.dumps({"linear": m_lr, "lightgbm": m_lgbm}, indent=2)
    )
    print("Regression metrics:", {"linear": m_lr, "lightgbm": m_lgbm})


def train_classification(target: str = "sold"):
    df = load_train()
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' not in training data.")
    # Ensure binary 0/1
    if df.get_column(target).dtype != pl.Boolean:
        df = df.with_columns(pl.col(target).cast(pl.Boolean, strict=False))
    X, y = feature_target_split(df, target)
    X_tr, X_va, y_tr, y_va = train_val_split(X, y)

    # Logistic Regression (liblinear works well on small)
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
    )
    lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    prob_l = lgbm.predict_proba(X_va)[:, 1]
    m_lgbm = classification_metrics(y_va, prob_l)

    # Save artifacts
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
