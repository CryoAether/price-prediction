from __future__ import annotations

import json
from pathlib import Path

import joblib
import polars as pl

ART = Path("data/artifacts/models")


def load_reg_model_and_columns() -> tuple[object, list[str], str]:
    """Load best-available regression model and its training column order."""
    mpath = ART / (
        "reg_lightgbm.joblib" if (ART / "reg_lightgbm.joblib").exists() else "reg_linear.joblib"
    )
    model = joblib.load(mpath)
    cols_path = ART / "reg_feature_columns.json"
    cols = json.loads(cols_path.read_text()) if cols_path.exists() else []
    return model, cols, mpath.name


def load_processed_features() -> pl.DataFrame:
    """Load processed training features (the same the model trained on)."""
    from ebay_price.features.build_features import build_features, load_listings

    df = load_listings()
    return build_features(df)
