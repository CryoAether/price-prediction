from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import polars as pl

from ebay_price.features.build_features import build_features
from ebay_price.ingest.normalize import to_polars

REQUIRED_DEFAULTS: dict[str, Any] = {
    # datetime.py expects these columns:
    "start_time": dt.datetime.utcnow().replace(tzinfo=dt.UTC).isoformat(),
    "end_time": (dt.datetime.utcnow().replace(tzinfo=dt.UTC) + dt.timedelta(days=1)).isoformat(),
    # safe fallbacks for other steps (already filled in normalize, but ensure here too)
    "listing_type": "Auction",
    "currency": "USD",
    "watchers": 0,
    "bids": 0,
    "shipping_cost": 0.0,
    "seller_feedback_score": 0,
    "seller_positive_percent": 100.0,
}


def _ensure_required_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure all columns needed by the training feature pipeline exist."""
    for k, v in REQUIRED_DEFAULTS.items():
        if k not in df.columns:
            df[k] = v
    # Cast obvious types to avoid downstream surprises
    numeric_cols = [
        "start_price",
        "shipping_cost",
        "watchers",
        "bids",
        "seller_feedback_score",
        "seller_positive_percent",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_inference_features(df: pd.DataFrame) -> pl.DataFrame:
    """
    Convert a pandas payload (one or many rows) into the model's feature frame.
    Uses the exact same build_features pipeline as training.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("build_inference_features expects a pandas.DataFrame")

    df = _ensure_required_fields(df)

    rows = df.to_dict(orient="records")
    pl_df = to_polars(rows)  # apply normalize types/timestamps
    feats = build_features(pl_df)  # run full training feature pipeline

    # Drop target if present
    if "final_price" in feats.columns:
        feats = feats.drop("final_price")
    return feats


# Backward-compat name used by older streamlit_app versions
def prepare_features_for_inference(df: pd.DataFrame) -> pl.DataFrame:
    return build_inference_features(df)
