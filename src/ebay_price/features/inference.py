from __future__ import annotations

from typing import Any

import polars as pl

from ebay_price.features.categorical import label_encode
from ebay_price.features.datetime import datetime_features
from ebay_price.features.numeric import numeric_features
from ebay_price.features.text import text_features


def prepare_features_for_inference(payload: dict[str, Any]) -> pl.DataFrame:
    # Build a single-row DataFrame
    df = pl.DataFrame({k: [v] for k, v in payload.items()})
    # Apply the same feature functions (no target enc in prod unless available)
    out = datetime_features(df)
    out = label_encode(out)
    # target encoding during inference is typically done with precomputed mapping.
    # Here we skip target_encode to avoid leakage. If needed, load enc tables from training.
    out = numeric_features(out)
    out = text_features(out)

    # Keep numeric/boolean columns for model input
    numeric_like = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Boolean,
    }
    cols = [c for c, t in zip(out.columns, out.dtypes, strict=False) if t in numeric_like]
    return out.select(cols)
