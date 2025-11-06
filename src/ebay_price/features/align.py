from __future__ import annotations

import polars as pl


def align_to_columns(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    # add any missing columns as zeros (float)
    for c in cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(c))
    # drop extras and order
    return df.select(cols)
