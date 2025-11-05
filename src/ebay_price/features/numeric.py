from __future__ import annotations

import polars as pl

NUMERIC_COLS = ["start_price", "final_price", "shipping_cost", "watchers", "bids"]


def _clip_nonneg(series: pl.Series) -> pl.Series:
    return series.fill_null(0).clip(lower_bound=0)


def numeric_features(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    for col in NUMERIC_COLS:
        if col in out.columns:
            out = out.with_columns(_clip_nonneg(out[col]).alias(col))

    # winsorize key monetary columns at 1% / 99% to reduce outlier impact
    for col in ("start_price", "final_price", "shipping_cost"):
        if col in out.columns:
            q = out.select(
                [
                    pl.quantile(col, 0.01, interpolation="nearest").alias("q1"),
                    pl.quantile(col, 0.99, interpolation="nearest").alias("q99"),
                ]
            )
            q1, q99 = q.item(0, 0), q.item(0, 1)
            out = out.with_columns(pl.col(col).clip(q1, q99).alias(f"{col}_win"))
            # log1p variant
            out = out.with_columns(pl.col(f"{col}_win").log1p().alias(f"log1p_{col}"))
    return out
