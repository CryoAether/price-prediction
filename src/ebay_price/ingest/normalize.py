from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl

from ebay_price.ingest.schema import ListingRaw


def validate_rows(rows: Iterable[dict[str, Any]]):
    """Validate raw dicts against ListingRaw and yield cleaned dicts."""
    for r in rows:
        yield ListingRaw(**r).model_dump()


def to_polars(rows: Iterable[dict]) -> pl.DataFrame:
    """Normalize incoming rows to a polars DataFrame with sane types."""
    df = pl.DataFrame(list(rows))
    if df.is_empty():
        return df

    # Parse timestamps robustly (accepts ISO-8601 with timezone like 'Z' or '+00:00')
    iso_fmt = "%+"
    for col in ("start_time", "end_time"):
        if col in df.columns:
            try:
                df = df.with_columns(
                    pl.col(col)
                    .cast(pl.Utf8, strict=False)
                    .str.to_datetime(
                        format=iso_fmt,
                        strict=False,
                        time_zone="UTC",
                    )
                    .alias(col)
                )
            except Exception:
                # Fallback: eager Series parsing (older polars behavior)
                parsed = (
                    df.get_column(col)
                    .cast(pl.Utf8, strict=False)
                    .str.to_datetime(
                        format=iso_fmt,
                        strict=False,
                        time_zone="UTC",
                    )
                )
                df = df.with_columns(parsed.alias(col))

    # Fill common nulls with sensible defaults
    fill_map = {
        "start_price": 0.0,
        "shipping_cost": 0.0,
        "seller_feedback_score": 0,
        "seller_positive_percent": 100.0,
        "watchers": 0,
        "bids": 0,
        "currency": "USD",
        "sold": 0,
    }
    for k, v in fill_map.items():
        if k in df.columns:
            df = df.with_columns(pl.col(k).fill_null(v))

    # Coerce 'sold' to {0,1} if present (handles None/True/False/'1'/'0'/yes/no)
    if "sold" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("sold").is_null())
            .then(0)
            .when(
                pl.col("sold")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .is_in(["true", "t", "yes", "y", "1"])
            )
            .then(1)
            .when(
                pl.col("sold")
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .is_in(["false", "f", "no", "n", "0"])
            )
            .then(0)
            .otherwise(pl.col("sold"))
            .alias("sold")
        ).with_columns(
            pl.col("sold").cast(pl.Int8, strict=False).fill_null(0).clip(0, 1).alias("sold")
        )

    return df
