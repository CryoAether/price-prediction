from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl

from ebay_price.ingest.schema import ListingRaw


def validate_rows(rows: Iterable[dict[str, Any]]):
    for r in rows:
        yield ListingRaw(**r).model_dump()


def to_polars(rows: Iterable[dict]) -> pl.DataFrame:
    df = pl.DataFrame(list(rows))
    if df.is_empty():
        return df

    # Parse timestamps robustly
    iso_fmt = "%+"
    for col in ("start_time", "end_time"):
        if col in df.columns:
            # Try expression-based parse first
            try:
                df = df.with_columns(
                    pl.col(col)
                    .cast(pl.Utf8, strict=False)
                    .str.to_datetime(format=iso_fmt, strict=False, time_zone="UTC")
                    .alias(col)
                )
            except Exception:
                # Fallback: eager Series parsing (as Polars suggests)
                parsed = (
                    df.get_column(col)
                    .cast(pl.Utf8, strict=False)
                    .str.to_datetime(format=iso_fmt, strict=False, time_zone="UTC")
                )
                df = df.with_columns(parsed.alias(col))

    # Fill NA sensibly
    fill_map = {
        "start_price": 0.0,
        "shipping_cost": 0.0,
        "seller_feedback_score": 0,
        "seller_positive_percent": 100.0,
        "watchers": 0,
        "bids": 0,
        "currency": "USD",
    }
    for k, v in fill_map.items():
        if k in df.columns:
            df = df.with_columns(pl.col(k).fill_null(v))
    return df
