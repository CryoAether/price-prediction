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

    # Cast columns
    def ts(col: str) -> pl.Series:
        return pl.col(col).str.strptime(pl.Datetime, strict=False, utc=True)

    cast = []
    for name, _dtype in (
        ("start_time", pl.Datetime),
        ("end_time", pl.Datetime),
    ):
        if name in df.columns:
            cast.append(ts(name).alias(name))
    if cast:
        df = df.with_columns(cast)

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
