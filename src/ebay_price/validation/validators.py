from __future__ import annotations

from pathlib import Path

import polars as pl
from pydantic import ValidationError

from ebay_price.validation.schemas import ListingRecord

REQUIRED_COLUMNS = [
    "item_id",
    "title",
    "category_path",
    "condition",
    "start_time",
    "end_time",
    "listing_type",
    "currency",
]
NUMERIC_NONNEG = [
    "start_price",
    "shipping_cost",
    "seller_feedback_score",
    "seller_positive_percent",
    "watchers",
    "bids",
    "final_price",
]


def check_required_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def check_unique_ids(df: pl.DataFrame) -> int:
    return int(df.get_column("item_id").n_unique() != df.height)


def check_non_negative(df: pl.DataFrame) -> list[str]:
    bad = []
    for c in NUMERIC_NONNEG:
        if c in df.columns and df.filter(pl.col(c) < 0).height > 0:
            bad.append(c)
    return bad


def check_time_order(df: pl.DataFrame) -> int:
    if "start_time" not in df.columns or "end_time" not in df.columns:
        return 0

    # Parse ISO 8601 with trailing 'Z' as naive datetimes (sufficient for ordering checks).
    # Your Polars version wants `format=` and does NOT support `utc=` here.
    iso_z = "%Y-%m-%dT%H:%M:%SZ"

    tmp = df.with_columns(
        [
            pl.col("start_time")
            .cast(pl.Utf8, strict=False)
            .str.strptime(pl.Datetime, format=iso_z, strict=False)
            .alias("start_dt"),
            pl.col("end_time")
            .cast(pl.Utf8, strict=False)
            .str.strptime(pl.Datetime, format=iso_z, strict=False)
            .alias("end_dt"),
        ]
    )

    return int(tmp.filter(pl.col("end_dt") < pl.col("start_dt")).height > 0)
    if "start_time" not in df.columns or "end_time" not in df.columns:
        return 0
    tmp = df.with_columns(
        [
            pl.col("start_time")
            .str.strptime(pl.Datetime, strict=False, utc=True)
            .alias("start_dt"),
            pl.col("end_time").str.strptime(pl.Datetime, strict=False, utc=True).alias("end_dt"),
        ]
    )
    return int(tmp.filter(pl.col("end_dt") < pl.col("start_dt")).height > 0)


def sample_pydantic_validation(df: pl.DataFrame, n: int = 50) -> list[tuple[int, str]]:
    errors = []
    for i in range(min(n, df.height)):
        try:
            ListingRecord(**df.row(i, named=True))
        except ValidationError as e:
            errors.append((i, str(e)))
    return errors


def validate_parquet(parquet_path: str | Path) -> None:
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pl.read_parquet(path)
    if miss := check_required_columns(df):
        raise AssertionError(f"Missing columns: {miss}")
    if check_unique_ids(df):
        raise AssertionError("item_id not unique")
    if bad := check_non_negative(df):
        raise AssertionError(f"Negative values: {bad}")
    if check_time_order(df):
        raise AssertionError("end_time earlier than start_time")
    pedantic = sample_pydantic_validation(df)
    if pedantic:
        preview = "\n".join([f"row {i}: {msg.splitlines()[0]}" for i, msg in pedantic[:5]])
        raise AssertionError(f"Pydantic validation failed:\n{preview}")
    print("Validation passed")
