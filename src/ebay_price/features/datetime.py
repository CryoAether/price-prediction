from __future__ import annotations

import polars as pl


def datetime_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds:
      - start_dt, end_dt (parsed)
      - duration_hours
      - start_weekday (0=Mon)
      - start_hour (0-23)
      - start_month (1-12)
    """
    out = df.with_columns(
        [
            pl.col("start_time")
            .cast(pl.Utf8, strict=False)
            .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ", strict=False)
            .alias("start_dt"),
            pl.col("end_time")
            .cast(pl.Utf8, strict=False)
            .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ", strict=False)
            .alias("end_dt"),
        ]
    ).with_columns(
        [
            (
                pl.col("end_dt").cast(pl.Datetime) - pl.col("start_dt").cast(pl.Datetime)
            ).dt.total_seconds()
            / (3600.0).alias("duration_hours"),
            pl.col("start_dt").dt.weekday().alias("start_weekday"),
            pl.col("start_dt").dt.hour().alias("start_hour"),
            pl.col("start_dt").dt.month().alias("start_month"),
        ]
    )
    return out
