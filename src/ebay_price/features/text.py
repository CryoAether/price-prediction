from __future__ import annotations

import re

import polars as pl


def _word_count(s: str) -> int:
    if not s:
        return 0
    return len(re.findall(r"[A-Za-z0-9]+", s))


def text_features(df: pl.DataFrame) -> pl.DataFrame:
    out = df.with_columns(
        [
            pl.col("title").cast(pl.Utf8, strict=False).str.len_chars().alias("title_len"),
            pl.col("title").cast(pl.Utf8, strict=False).map_elements(_word_count).alias("title_wc"),
            pl.col("title")
            .cast(pl.Utf8, strict=False)
            .str.contains(r"\d")
            .fill_null(False)
            .cast(pl.Int8)
            .alias("title_has_digit"),
        ]
    )

    if "brand" in out.columns:
        out = out.with_columns(
            pl.when(
                (pl.col("brand").is_not_null())
                & (
                    pl.col("title")
                    .cast(pl.Utf8, strict=False)
                    .str.contains(
                        pl.col("brand").cast(pl.Utf8, strict=False), literal=False, strict=False
                    )
                )
            )
            .then(1)
            .otherwise(0)
            .alias("title_has_brand")
        )
    return out
