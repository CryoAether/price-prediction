from __future__ import annotations

import polars as pl


def _label_encode(series: pl.Series) -> pl.Series:
    # Map unique categories to 0..K-1; nulls -> -1
    cats = series.drop_nulls().unique()
    mapping = {v: i for i, v in enumerate(cats.to_list())}
    return series.map_elements(lambda v: mapping.get(v, -1)).alias(series.name + "_le")


def label_encode(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    for col in ("brand", "model", "condition", "listing_type", "category_path"):
        if col in out.columns:
            out = out.with_columns(_label_encode(out[col]))
    return out


def _target_encode_mean(
    df: pl.DataFrame, cat_col: str, target_col: str, m: float = 5.0
) -> pl.DataFrame:
    """
    Mean target encoding with simple m-smoothing:
      enc = (count * mean + m * global_mean) / (count + m)
    """
    if cat_col not in df.columns or target_col not in df.columns:
        return df

    gmean = df.select(pl.mean(target_col).alias("gmean")).item(0, 0)

    agg = (
        df.group_by(cat_col)
        .agg([pl.count().alias("cnt"), pl.mean(target_col).alias("mean_t")])
        .with_columns(
            ((pl.col("cnt") * pl.col("mean_t") + m * gmean) / (pl.col("cnt") + m)).alias(
                f"{cat_col}__te_{target_col}"
            )
        )[[cat_col, f"{cat_col}__te_{target_col}"]]
    )

    out = df.join(agg, on=cat_col, how="left")
    return out


def target_encode(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    for cat in ("brand", "category_path"):
        if "final_price" in out.columns:
            out = _target_encode_mean(out, cat, "final_price", m=10.0)
        if "sold" in out.columns:
            # sold as 0/1 mean by category/brand approximates sell-through
            out = _target_encode_mean(out, cat, "sold", m=10.0)
    return out
