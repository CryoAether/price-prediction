from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from ebay_price.features.categorical import label_encode, target_encode
from ebay_price.features.datetime import datetime_features
from ebay_price.features.numeric import numeric_features
from ebay_price.features.text import text_features

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_listings(db_path: str = "data/artifacts/warehouse.duckdb") -> pl.DataFrame:
    con = duckdb.connect(db_path)
    try:
        df = con.execute("SELECT * FROM listings").pl()
        return df
    finally:
        con.close()


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    out = datetime_features(out)
    out = label_encode(out)
    out = target_encode(out)
    out = numeric_features(out)
    out = text_features(out)
    return out


def save_outputs(feat: pl.DataFrame) -> None:
    feat_path = PROCESSED_DIR / "features.parquet"
    feat.write_parquet(feat_path)

    # Optional supervised set if targets exist
    targets = [c for c in ("final_price", "sold") if c in feat.columns]
    if targets:
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
        keep: list[str] = []
        for c, t in zip(feat.columns, feat.dtypes, strict=False):
            if c in ("title", "start_time", "end_time"):
                continue
            if (c in targets) or (t in numeric_like):
                keep.append(c)
        train = feat.select(keep)
        train.write_parquet(PROCESSED_DIR / "train.parquet")


def main() -> None:
    df = load_listings()
    if df.is_empty():
        print("No listings found.")
        return
    feat = build_features(df)
    save_outputs(feat)
    print("Features written to data/processed/")


if __name__ == "__main__":
    main()
