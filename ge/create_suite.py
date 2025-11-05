from pathlib import Path

import polars as pl


# Simple inline checks to keep GE overhead light for now
def validate_listings(parquet_path: str) -> None:
    df = pl.read_parquet(parquet_path)
    required = [
        "item_id",
        "title",
        "category_path",
        "condition",
        "start_time",
        "end_time",
        "listing_type",
        "currency",
    ]
    for col in required:
        if col not in df.columns:
            raise AssertionError(f"Missing required column: {col}")

    assert df["item_id"].n_unique() == df.shape[0], "item_id should be unique"
    print("Validation passed")


if __name__ == "__main__":
    p = Path("data/raw/listings_snapshot.parquet")
    if not p.exists():
        raise SystemExit("Parquet not found. Run ingestion first.")
    validate_listings(str(p))
