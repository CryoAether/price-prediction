from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from great_expectations.dataset import PandasDataset

WAREHOUSE = Path("data/artifacts/warehouse.duckdb")


def _read_table(name: str) -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE))
    try:
        return con.execute(f"SELECT * FROM {name}").df()
    finally:
        con.close()


def validate_raw_listings() -> dict:
    df = _read_table("raw_listings")
    ds = PandasDataset(df)

    results = {}
    results["row_count_gt_0"] = ds.expect_table_row_count_to_be_between(min_value=1)
    # required columns
    required = [
        "item_id",
        "title",
        "brand",
        "model",
        "category_path",
        "start_price",
        "shipping_cost",
        "watchers",
        "bids",
        "seller_feedback_score",
        "seller_positive_percent",
    ]
    for col in required:
        results[f"{col}_exists"] = ds.expect_column_to_exist(col)

    # types / ranges (coerce to numeric for robustness)
    numeric = [
        "start_price",
        "shipping_cost",
        "watchers",
        "bids",
        "seller_feedback_score",
        "seller_positive_percent",
    ]
    for col in numeric:
        if col in df.columns:
            results[f"{col}_not_null"] = ds.expect_column_values_to_not_be_null(col)
            results[f"{col}_non_negative"] = ds.expect_column_min_to_be_between(col, min_value=0)

    if "seller_positive_percent" in df.columns:
        results["seller_positive_percent_lte_100"] = ds.expect_column_max_to_be_between(
            "seller_positive_percent", max_value=100
        )

    # duplicates
    if "item_id" in df.columns:
        results["item_id_unique"] = ds.expect_column_values_to_be_unique("item_id")

    return {k: bool(v["success"]) for k, v in results.items()}


def validate_listings_features() -> dict:
    # post-feature table used for training
    df = _read_table("listings")
    ds = PandasDataset(df)
    results = {}
    results["row_count_gt_0"] = ds.expect_table_row_count_to_be_between(min_value=1)

    # target & key feature existence
    must_have = ["final_price", "start_price", "shipping_cost", "watchers", "bids"]
    for col in must_have:
        results[f"{col}_exists"] = ds.expect_column_to_exist(col)
        if col in df.columns:
            results[f"{col}_not_null"] = ds.expect_column_values_to_not_be_null(col)

    # target sanity
    if "final_price" in df.columns:
        results["final_price_non_negative"] = ds.expect_column_min_to_be_between(
            "final_price", min_value=0
        )

    return {k: bool(v["success"]) for k, v in results.items()}


def main(fail_fast: bool = False) -> int:
    raw = validate_raw_listings()
    feats = validate_listings_features()

    all_ok = all(raw.values()) and all(feats.values())
    if fail_fast and not all_ok:
        # Print failing checks to aid CI logs
        print("Failed checks (raw_listings):", [k for k, v in raw.items() if not v])
        print("Failed checks (listings):", [k for k, v in feats.items() if not v])
        return 1
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args()
    raise SystemExit(main(fail_fast=args.fail_fast))
