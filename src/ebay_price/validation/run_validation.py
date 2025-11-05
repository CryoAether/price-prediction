from __future__ import annotations

import duckdb

from ebay_price.validation.validators import validate_parquet


def validate_latest_snapshot() -> None:
    validate_parquet("data/raw/listings_snapshot.parquet")


def validate_duckdb() -> None:
    con = duckdb.connect("data/artifacts/warehouse.duckdb")
    try:
        total = con.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        if total <= 0:
            raise AssertionError("No rows in listings table.")
        nulls = con.execute("SELECT COUNT(*) FROM listings WHERE item_id IS NULL").fetchone()[0]
        if nulls > 0:
            raise AssertionError("NULL item_id detected.")
        print(f"DuckDB validation passed. Row count: {total}")
    finally:
        con.close()


if __name__ == "__main__":
    validate_latest_snapshot()
    validate_duckdb()
