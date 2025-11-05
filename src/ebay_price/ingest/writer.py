from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from sqlalchemy import create_engine, text


def write_parquet(records: list[dict[str, Any]], parquet_path: str | Path) -> None:
    if not records:
        return
    df = pl.from_dicts(records)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)


def duckdb_upsert_listings(duckdb_path: str | Path, listings: list[dict[str, Any]]) -> None:
    if not listings:
        return
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(
            """
        CREATE TABLE IF NOT EXISTS listings (
            item_id VARCHAR PRIMARY KEY,
            title VARCHAR,
            category_path VARCHAR,
            brand VARCHAR,
            model VARCHAR,
            condition VARCHAR,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            listing_type VARCHAR,
            start_price DOUBLE,
            shipping_cost DOUBLE,
            return_policy VARCHAR,
            seller_username VARCHAR,
            seller_feedback_score INTEGER,
            seller_positive_percent DOUBLE,
            watchers INTEGER,
            bids INTEGER,
            final_price DOUBLE,
            sold BOOLEAN,
            currency VARCHAR
        );
        """
        )
        df = pl.from_dicts(listings)
        con.register("incoming_df", df.to_pandas())  # register as table
        con.execute(
            """
            INSERT OR REPLACE INTO listings
            SELECT * FROM incoming_df;
        """
        )
    finally:
        con.close()


def postgres_load_staging_and_merge(pg_url: str, raw_jsonl_path: str | Path) -> None:
    # Load raw into staging, then merge into listings with SQL.
    # Use JSON parsing in Postgres if needed.
    engine = create_engine(pg_url)
    raw_jsonl_path = Path(raw_jsonl_path)

    with engine.begin() as conn:
        # staging insert
        lines = [
            line for line in raw_jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        for chunk_start in range(0, len(lines), 1000):
            chunk = lines[chunk_start : chunk_start + 1000]
            values = ",".join([f"('{line}')" for line in chunk])
            conn.execute(text(f"INSERT INTO staging_listings (raw_json) VALUES {values};"))

        # You can write a server-side merge using JSONB operators
        # when your key mapping is final.
        # For now, keep Postgres optional, DuckDB will be the main local store.
