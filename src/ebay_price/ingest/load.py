from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

WAREHOUSE = Path("data/artifacts/warehouse.duckdb")
DDL = Path("warehouse/ddl.sql")


def ensure_warehouse() -> None:
    WAREHOUSE.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(WAREHOUSE))
    try:
        con.execute(DDL.read_text())
    finally:
        con.close()


def upsert_raw(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    con = duckdb.connect(str(WAREHOUSE))
    try:
        con.register("stg", df.to_arrow())
        # Create temp staging table with same columns as raw_listings
        con.execute(
            """
            CREATE TEMP TABLE stg_raw AS SELECT * FROM stg;
            CREATE TEMP TABLE to_upsert AS
            SELECT * FROM stg_raw;
        """
        )
        # Merge by item_id
        con.execute(
            """
            MERGE INTO raw_listings AS t
            USING to_upsert AS s
            ON t.item_id = s.item_id
            WHEN MATCHED THEN UPDATE SET
                title = s.title,
                category_path = s.category_path,
                brand = s.brand,
                model = s.model,
                condition = s.condition,
                start_time = s.start_time,
                end_time = s.end_time,
                listing_type = s.listing_type,
                start_price = s.start_price,
                shipping_cost = s.shipping_cost,
                seller_username = s.seller_username,
                seller_feedback_score = s.seller_feedback_score,
                seller_positive_percent = s.seller_positive_percent,
                watchers = s.watchers,
                bids = s.bids,
                currency = s.currency,
                _ingested_at = now()
            WHEN NOT MATCHED THEN INSERT SELECT * FROM s;
        """
        )
        # Optionally refresh canonical table
        con.execute(
            """
            INSERT INTO listings
            SELECT * FROM to_upsert
            ON CONFLICT (item_id) DO UPDATE SET
                title=excluded.title,
                category_path=excluded.category_path,
                brand=excluded.brand,
                model=excluded.model,
                condition=excluded.condition,
                start_time=excluded.start_time,
                end_time=excluded.end_time,
                listing_type=excluded.listing_type,
                start_price=excluded.start_price,
                shipping_cost=excluded.shipping_cost,
                seller_username=excluded.seller_username,
                seller_feedback_score=excluded.seller_feedback_score,
                seller_positive_percent=excluded.seller_positive_percent,
                watchers=excluded.watchers,
                bids=excluded.bids,
                currency=excluded.currency;
        """
        )
        cnt = con.execute("SELECT COUNT(*) FROM to_upsert").fetchone()[0]
        return int(cnt)
    finally:
        con.close()
