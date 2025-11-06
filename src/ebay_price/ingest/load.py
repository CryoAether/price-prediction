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
        # register Arrow table and stage
        con.register("stg", df.to_arrow())
        con.execute("CREATE TEMP TABLE stg_raw AS SELECT * FROM stg;")

        # 1) UPDATE existing rows in raw_listings
        con.execute(
            """
            UPDATE raw_listings AS t
            SET
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
            FROM stg_raw AS s
            WHERE t.item_id = s.item_id
        """
        )

        # 2) INSERT new rows into raw_listings (anti-join)
        con.execute(
            """
            INSERT INTO raw_listings
            SELECT *
            FROM stg_raw s
            WHERE NOT EXISTS (
                SELECT 1 FROM raw_listings t WHERE t.item_id = s.item_id
            )
        """
        )

        # Repeat for canonical 'listings' table
        con.execute(
            """
            UPDATE listings AS t
            SET
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
                currency = s.currency
            FROM stg_raw AS s
            WHERE t.item_id = s.item_id
        """
        )

        con.execute(
            """
            INSERT INTO listings
            SELECT *
            FROM stg_raw s
            WHERE NOT EXISTS (
                SELECT 1 FROM listings t WHERE t.item_id = s.item_id
            )
        """
        )

        cnt = con.execute("SELECT COUNT(*) FROM stg_raw").fetchone()[0]
        return int(cnt)
    finally:
        con.close()
