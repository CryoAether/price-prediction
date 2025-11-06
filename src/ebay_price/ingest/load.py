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
        # Guarantee canonical table has _ingested_at (CTAS can drop defaults/cols in some versions)
        con.execute("CREATE TABLE IF NOT EXISTS listings AS SELECT * FROM raw_listings WHERE 1=0")
        con.execute("ALTER TABLE listings ADD COLUMN IF NOT EXISTS _ingested_at TIMESTAMP")
    finally:
        con.close()


def upsert_raw(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0

    base_cols = [
        "item_id",
        "title",
        "category_path",
        "brand",
        "model",
        "condition",
        "start_time",
        "end_time",
        "listing_type",
        "start_price",
        "shipping_cost",
        "seller_username",
        "seller_feedback_score",
        "seller_positive_percent",
        "watchers",
        "bids",
        "currency",
    ]
    cols_csv = ", ".join(base_cols)
    cols_csv_s = ", ".join([f"s.{c}" for c in base_cols])

    con = duckdb.connect(str(WAREHOUSE))
    try:
        # Stage incoming rows
        con.register("stg", df.to_arrow())
        con.execute("CREATE TEMP TABLE stg_raw AS SELECT * FROM stg")

        # Ensure canonical has _ingested_at (idempotent)
        con.execute("ALTER TABLE listings ADD COLUMN IF NOT EXISTS _ingested_at TIMESTAMP")

        # 1) UPDATE existing rows in raw_listings
        con.execute(
            """
            UPDATE raw_listings AS t
            SET
                title                   = s.title,
                category_path           = s.category_path,
                brand                   = s.brand,
                model                   = s.model,
                condition               = s.condition,
                start_time              = s.start_time,
                end_time                = s.end_time,
                listing_type            = s.listing_type,
                start_price             = s.start_price,
                shipping_cost           = s.shipping_cost,
                seller_username         = s.seller_username,
                seller_feedback_score   = s.seller_feedback_score,
                seller_positive_percent = s.seller_positive_percent,
                watchers                = s.watchers,
                bids                    = s.bids,
                currency                = s.currency,
                _ingested_at            = now()
            FROM stg_raw AS s
            WHERE t.item_id = s.item_id
            """
        )

        # 2) INSERT new rows into raw_listings (DEFAULT fills _ingested_at)
        con.execute(
            f"""
            INSERT INTO raw_listings (
                {cols_csv}
            )
            SELECT
                {cols_csv_s}
            FROM stg_raw s
            WHERE NOT EXISTS (
                SELECT 1 FROM raw_listings t WHERE t.item_id = s.item_id
            )
            """
        )

        # 3) UPDATE listings (canonical)
        con.execute(
            """
            UPDATE listings AS t
            SET
                title                   = s.title,
                category_path           = s.category_path,
                brand                   = s.brand,
                model                   = s.model,
                condition               = s.condition,
                start_time              = s.start_time,
                end_time                = s.end_time,
                listing_type            = s.listing_type,
                start_price             = s.start_price,
                shipping_cost           = s.shipping_cost,
                seller_username         = s.seller_username,
                seller_feedback_score   = s.seller_feedback_score,
                seller_positive_percent = s.seller_positive_percent,
                watchers                = s.watchers,
                bids                    = s.bids,
                currency                = s.currency,
                _ingested_at            = now()
            FROM stg_raw AS s
            WHERE t.item_id = s.item_id
            """
        )

        # 4) INSERT new rows into listings (must supply _ingested_at explicitly)
        con.execute(
            f"""
            INSERT INTO listings (
                {cols_csv}, _ingested_at
            )
            SELECT
                {cols_csv_s}, now()
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
