from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ebay_price.ingest.ebay_client import EbayClient
from ebay_price.ingest.normalize import normalize_item
from ebay_price.ingest.writer import duckdb_upsert_listings, write_parquet
from ebay_price.utils.settings import load_settings


def main():
    parser = argparse.ArgumentParser(description="Ingest eBay listing data")
    parser.add_argument("--mode", choices=["local", "completed", "active"], default="local")
    parser.add_argument("--input", type=str, help="Path to local JSONL when mode=local")
    parser.add_argument("--query", type=str, default="iphone")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    cfg = load_settings()
    client = EbayClient(site=cfg.ebay__site)

    raw_items: list[dict[str, Any]]
    if args.mode == "local":
        if not args.input:
            raise SystemExit("Provide --input path to a JSONL file in local mode.")
        raw_items = EbayClient.load_local_jsonl(args.input)
    elif args.mode == "completed":
        raw_items = client.list_completed(query=args.query, limit=args.limit)
    else:
        raw_items = client.list_active(query=args.query, limit=args.limit)

    # Normalize
    listings = [normalize_item(r) for r in raw_items if r]
    # Write to Parquet (snapshot)
    parquet_path = Path(cfg.storage__parquet_bucket) / "listings_snapshot.parquet"
    write_parquet(listings, parquet_path)

    # Upsert into DuckDB warehouse
    duckdb_upsert_listings(cfg.storage__duckdb_path, listings)

    print(f"Ingested {len(listings)} records.")
    print(f"Parquet: {parquet_path}")
    print(f"DuckDB:  {cfg.storage__duckdb_path}")


if __name__ == "__main__":
    main()
