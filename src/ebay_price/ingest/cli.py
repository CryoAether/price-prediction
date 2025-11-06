from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ebay_price.features.build_features import build_features, load_listings, save_outputs
from ebay_price.ingest.load import ensure_warehouse, upsert_raw
from ebay_price.ingest.normalize import to_polars, validate_rows
from ebay_price.ingest.sources import read_csv, read_jsonl


def _rows_from_path(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return read_jsonl(p)
    if p.suffix.lower() == ".csv":
        return read_csv(p)
    raise SystemExit(f"Unsupported file type: {p.suffix}")


def ingest_file(path: str | Path) -> int:
    ensure_warehouse()
    rows = validate_rows(_rows_from_path(path))
    df = to_polars(rows)
    count = upsert_raw(df)
    return count


def refresh_features() -> None:
    df = load_listings()  # from DuckDB
    if df.is_empty():
        print("No listings in warehouse.")
        return
    feat = build_features(df)
    save_outputs(feat)
    print("Refreshed features from warehouse.")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ingest", type=str, help="Path to CSV or JSONL file to ingest")
    p.add_argument(
        "--refresh-features", action="store_true", help="Rebuild processed features from warehouse"
    )
    args = p.parse_args()

    if args.ingest:
        n = ingest_file(args.ingest)
        print(f"Ingested {n} rows from {args.ingest}")

    if args.refresh_features:
        refresh_features()


if __name__ == "__main__":
    main()
