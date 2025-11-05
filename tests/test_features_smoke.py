from __future__ import annotations

import polars as pl

from ebay_price.features.build_features import build_features


def test_build_features_smoke():
    df = pl.DataFrame(
        {
            "item_id": ["a1", "b2"],
            "title": ["Apple iPhone 12 128GB", "Samsung Galaxy S21 256GB"],
            "category_path": [
                "Cell Phones & Accessories > Cell Phones & Smartphones",
                "Cell Phones & Accessories > Cell Phones & Smartphones",
            ],
            "brand": ["Apple", "Samsung"],
            "model": ["iPhone 12", "Galaxy S21"],
            "condition": ["Used", "Used"],
            "start_time": ["2025-08-01T10:00:00Z", "2025-08-02T11:00:00Z"],
            "end_time": ["2025-08-08T10:00:00Z", "2025-08-09T11:00:00Z"],
            "listing_type": ["Auction", "BuyItNow"],
            "start_price": [250.0, 399.0],
            "shipping_cost": [10.0, 0.0],
            "seller_username": ["trusted_seller", "pro_seller"],
            "seller_feedback_score": [1200, 800],
            "seller_positive_percent": [99.2, 98.7],
            "watchers": [15, 7],
            "bids": [12, 0],
            "final_price": [355.0, 399.0],
            "sold": [True, True],
            "currency": ["USD", "USD"],
        }
    )
    feat = build_features(df)
    assert feat.height == 2
    for must in [
        "duration_hours",
        "start_weekday",
        "start_hour",
        "start_month",
        "brand_le",
        "category_path_le",
        "log1p_start_price",
        "log1p_final_price",
        "title_len",
        "title_wc",
    ]:
        assert must in feat.columns
