from __future__ import annotations

import polars as pl

from ebay_price.features.build_features import build_features
from ebay_price.modeling.datasets import feature_target_split, train_val_split
from ebay_price.modeling.metrics import classification_metrics, regression_metrics


def _mini_df():
    return pl.DataFrame(
        {
            "item_id": ["a1", "b2"],
            "title": ["Apple iPhone 12 128GB", "Samsung Galaxy S21 256GB"],
            "category_path": ["A > B", "A > B"],
            "brand": ["Apple", "Samsung"],
            "model": ["iPhone 12", "Galaxy S21"],
            "condition": ["Used", "Used"],
            "start_time": ["2025-08-01T10:00:00Z", "2025-08-02T11:00:00Z"],
            "end_time": ["2025-08-08T10:00:00Z", "2025-08-09T11:00:00Z"],
            "listing_type": ["Auction", "BuyItNow"],
            "start_price": [250.0, 399.0],
            "shipping_cost": [10.0, 0.0],
            "seller_username": ["trusted", "pro"],
            "seller_feedback_score": [1200, 800],
            "seller_positive_percent": [99.2, 98.7],
            "watchers": [15, 7],
            "bids": [12, 0],
            "final_price": [355.0, 399.0],
            "sold": [True, True],
            "currency": ["USD", "USD"],
        }
    )


def test_feature_and_split_smoke():
    feat = build_features(_mini_df())
    X, y = feature_target_split(feat, "final_price")
    X_tr, X_va, y_tr, y_va = train_val_split(X, y, test_size=0.5, random_state=0)
    assert X_tr.shape[0] == 1 and X_va.shape[0] == 1
    m = regression_metrics(y_va, y_tr)  # nonsense, but API smoke check
    assert "rmse" in m and "mae" in m


def test_cls_metrics_smoke():
    y_true = [0, 1, 1, 0]
    y_prob = [0.2, 0.8, 0.6, 0.4]
    m = classification_metrics(y_true, y_prob)
    assert "roc_auc" in m and "avg_precision" in m
