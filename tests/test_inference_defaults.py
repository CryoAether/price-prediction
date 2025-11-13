from __future__ import annotations

import pandas as pd

from ebay_price.features.inference import prepare_features_for_inference


def test_inference_adds_default_datetimes() -> None:
    payload = pd.DataFrame(
        [
            {
                "item_id": "missing-times",
                "title": "Sample listing",
                "listing_type": "Auction",
                "start_price": 100.0,
                "shipping_cost": 0.0,
                "watchers": 0,
                "bids": 0,
                "seller_feedback_score": 0,
                "seller_positive_percent": 100.0,
                "currency": "USD",
            }
        ]
    )

    feats = prepare_features_for_inference(payload)
    assert feats.height == 1
    assert "duration_hours" in feats.columns
    assert feats["duration_hours"][0] > 0
    assert feats["start_dt"][0] is not None
    assert feats["end_dt"][0] is not None
