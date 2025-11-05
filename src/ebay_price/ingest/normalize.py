from __future__ import annotations

from typing import Any


def safe_get(d: dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def normalize_item(raw: dict[str, Any]) -> dict[str, Any]:
    # This mapping assumes a generic eBay-like shape; adjust keys when you wire real API
    item = {
        "item_id": raw.get("itemId") or raw.get("item_id") or raw.get("id"),
        "title": raw.get("title"),
        "category_path": " > ".join(
            safe_get(raw, "categoryPath", []) or safe_get(raw, "category.path", []) or []
        ),
        "brand": safe_get(raw, "itemSpecifics.brand") or safe_get(raw, "brand"),
        "model": safe_get(raw, "itemSpecifics.model") or safe_get(raw, "model"),
        "condition": raw.get("condition") or safe_get(raw, "condition.displayName"),
        "start_time": raw.get("startTime") or raw.get("start_time"),
        "end_time": raw.get("endTime") or raw.get("end_time"),
        "listing_type": raw.get("listingType") or raw.get("format"),
        "start_price": safe_get(raw, "price.value") or raw.get("startPrice"),
        "shipping_cost": safe_get(raw, "shipping.price.value") or raw.get("shippingCost"),
        "return_policy": safe_get(raw, "returnPolicy.returnsAccepted"),
        "seller_username": safe_get(raw, "seller.username") or raw.get("seller"),
        "seller_feedback_score": safe_get(raw, "seller.feedbackScore"),
        "seller_positive_percent": safe_get(raw, "seller.positivePercent"),
        "watchers": raw.get("watchCount") or raw.get("watchers"),
        "bids": raw.get("bidCount") or raw.get("bids"),
        "final_price": safe_get(raw, "finalPrice.value") or raw.get("finalPrice"),
        "sold": bool(raw.get("sold") or raw.get("endedAsSold")),
        "currency": safe_get(raw, "price.currency") or raw.get("currency"),
    }
    return item
