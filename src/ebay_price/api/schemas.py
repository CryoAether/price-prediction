from __future__ import annotations

from pydantic import BaseModel, Field


class ListingIn(BaseModel):
    item_id: str = Field(..., min_length=1)
    title: str | None = None
    category_path: str | None = None
    brand: str | None = None
    model: str | None = None
    condition: str | None = None
    start_time: str | None = None  # ISO8601
    end_time: str | None = None  # ISO8601
    listing_type: str | None = None
    start_price: float | None = 0.0
    shipping_cost: float | None = 0.0
    seller_username: str | None = None
    seller_feedback_score: int | None = 0
    seller_positive_percent: float | None = 100.0
    watchers: int | None = 0
    bids: int | None = 0
    currency: str | None = "USD"
