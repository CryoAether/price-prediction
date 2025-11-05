from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, validator


class ListingRecord(BaseModel):
    item_id: str = Field(..., min_length=1)
    title: str | None = None
    category_path: str | None = None
    brand: str | None = None
    model: str | None = None
    condition: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    listing_type: str | None = Field(None, description="Auction or BuyItNow")
    start_price: float | None = Field(None, ge=0)
    shipping_cost: float | None = Field(None, ge=0)
    return_policy: str | None = None
    seller_username: str | None = None
    seller_feedback_score: int | None = Field(None, ge=0)
    seller_positive_percent: float | None = Field(None, ge=0, le=100)
    watchers: int | None = Field(None, ge=0)
    bids: int | None = Field(None, ge=0)
    final_price: float | None = Field(None, ge=0)
    sold: bool | None = None
    currency: str | None = Field(None, description="ISO currency like USD")

    @validator("listing_type")
    def _listing_type_ok(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"Auction", "BuyItNow", "BIN", "FixedPrice"}
        if v not in allowed:
            raise ValueError(f"listing_type must be one of {allowed}, got {v}")
        return v

    @validator("currency")
    def _currency_ok(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) != 3:
            raise ValueError("currency must be a 3-letter code")
        return v

    @validator("end_time")
    def _end_after_start(cls, v: datetime | None, values) -> datetime | None:
        start = values.get("start_time")
        if v and start and v < start:
            raise ValueError("end_time must be after start_time")
        return v
