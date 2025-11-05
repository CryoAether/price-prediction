CREATE TABLE IF NOT EXISTS staging_listings (
    ingest_ts TIMESTAMPTZ DEFAULT NOW(),
    raw_json  JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS listings (
    item_id TEXT PRIMARY KEY,
    title TEXT,
    category_path TEXT,
    brand TEXT,
    model TEXT,
    condition TEXT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    listing_type TEXT,         -- BIN or Auction
    start_price NUMERIC,
    shipping_cost NUMERIC,
    return_policy TEXT,
    seller_username TEXT,
    seller_feedback_score INT,
    seller_positive_percent NUMERIC,
    watchers INT,
    bids INT,
    final_price NUMERIC,
    sold BOOLEAN,
    currency TEXT
);

CREATE INDEX IF NOT EXISTS idx_listings_category ON listings (category_path);
CREATE INDEX IF NOT EXISTS idx_listings_times ON listings (start_time, end_time);
