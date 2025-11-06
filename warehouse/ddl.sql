-- Raw landing table (as ingested)
CREATE TABLE IF NOT EXISTS raw_listings (
    item_id TEXT PRIMARY KEY,
    title TEXT,
    category_path TEXT,
    brand TEXT,
    model TEXT,
    condition TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    listing_type TEXT,
    start_price DOUBLE,
    shipping_cost DOUBLE,
    seller_username TEXT,
    seller_feedback_score BIGINT,
    seller_positive_percent DOUBLE,
    watchers BIGINT,
    bids BIGINT,
    currency TEXT,
    _ingested_at TIMESTAMP DEFAULT now()
);

-- Canonical "listings" table (lightly cleaned)
CREATE TABLE IF NOT EXISTS listings AS
SELECT * FROM raw_listings
WHERE 1=0;

-- Upsert raw → raw_listings
-- We do merges from staging temp tables created by the loader.

-- Canonicalization view: future place for denormalization if needed
CREATE OR REPLACE VIEW v_listings AS
SELECT
    item_id,
    title,
    category_path,
    brand,
    model,
    condition,
    start_time,
    end_time,
    listing_type,
    COALESCE(start_price, 0.0) AS start_price,
    COALESCE(shipping_cost, 0.0) AS shipping_cost,
    seller_username,
    COALESCE(seller_feedback_score, 0) AS seller_feedback_score,
    COALESCE(seller_positive_percent, 100.0) AS seller_positive_percent,
    COALESCE(watchers, 0) AS watchers,
    COALESCE(bids, 0) AS bids,
    COALESCE(currency, 'USD') AS currency
FROM raw_listings;
