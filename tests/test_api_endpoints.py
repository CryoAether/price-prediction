from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.app as api_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(api_app.app)


@pytest.fixture()
def sample_listing() -> dict[str, object]:
    return {
        "item_id": "test123",
        "title": "Apple iPhone 12 128GB",
        "category_path": "Cell Phones & Accessories > Cell Phones & Smartphones",
        "brand": "Apple",
        "model": "iPhone 12",
        "condition": "Used",
        "start_time": "2025-08-01T10:00:00Z",
        "end_time": "2025-08-08T10:00:00Z",
        "listing_type": "Auction",
        "start_price": 250.0,
        "shipping_cost": 10.0,
        "seller_username": "trusted_seller",
        "seller_feedback_score": 1200,
        "seller_positive_percent": 99.2,
        "watchers": 15,
        "bids": 12,
        "currency": "USD",
    }


def test_predict_price_endpoint(monkeypatch: pytest.MonkeyPatch, client: TestClient, sample_listing: dict[str, object]) -> None:
    class DummyRegressor:
        def predict(self, X):  # pragma: no cover - invoked indirectly
            assert X.shape[0] == 1
            return [355.0]

    monkeypatch.setattr(api_app, "_load_model", lambda _: DummyRegressor())

    response = client.post("/predict/price", json=sample_listing)
    assert response.status_code == 200
    body = response.json()
    assert body["model"].endswith(".joblib")
    assert body["prediction"] == pytest.approx(355.0)


def test_predict_sold_endpoint(monkeypatch: pytest.MonkeyPatch, client: TestClient, sample_listing: dict[str, object]) -> None:
    class DummyClassifier:
        def predict_proba(self, X):  # pragma: no cover - invoked indirectly
            assert X.shape[0] == 1
            return np.array([[0.2, 0.8]])

    monkeypatch.setattr(api_app, "_load_model", lambda _: DummyClassifier())

    response = client.post("/predict/sold", json=sample_listing)
    assert response.status_code == 200
    body = response.json()
    assert body["probability"] == pytest.approx(0.8)
    assert body["label"] == 1
