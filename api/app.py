from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException

from ebay_price.api.schemas import ListingIn
from ebay_price.features.inference import prepare_features_for_inference

ART_DIR = Path("data/artifacts/models")

app = FastAPI(title="eBay Price Prediction API", version="0.1.0")


def _load_model(fname: str):
    f = ART_DIR / fname
    if not f.exists():
        raise FileNotFoundError(f"Model not found: {f}")
    return joblib.load(f)


@app.post("/predict/price")
def predict_price(item: ListingIn) -> dict[str, Any]:
    # Prefer LightGBM if present; else fall back to linear
    model_path = (
        "reg_lightgbm.joblib" if (ART_DIR / "reg_lightgbm.joblib").exists() else "reg_linear.joblib"
    )
    try:
        model = _load_model(model_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    X = prepare_features_for_inference(item.model_dump())
    if X.height == 0:
        raise HTTPException(status_code=400, detail="No features produced from payload.")
    yhat = float(model.predict(X.to_pandas().fillna(0).values)[0])
    return {"model": model_path, "prediction": yhat}


@app.post("/predict/sold")
def predict_sold(item: ListingIn) -> dict[str, Any]:
    # Prefer LightGBM if present; else fall back to logistic
    model_path = (
        "clf_lightgbm.joblib" if (ART_DIR / "clf_lightgbm.joblib").exists() else "clf_logit.joblib"
    )
    try:
        model = _load_model(model_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    X = prepare_features_for_inference(item.model_dump())
    if X.height == 0:
        raise HTTPException(status_code=400, detail="No features produced from payload.")
    proba = float(model.predict_proba(X.to_pandas().fillna(0).values)[0, 1])
    return {"model": model_path, "probability": proba, "label": int(proba >= 0.5)}
