from __future__ import annotations

import json
from pathlib import Path

import joblib
import streamlit as st

from ebay_price.features.align import align_to_columns
from ebay_price.features.inference import prepare_features_for_inference

ART_DIR = Path("data/artifacts/models")

st.title("eBay Price Prediction — Demo")

tabs = st.tabs(["Predict", "Insights"])

with st.sidebar:
    st.header("Model selection")
    reg_light = ART_DIR / "reg_lightgbm.joblib"
    reg_lin = ART_DIR / "reg_linear.joblib"
    if reg_light.exists():
        model_path = reg_light
    elif reg_lin.exists():
        model_path = reg_lin
    else:
        st.error("No regression model artifact found. Run `make train-regression` first.")
        st.stop()

st.caption(f"Using model: {model_path.name}")

st.subheader("Listing")
title = st.text_input("Title", "Apple iPhone 12 128GB")
brand = st.text_input("Brand", "Apple")
model = st.text_input("Model", "iPhone 12")
category = st.text_input("Category Path", "Cell Phones & Accessories > Cell Phones & Smartphones")
condition = st.selectbox("Condition", ["Used", "New", "Refurbished"], index=0)
start_price = st.number_input("Start Price", min_value=0.0, value=250.0)
shipping = st.number_input("Shipping Cost", min_value=0.0, value=10.0)
watchers = st.number_input("Watchers", min_value=0, value=15)
bids = st.number_input("Bids", min_value=0, value=12)

if st.button("Predict price"):
    payload = {
        "item_id": "demo",
        "title": title,
        "category_path": category,
        "brand": brand,
        "model": model,
        "condition": condition,
        "start_time": "2025-08-01T10:00:00Z",
        "end_time": "2025-08-08T10:00:00Z",
        "listing_type": "Auction",
        "start_price": float(start_price),
        "shipping_cost": float(shipping),
        "seller_feedback_score": 1200,
        "seller_positive_percent": 99.2,
        "watchers": int(watchers),
        "bids": int(bids),
        "currency": "USD",
    }
    X = prepare_features_for_inference(payload)
    # Align to training feature columns saved during training
    cols_path = ART_DIR / "reg_feature_columns.json"
    if cols_path.exists():
        cols = json.loads(cols_path.read_text())
        X = align_to_columns(X, cols)
    model = joblib.load(model_path)
    pred = float(model.predict(X.to_pandas().fillna(0))[0])
    st.success(f"Predicted price: ${pred:,.2f}")


with tabs[1]:
    st.header("Model Insights")
    from pathlib import Path

    import pandas as pd

    plots = Path("data/artifacts/plots")
    perm_csv = plots / "perm_importance.csv"
    nat_csv = plots / "native_importance.csv"
    if perm_csv.exists():
        st.subheader("Permutation importance")
        st.dataframe(pd.read_csv(perm_csv).head(20))
    else:
        st.info("Run `make explain` to compute insights.")
    if nat_csv.exists():
        st.subheader("Native feature importance")
        st.dataframe(pd.read_csv(nat_csv).head(20))
    shap_img = plots / "shap_summary.png"
    if shap_img.exists():
        st.subheader("SHAP summary")
        st.image(str(shap_img), use_container_width=True)
    # PD/ICE gallery
    st.subheader("Partial Dependence / ICE")
    imgs = list(plots.glob("pd_ice_*.png"))
    if imgs:
        for im in imgs:
            st.image(str(im), caption=im.name, use_container_width=True)
    else:
        st.caption("No PD/ICE plots yet. Run `make explain`.")
