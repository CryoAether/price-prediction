from pathlib import Path

import pandas as pd
import streamlit as st

from ebay_price.features.align import align_to_columns
from ebay_price.features.inference import prepare_features_for_inference

ART_DIR = Path("data/artifacts/models")

st.set_page_config(page_title="eBay Price Prediction", page_icon="📈", layout="wide")
st.title("eBay Price Prediction")
tabs = st.tabs(["Predict", "Insights"])

with tabs[0]:
    # Short caption line to satisfy Ruff E501
    model_name = (
        ART_DIR
        / (
            "reg_lightgbm.joblib"
            if (ART_DIR / "reg_lightgbm.joblib").exists()
            else "reg_linear.joblib"
        )
    ).name
    st.caption(f"Using model: {model_name}")

    st.header("Listing")
    with st.form("predict_form"):
        title = st.text_input("Title", value="Apple iPhone 12 128GB")
        brand = st.selectbox("Brand", ["Apple", "Samsung", "Google", "OnePlus"], index=0)
        model = st.selectbox(
            "Model", ["iPhone 12", "iPhone 13", "Galaxy S21", "Pixel 6", "OnePlus 9"], index=0
        )
        category_path = st.text_input(
            "Category Path", "Cell Phones & Accessories > Cell Phones & Smartphones"
        )
        condition = st.selectbox("Condition", ["Used", "Refurbished", "New"], index=0)
        start_price = st.number_input(
            "Start Price", min_value=0.0, value=250.0, step=5.0, format="%.2f"
        )
        shipping_cost = st.number_input(
            "Shipping Cost", min_value=0.0, value=10.0, step=1.0, format="%.2f"
        )
        watchers = st.number_input("Watchers", min_value=0, value=15, step=1)
        bids = st.number_input("Bids", min_value=0, value=12, step=1)

        submitted = st.form_submit_button("Predict price")
        if submitted:
            payload = {
                "title": title,
                "brand": brand,
                "model": model,
                "category_path": category_path,
                "condition": condition,
                "start_price": float(start_price),
                "shipping_cost": float(shipping_cost),
                "watchers": int(watchers),
                "bids": int(bids),
                "currency": "USD",
            }
            # ✅ FIX: pass a pandas DataFrame, not a dict
            X_pl = prepare_features_for_inference(pd.DataFrame([payload]))

            # Align to training feature columns if saved
            cols_path = ART_DIR / "reg_feature_columns.json"
            if cols_path.exists():
                X_pl = align_to_columns(X_pl, cols_path.read_text())

            # Load model and predict
            from ebay_price.modeling.loaders import load_reg_model_and_columns

            model_obj, cols, mname = load_reg_model_and_columns()
            if cols:
                X_pl = align_to_columns(X_pl, cols)
            pred = float(model_obj.predict(X_pl.to_pandas().fillna(0))[0])
            st.caption(f"Using model: {mname}")
            st.success(f"Predicted price: ${pred:,.2f}")

with tabs[1]:
    st.header("Model Insights")
    from pathlib import Path as _P

    import pandas as _pd

    plots = _P("data/artifacts/plots")
    perm_csv = plots / "perm_importance.csv"
    nat_csv = plots / "native_importance.csv"
    shap_img = plots / "shap_summary.png"
    imgs = sorted(plots.glob("pd_ice_*.png"))

    if perm_csv.exists():
        st.subheader("Permutation importance")
        st.dataframe(_pd.read_csv(perm_csv).head(30))
    else:
        st.info("Run `make explain-no-shap` (or `make explain`) to compute insights.")

    if nat_csv.exists():
        st.subheader("Native feature importance")
        st.dataframe(_pd.read_csv(nat_csv).head(30))

    if shap_img.exists():
        st.subheader("SHAP summary")
        st.image(str(shap_img), width="stretch")

    st.subheader("Partial Dependence / ICE")
    if imgs:
        for im in imgs:
            st.image(str(im), caption=im.name, width="stretch")
    else:
        st.caption("No PD/ICE plots yet. Run `make explain-no-shap`.")
