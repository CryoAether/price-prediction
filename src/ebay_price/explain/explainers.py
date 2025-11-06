from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import train_test_split

from ebay_price.features.align import align_to_columns
from ebay_price.modeling.loaders import load_reg_model_and_columns
from ebay_price.modeling.train_baselines import feature_target_split, load_train

ART = Path("data/artifacts/models")
PLOTS = Path("data/artifacts/plots")
PLOTS.mkdir(parents=True, exist_ok=True)


def _to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas()


def compute_permutation_importance(n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    """
    Model-agnostic permutation importance computed on a held-out validation split.
    Uses the same feature pipeline + column alignment as training.
    """
    model, cols, _ = load_reg_model_and_columns()

    # Load the exact training frame the baselines use, then split into X/y
    df = load_train()
    X_pl, y_pl = feature_target_split(df, "final_price")

    # Align to saved training column order for the fitted model
    if cols:
        X_pl = align_to_columns(X_pl, cols)

    X = _to_pandas(X_pl)
    # Ensure numeric columns are float for PD/ICE (sklearn warns on integer dtypes)
    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].astype("float64")
    y = _to_pandas(y_pl.to_frame())["final_price"]

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # If the model has no native importances (e.g., LinearRegression), ensure it's fitted
    if hasattr(model, "fit") and not hasattr(model, "feature_importances_"):
        model.fit(X_tr, y_tr)

    perm = permutation_importance(
        model,
        X_va,
        y_va,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    df_imp = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    (PLOTS / "perm_importance.csv").write_text(df_imp.to_csv(index=False))
    return df_imp


def compute_native_importance() -> pd.DataFrame | None:
    """
    Native feature importance for tree models (e.g., LightGBM).
    Falls back to None for models without `feature_importances_`.
    """
    model, cols, _ = load_reg_model_and_columns()
    if not hasattr(model, "feature_importances_"):
        return None

    df = load_train()
    X_pl, _ = feature_target_split(df, "final_price")
    if cols:
        X_pl = align_to_columns(X_pl, cols)
    X = _to_pandas(X_pl)
    # Ensure numeric columns are float for PD/ICE (sklearn warns on integer dtypes)
    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].astype("float64")

    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return None

    df_imp = (
        pd.DataFrame({"feature": X.columns, "gain_or_split": imp})
        .sort_values("gain_or_split", ascending=False)
        .reset_index(drop=True)
    )
    df_imp.to_csv(PLOTS / "native_importance.csv", index=False)
    return df_imp


def compute_shap_summary(max_samples: int = 2000) -> Path | None:
    """
    SHAP summary for tree models. Returns image path or None if SHAP/model unavailable.
    """
    try:
        import shap  # optional dependency
    except Exception:
        return None

    model, cols, mname = load_reg_model_and_columns()
    if "lightgbm" not in mname.lower():
        return None

    df = load_train()
    X_pl, _ = feature_target_split(df, "final_price")
    if cols:
        X_pl = align_to_columns(X_pl, cols)
    X = _to_pandas(X_pl)
    # Ensure numeric columns are float for PD/ICE (sklearn warns on integer dtypes)
    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].astype("float64")
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        fig = plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        out = PLOTS / "shap_summary.png"
        fig.savefig(out, bbox_inches="tight", dpi=160)
        plt.close(fig)
        return out
    except Exception:
        return None


def compute_pd_ice(features: list[str]) -> list[tuple[str, Path]]:
    """
    Generate Partial Dependence + ICE plots for selected features.
    Returns list of (feature, image_path).
    """
    model, cols, _ = load_reg_model_and_columns()
    df = load_train()
    X_pl, _ = feature_target_split(df, "final_price")
    if cols:
        X_pl = align_to_columns(X_pl, cols)
    X = _to_pandas(X_pl)
    # Ensure numeric columns are float for PD/ICE (sklearn warns on integer dtypes)
    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].astype("float64")

    paths: list[tuple[str, Path]] = []
    for f in features:
        if f not in X.columns:
            continue
        fig = plt.figure()
        try:
            PartialDependenceDisplay.from_estimator(model, X, [f], kind="both")
            out = PLOTS / f"pd_ice_{f}.png"
            fig.savefig(out, bbox_inches="tight", dpi=160)
            plt.close(fig)
            paths.append((f, out))
        except Exception:
            plt.close(fig)
            continue
    return paths
