from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import train_test_split

from ebay_price.features.align import align_to_columns
from ebay_price.modeling.loaders import load_processed_features, load_reg_model_and_columns

ART = Path("data/artifacts/models")
PLOTS = Path("data/artifacts/plots")
PLOTS.mkdir(parents=True, exist_ok=True)


def _to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas()


def compute_permutation_importance(n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    model, cols, _ = load_reg_model_and_columns()
    feat = load_processed_features()
    if cols:
        feat = align_to_columns(feat, cols)
    X = _to_pandas(feat.drop("final_price", strict=False))
    y = _to_pandas(feat.select("final_price"))["final_price"]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=random_state)
    (
        model.fit(X_tr, y_tr)
        if hasattr(model, "fit") and not hasattr(model, "feature_importances_")
        else None
    )
    perm = permutation_importance(
        model, X_va, y_va, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    )
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    out = PLOTS / "perm_importance.csv"
    df.to_csv(out, index=False)
    return df


def compute_native_importance() -> pd.DataFrame | None:
    model, cols, _ = load_reg_model_and_columns()
    if not hasattr(model, "feature_importances_"):
        return None
    feat = load_processed_features()
    if cols:
        feat = align_to_columns(feat, cols)
    X = _to_pandas(feat.drop("final_price", strict=False))
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return None
    df = pd.DataFrame({"feature": X.columns, "gain_or_split": imp}).sort_values(
        "gain_or_split", ascending=False
    )
    df.to_csv(PLOTS / "native_importance.csv", index=False)
    return df


def compute_shap_summary(max_samples: int = 2000) -> Path | None:
    """Compute SHAP summary plot for tree-based models. Returns image path or None."""
    try:
        import shap  # heavy, but available in many setups; if missing we silently skip
    except Exception:
        return None
    model, cols, mname = load_reg_model_and_columns()
    # Only run TreeExplainer for tree models
    if "lightgbm" not in mname.lower():
        return None
    feat = load_processed_features()
    if cols:
        feat = align_to_columns(feat, cols)
    X = _to_pandas(feat.drop("final_price", strict=False))
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
    """Generate PD/ICE plots for selected features. Returns list of (feature, path)."""
    model, cols, _ = load_reg_model_and_columns()
    feat = load_processed_features()
    if cols:
        feat = align_to_columns(feat, cols)
    X = _to_pandas(feat.drop("final_price", strict=False))
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
