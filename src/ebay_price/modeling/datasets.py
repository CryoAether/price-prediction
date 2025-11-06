from __future__ import annotations

from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path("data/processed")


def load_train(path: str | None = None) -> pl.DataFrame:
    p = Path(path) if path else PROCESSED_DIR / "train.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Training parquet not found: {p}")
    return pl.read_parquet(p)


def feature_target_split(
    df: pl.DataFrame, target: str, drop_cols: list[str] | None = None
) -> tuple[pl.DataFrame, pl.Series]:
    drop = set(drop_cols or [])
    drop.update([target, "item_id", "title", "start_time", "end_time"])
    cols = [c for c in df.columns if c not in drop]
    X = df.select(cols)
    y = df.get_column(target)
    return X, y


def to_numpy(df: pl.DataFrame):
    return df.to_pandas().values  # simple and compatible with most estimators


def train_val_split(X: pl.DataFrame, y: pl.Series, test_size: float = 0.2, random_state: int = 42):
    X_np = to_numpy(X)
    y_np = y.to_pandas().values
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_state
    )
    return X_tr, X_va, y_tr, y_va
