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

    numeric_like = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Boolean,
    }
    cols: list[str] = []
    for c, t in zip(df.columns, df.dtypes, strict=False):
        if c in drop:
            continue
        if t in numeric_like:
            cols.append(c)

    X = df.select(cols)
    y = df.get_column(target)
    return X, y


def to_numpy(df: pl.DataFrame):
    return df.to_pandas().fillna(0).values  # simple and compatible with most estimators


def train_val_split(
    X: pl.DataFrame,
    y: pl.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
):
    X_np = X.to_pandas().fillna(0).values
    y_np = y.to_pandas().values
    strat = y_np if stratify else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_state, stratify=strat
    )
    return X_tr, X_va, y_tr, y_va
