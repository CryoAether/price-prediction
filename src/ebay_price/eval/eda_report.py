from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import polars as pl

REPORT_DIR = Path("reports")
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_duckdb(db="data/artifacts/warehouse.duckdb") -> pl.DataFrame:
    con = duckdb.connect(db)
    df = con.execute("SELECT * FROM listings").pl()
    con.close()
    return df


def save_csv(df: pl.DataFrame, name: str):
    df.write_csv(TABLE_DIR / f"{name}.csv")


def plot_hist(series: pl.Series, title: str, fname: str):
    s = series.drop_nulls()
    if s.is_empty():
        return
    plt.figure()
    plt.hist(s.to_list(), bins=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname)
    plt.close()


def plot_bar(series: pl.Series, title: str, fname: str):
    s = series.drop_nulls()
    if s.is_empty():
        return
    counts = s.value_counts().sort("count", descending=True).head(20)
    plt.figure(figsize=(8, 4))
    plt.bar(counts[s.name], counts["count"])
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname)
    plt.close()


def main():
    df = load_duckdb()
    print(f"Listings: {df.height}")
    # Numeric summary
    numeric = [
        c
        for c, t in zip(df.columns, df.dtypes, strict=False)
        if "Int" in str(t) or "Float" in str(t)
    ]
    df.select(numeric).describe().write_csv(TABLE_DIR / "numeric_summary.csv")

    # Missingness
    miss = pl.DataFrame(
        {"column": df.columns, "missing": [df.get_column(c).null_count() for c in df.columns]}
    )
    miss.write_csv(TABLE_DIR / "missingness.csv")

    # Top brands and categories
    if "brand" in df.columns:
        plot_bar(df["brand"], "Top Brands", "brands.png")
    if "category_path" in df.columns:
        plot_bar(df["category_path"], "Top Categories", "categories.png")

    # Price histograms
    for col in ("start_price", "final_price", "shipping_cost"):
        if col in df.columns:
            plot_hist(df[col], f"Distribution of {col}", f"{col}.png")

    print("EDA written to reports/")


if __name__ == "__main__":
    main()
