from __future__ import annotations

from ebay_price.explain.explainers import (
    compute_native_importance,
    compute_pd_ice,
    compute_permutation_importance,
    compute_shap_summary,
)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--pd", nargs="*", default=["start_price", "shipping_cost", "watchers", "bids"])
    p.add_argument("--no-shap", action="store_true")
    args = p.parse_args()

    print("Computing permutation importance...")
    perm = compute_permutation_importance()
    print(perm.head(10).to_string(index=False))

    nat = compute_native_importance()
    if nat is not None:
        print("\nNative importance (top 10):")
        print(nat.head(10).to_string(index=False))
    else:
        print("\nNative importance: not available for this model.")

    if not args.no_shap:
        shp = compute_shap_summary()
        print(f"\nSHAP summary: {'saved to ' + str(shp) if shp else 'skipped'}")

    print("\nPartial Dependence/ICE:")
    paths = compute_pd_ice(args.pd)
    for f, path in paths:
        print(f"  {f}: {path}")


if __name__ == "__main__":
    main()
