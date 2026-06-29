from pathlib import Path
import argparse
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Compare multi-maturity PDV grid summary against compact market targets."
    )

    parser.add_argument(
        "--pdv-summary",
        default="/users/4/trest017/urop_pdv/benchmarks/pdv_market_grid/pdv_market_grid_multi_maturity_summary.csv",
    )
    parser.add_argument(
        "--market-targets",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/spxw_2021-06-02_155500_market_targets.json",
    )
    parser.add_argument(
        "--output",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/pdv_multi_maturity_vs_market_targets.csv",
    )

    args = parser.parse_args()

    pdv_path = Path(args.pdv_summary)
    market_path = Path(args.market_targets)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdv = pd.read_csv(pdv_path)

    with open(market_path) as f:
        market_json = json.load(f)

    market = pd.DataFrame(market_json["targets_by_dte"])

    merged = pd.merge(
        pdv,
        market,
        on="dte",
        how="inner",
        suffixes=("_pdv", "_market"),
    )

    merged["atm_iv_error_pdv_minus_market"] = merged["atm_iv_pdv"] - merged["atm_iv_market"]
    merged["left_5pct_iv_error_pdv_minus_market"] = merged["left_5pct_iv_pdv"] - merged["left_5pct_iv_market"]
    merged["right_5pct_iv_error_pdv_minus_market"] = merged["right_5pct_iv_pdv"] - merged["right_5pct_iv_market"]
    merged["wing_gap_error_pdv_minus_market"] = (
        merged["wing_iv_gap_left_minus_right_pdv"]
        - merged["wing_iv_gap_left_minus_right_market"]
    )
    merged["skew_slope_error_pdv_minus_market"] = (
        merged["atm_skew_slope_pdv"]
        - merged["atm_skew_slope_market"]
    )

    cols = [
        "dte",
        "expiration",
        "atm_iv_pdv",
        "atm_iv_market",
        "atm_iv_error_pdv_minus_market",
        "left_5pct_iv_pdv",
        "left_5pct_iv_market",
        "left_5pct_iv_error_pdv_minus_market",
        "right_5pct_iv_pdv",
        "right_5pct_iv_market",
        "right_5pct_iv_error_pdv_minus_market",
        "wing_iv_gap_left_minus_right_pdv",
        "wing_iv_gap_left_minus_right_market",
        "wing_gap_error_pdv_minus_market",
        "atm_skew_slope_pdv",
        "atm_skew_slope_market",
        "skew_slope_error_pdv_minus_market",
        "pricing_seconds",
    ]

    existing_cols = [c for c in cols if c in merged.columns]
    out = merged[existing_cols].sort_values("dte").copy()
    out.to_csv(out_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    print("Loaded PDV summary:", pdv_path)
    print("Loaded market targets:", market_path)
    print("\nComparison:")
    print(out)

    print("\nAggregate errors:")
    metric_cols = [
        "atm_iv_error_pdv_minus_market",
        "left_5pct_iv_error_pdv_minus_market",
        "right_5pct_iv_error_pdv_minus_market",
        "wing_gap_error_pdv_minus_market",
        "skew_slope_error_pdv_minus_market",
    ]
    print(out[metric_cols].describe())

    print("\nWrote:", out_path)


if __name__ == "__main__":
    main()
