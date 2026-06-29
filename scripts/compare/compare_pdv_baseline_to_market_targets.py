from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd


def nearest_row(df, column, target):
    idx = (df[column] - target).abs().idxmin()
    return df.loc[idx]


def estimate_skew(df):
    near = df[df["log_moneyness"].abs() <= 0.03].copy()

    if len(near) < 3:
        return np.nan, np.nan

    slope, intercept = np.polyfit(
        near["log_moneyness"].to_numpy(dtype=float),
        near["pdv_iv"].to_numpy(dtype=float),
        1,
    )

    return slope, intercept


def main():
    parser = argparse.ArgumentParser(
        description="Compare existing PDV baseline IV smile to compact market targets."
    )

    parser.add_argument(
        "--pdv-benchmark",
        default="/users/4/trest017/urop_pdv/benchmarks/msi_baseline/spx_option_benchmark.csv",
    )
    parser.add_argument(
        "--pdv-params",
        default="/users/4/trest017/urop_pdv/benchmarks/msi_baseline/baseline_parameters.json",
    )
    parser.add_argument(
        "--market-targets",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/spxw_2021-06-02_155500_market_targets.json",
    )
    parser.add_argument(
        "--output",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/pdv_baseline_vs_market_targets_2021-06-02_155500.csv",
    )

    args = parser.parse_args()

    pdv_path = Path(args.pdv_benchmark)
    params_path = Path(args.pdv_params)
    market_path = Path(args.market_targets)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdv = pd.read_csv(pdv_path)

    with open(params_path) as f:
        params = json.load(f)

    with open(market_path) as f:
        market = json.load(f)

    # Existing baseline uses relative strikes, roughly K/S0 or K/F0.
    # Treat these as relative moneyness for the first bridge comparison.
    pdv = pdv.rename(columns={"implied_vol": "pdv_iv"}).copy()
    pdv["moneyness"] = pdv["strike"]
    pdv["log_moneyness"] = np.log(pdv["moneyness"])

    option_maturity_years = params.get("option_maturity", params.get("maturity", None))

    if option_maturity_years is None:
        print("Could not find option_maturity or maturity in baseline parameters.")
        print("Available parameter keys:", sorted(params.keys()))
        option_maturity_years = 1.0 / 12.0

    pdv_dte_estimate = float(option_maturity_years) * 365.0

    targets = pd.DataFrame(market["targets_by_dte"])
    targets["dte_distance"] = (targets["dte"] - pdv_dte_estimate).abs()
    market_row = targets.sort_values("dte_distance").iloc[0]

    atm = nearest_row(pdv, "log_moneyness", 0.0)
    left = nearest_row(pdv, "log_moneyness", -0.05)
    right = nearest_row(pdv, "log_moneyness", 0.05)

    pdv_skew_slope, pdv_skew_intercept = estimate_skew(pdv)

    comparison = {
        "market_symbol": market["symbol"],
        "market_trade_date": market["trade_date"],
        "market_quote_time": market["quote_time"],
        "pdv_option_maturity_years": float(option_maturity_years),
        "pdv_dte_estimate": pdv_dte_estimate,
        "matched_market_expiration": market_row["expiration"],
        "matched_market_dte": int(market_row["dte"]),

        "pdv_atm_iv": float(atm["pdv_iv"]),
        "market_atm_iv": float(market_row["atm_iv"]),
        "atm_iv_error_pdv_minus_market": float(atm["pdv_iv"] - market_row["atm_iv"]),

        "pdv_left_5pct_iv": float(left["pdv_iv"]),
        "market_left_5pct_iv": float(market_row["left_5pct_iv"]),
        "left_5pct_iv_error_pdv_minus_market": float(left["pdv_iv"] - market_row["left_5pct_iv"]),

        "pdv_right_5pct_iv": float(right["pdv_iv"]),
        "market_right_5pct_iv": float(market_row["right_5pct_iv"]),
        "right_5pct_iv_error_pdv_minus_market": float(right["pdv_iv"] - market_row["right_5pct_iv"]),

        "pdv_wing_iv_gap_left_minus_right": float(left["pdv_iv"] - right["pdv_iv"]),
        "market_wing_iv_gap_left_minus_right": float(market_row["wing_iv_gap_left_minus_right"]),
        "wing_gap_error_pdv_minus_market": float((left["pdv_iv"] - right["pdv_iv"]) - market_row["wing_iv_gap_left_minus_right"]),

        "pdv_atm_skew_slope": float(pdv_skew_slope),
        "market_atm_skew_slope": float(market_row["atm_skew_slope"]),
        "atm_skew_slope_error_pdv_minus_market": float(pdv_skew_slope - market_row["atm_skew_slope"]),

        "pdv_rows": len(pdv),
        "market_clean_iv_count": int(market_row["clean_iv_count"]),
    }

    out = pd.DataFrame([comparison])
    out.to_csv(out_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    print("Loaded PDV benchmark:", pdv_path)
    print("Loaded PDV params:", params_path)
    print("Loaded market targets:", market_path)
    print("\nPDV benchmark preview:")
    print(pdv.head())
    print("\nPDV estimated DTE:", pdv_dte_estimate)
    print("Matched market DTE:", int(market_row["dte"]))
    print("Matched market expiration:", market_row["expiration"])
    print("\nComparison:")
    print(out)
    print("\nWrote:", out_path)


if __name__ == "__main__":
    main()
