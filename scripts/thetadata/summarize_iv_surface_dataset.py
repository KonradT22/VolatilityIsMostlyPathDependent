from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def nearest_row(group, target_log_moneyness):
    idx = (group["log_moneyness"] - target_log_moneyness).abs().idxmin()
    return group.loc[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a standardized IV surface dataset by maturity."
    )
    parser.add_argument(
        "--input",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/spxw_2021-06-02_155500_iv_surface_dataset.csv",
    )
    parser.add_argument(
        "--output",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/spxw_2021-06-02_155500_iv_surface_diagnostics.csv",
    )
    args = parser.parse_args()

    infile = Path(args.input)
    outfile = Path(args.output)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)

    clean = df[df["liquidity_filter"] == True].copy()
    clean = clean[clean["iv_valid"] == True].copy()

    rows = []

    for dte, group in clean.groupby("dte"):
        group = group.sort_values("log_moneyness").copy()

        atm = nearest_row(group, 0.0)
        left = nearest_row(group, -0.05)
        right = nearest_row(group, 0.05)

        # Simple skew estimate near ATM using linear regression on |log-moneyness| <= 0.03.
        near_atm = group[group["log_moneyness"].abs() <= 0.03].copy()
        if len(near_atm) >= 3:
            slope, intercept = np.polyfit(
                near_atm["log_moneyness"].to_numpy(),
                near_atm["iv"].to_numpy(),
                1,
            )
        else:
            slope = np.nan
            intercept = np.nan

        rows.append({
            "symbol": group["symbol"].iloc[0],
            "trade_date": group["trade_date"].iloc[0],
            "quote_time": group["quote_time"].iloc[0],
            "expiration": group["expiration"].iloc[0],
            "dte": int(dte),
            "clean_iv_count": len(group),
            "forward": group["forward"].iloc[0],
            "iv_min": group["iv"].min(),
            "iv_median": group["iv"].median(),
            "iv_max": group["iv"].max(),
            "atm_strike": atm["strike"],
            "atm_log_moneyness": atm["log_moneyness"],
            "atm_iv": atm["iv"],
            "left_5pct_strike": left["strike"],
            "left_5pct_log_moneyness": left["log_moneyness"],
            "left_5pct_iv": left["iv"],
            "right_5pct_strike": right["strike"],
            "right_5pct_log_moneyness": right["log_moneyness"],
            "right_5pct_iv": right["iv"],
            "wing_iv_gap_left_minus_right": left["iv"] - right["iv"],
            "atm_skew_slope": slope,
            "atm_skew_intercept": intercept,
            "parity_resid_std": group["parity_residual"].std(),
        })

    out = pd.DataFrame(rows).sort_values("dte")
    out.to_csv(outfile, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("Loaded:", infile)
    print("Rows:", len(df))
    print("Clean rows:", len(clean))
    print("\nDiagnostics:")
    print(out)
    print("\nWrote:", outfile)


if __name__ == "__main__":
    main()
