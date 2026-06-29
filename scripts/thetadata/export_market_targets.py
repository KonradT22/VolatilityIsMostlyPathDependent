from pathlib import Path
import argparse
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Export compact market targets from an IV surface diagnostic CSV."
    )
    parser.add_argument(
        "--input",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/spxw_2021-06-02_155500_iv_surface_diagnostics.csv",
    )
    parser.add_argument(
        "--output",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/spxw_2021-06-02_155500_market_targets.json",
    )
    args = parser.parse_args()

    infile = Path(args.input)
    outfile = Path(args.output)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile).sort_values("dte")

    targets = {
        "symbol": df["symbol"].iloc[0],
        "trade_date": df["trade_date"].iloc[0],
        "quote_time": df["quote_time"].iloc[0],
        "source_file": str(infile),
        "targets_by_dte": [],
    }

    for _, row in df.iterrows():
        targets["targets_by_dte"].append({
            "expiration": row["expiration"],
            "dte": int(row["dte"]),
            "forward": float(row["forward"]),
            "clean_iv_count": int(row["clean_iv_count"]),
            "atm_iv": float(row["atm_iv"]),
            "left_5pct_iv": float(row["left_5pct_iv"]),
            "right_5pct_iv": float(row["right_5pct_iv"]),
            "wing_iv_gap_left_minus_right": float(row["wing_iv_gap_left_minus_right"]),
            "atm_skew_slope": float(row["atm_skew_slope"]),
            "parity_resid_std": float(row["parity_resid_std"]),
        })

    with open(outfile, "w") as f:
        json.dump(targets, f, indent=2)

    print("Loaded:", infile)
    print("Wrote:", outfile)
    print(json.dumps(targets, indent=2))


if __name__ == "__main__":
    main()
