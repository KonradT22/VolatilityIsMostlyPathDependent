from pathlib import Path
import argparse
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot multi-maturity PDV-vs-market target comparisons."
    )

    parser.add_argument(
        "--input",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/pdv_multi_maturity_vs_market_targets.csv",
    )
    parser.add_argument(
        "--out-prefix",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/pdv_multi_maturity_vs_market",
    )

    args = parser.parse_args()

    infile = Path(args.input)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile).sort_values("dte")

    plots = [
        (
            "atm_iv",
            "atm_iv_pdv",
            "atm_iv_market",
            "ATM Implied Volatility Term Structure",
            "ATM implied volatility",
        ),
        (
            "left_5pct_iv",
            "left_5pct_iv_pdv",
            "left_5pct_iv_market",
            "Left-Wing IV Term Structure",
            "Left 5% implied volatility",
        ),
        (
            "right_5pct_iv",
            "right_5pct_iv_pdv",
            "right_5pct_iv_market",
            "Right-Wing IV Term Structure",
            "Right 5% implied volatility",
        ),
        (
            "wing_gap",
            "wing_iv_gap_left_minus_right_pdv",
            "wing_iv_gap_left_minus_right_market",
            "Left-Right Wing Gap by Maturity",
            "Left minus right IV",
        ),
        (
            "skew_slope",
            "atm_skew_slope_pdv",
            "atm_skew_slope_market",
            "ATM Skew Slope by Maturity",
            "ATM skew slope",
        ),
    ]

    written = []

    for name, pdv_col, market_col, title, ylabel in plots:
        out_path = Path(f"{out_prefix}_{name}.png")

        plt.figure()
        plt.plot(df["dte"], df[pdv_col], marker="o", label="PDV")
        plt.plot(df["dte"], df[market_col], marker="o", label="Market")
        plt.xlabel("DTE")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)

        written.append(out_path)

    error_path = Path(f"{out_prefix}_errors.png")

    plt.figure()
    plt.plot(df["dte"], df["atm_iv_error_pdv_minus_market"], marker="o", label="ATM IV error")
    plt.plot(df["dte"], df["left_5pct_iv_error_pdv_minus_market"], marker="o", label="Left 5% IV error")
    plt.plot(df["dte"], df["right_5pct_iv_error_pdv_minus_market"], marker="o", label="Right 5% IV error")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("DTE")
    plt.ylabel("PDV minus market IV")
    plt.title("PDV Minus Market IV Errors by Maturity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(error_path, dpi=150)
    written.append(error_path)

    print("Loaded:", infile)
    print("Rows:", len(df))
    print("\nWrote plots:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
