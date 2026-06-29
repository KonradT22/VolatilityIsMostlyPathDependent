from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def main():
    parser = argparse.ArgumentParser(
        description="Compare dense PDV market-grid smile to matched market IV smile."
    )

    parser.add_argument(
        "--pdv-grid",
        default="/users/4/trest017/urop_pdv/benchmarks/pdv_market_grid/pdv_market_grid_30d.csv",
    )
    parser.add_argument(
        "--market-surface",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/spxw_2021-06-02_155500_iv_surface_dataset.csv",
    )
    parser.add_argument(
        "--market-dte",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--out-plot",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/pdv_market_grid_vs_market_30d_smile.png",
    )
    parser.add_argument(
        "--out-csv",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/pdv_market_grid_vs_market_30d_errors.csv",
    )

    args = parser.parse_args()

    pdv_path = Path(args.pdv_grid)
    market_path = Path(args.market_surface)
    plot_path = Path(args.out_plot)
    csv_path = Path(args.out_csv)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pdv = pd.read_csv(pdv_path).sort_values("log_moneyness").copy()

    market = pd.read_csv(market_path)
    market = market[bool_series(market["liquidity_filter"])].copy()
    market = market[bool_series(market["iv_valid"])].copy()
    market = market[market["dte"] == args.market_dte].copy()
    market = market.sort_values("log_moneyness").copy()

    x_pdv = pdv["log_moneyness"].to_numpy(dtype=float)
    y_pdv = pdv["pdv_iv"].to_numpy(dtype=float)

    market["pdv_iv_interp"] = np.interp(
        market["log_moneyness"].to_numpy(dtype=float),
        x_pdv,
        y_pdv,
        left=np.nan,
        right=np.nan,
    )

    market["iv_error_pdv_minus_market"] = market["pdv_iv_interp"] - market["iv"]

    err = market.dropna(subset=["pdv_iv_interp"]).copy()

    rmse = float(np.sqrt(np.mean(err["iv_error_pdv_minus_market"] ** 2)))
    mae = float(np.mean(np.abs(err["iv_error_pdv_minus_market"])))
    bias = float(np.mean(err["iv_error_pdv_minus_market"]))

    err.to_csv(csv_path, index=False)

    plt.figure()
    plt.scatter(
        market["log_moneyness"],
        market["iv"],
        label=f"Market SPXW {args.market_dte} DTE",
        s=22,
    )
    plt.plot(
        pdv["log_moneyness"],
        pdv["pdv_iv"],
        marker="o",
        label="PDV dense market grid 30D",
    )

    plt.xlabel("log(K / F)")
    plt.ylabel("Implied volatility")
    plt.title("Dense PDV Market Grid vs Market IV Smile")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    print("Loaded PDV grid:", pdv_path)
    print("Loaded market surface:", market_path)
    print("Market DTE:", args.market_dte)
    print("PDV rows:", len(pdv))
    print("Market rows:", len(market))
    print("Rows used for error metrics:", len(err))
    print()
    print("Error metrics:")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("Bias PDV minus market:", bias)
    print()
    print("Wrote plot:", plot_path)
    print("Wrote error CSV:", csv_path)


if __name__ == "__main__":
    main()
