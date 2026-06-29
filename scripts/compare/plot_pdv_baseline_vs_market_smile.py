from pathlib import Path
import argparse
import json

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
        description="Plot existing PDV baseline smile against matched market IV smile."
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
        "--market-surface",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/spxw_2021-06-02_155500_iv_surface_dataset.csv",
    )
    parser.add_argument(
        "--out-plot",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/pdv_baseline_vs_market_30d_smile.png",
    )
    parser.add_argument(
        "--out-csv",
        default="/users/4/trest017/urop_pdv/data/processed/thetadata/pdv_baseline_vs_market_30d_smile_errors.csv",
    )

    args = parser.parse_args()

    pdv_path = Path(args.pdv_benchmark)
    params_path = Path(args.pdv_params)
    market_path = Path(args.market_surface)
    plot_path = Path(args.out_plot)
    csv_path = Path(args.out_csv)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pdv = pd.read_csv(pdv_path).rename(columns={"implied_vol": "pdv_iv"}).copy()
    pdv["moneyness"] = pdv["strike"]
    pdv["log_moneyness"] = np.log(pdv["moneyness"])

    with open(params_path) as f:
        params = json.load(f)

    pdv_maturity = params.get("option_maturity", params.get("maturity", 1.0 / 12.0))
    pdv_dte = float(pdv_maturity) * 365.0

    market = pd.read_csv(market_path)
    market = market[bool_series(market["liquidity_filter"])].copy()
    market = market[bool_series(market["iv_valid"])].copy()

    dtes = sorted(market["dte"].unique())
    matched_dte = min(dtes, key=lambda x: abs(x - pdv_dte))

    market_slice = market[market["dte"] == matched_dte].copy()
    market_slice = market_slice.sort_values("log_moneyness")
    pdv = pdv.sort_values("log_moneyness")

    # Interpolate PDV IV onto market log-moneyness points where possible.
    x_pdv = pdv["log_moneyness"].to_numpy(dtype=float)
    y_pdv = pdv["pdv_iv"].to_numpy(dtype=float)

    market_slice["pdv_iv_interp"] = np.interp(
        market_slice["log_moneyness"].to_numpy(dtype=float),
        x_pdv,
        y_pdv,
        left=np.nan,
        right=np.nan,
    )

    market_slice["iv_error_pdv_minus_market"] = (
        market_slice["pdv_iv_interp"] - market_slice["iv"]
    )

    error_slice = market_slice.dropna(subset=["pdv_iv_interp"]).copy()

    rmse = float(np.sqrt(np.mean(error_slice["iv_error_pdv_minus_market"] ** 2)))
    mae = float(np.mean(np.abs(error_slice["iv_error_pdv_minus_market"])))
    bias = float(np.mean(error_slice["iv_error_pdv_minus_market"]))

    error_slice.to_csv(csv_path, index=False)

    plt.figure()
    plt.scatter(
        market_slice["log_moneyness"],
        market_slice["iv"],
        label=f"Market SPXW {int(matched_dte)} DTE",
        s=22,
    )
    plt.plot(
        pdv["log_moneyness"],
        pdv["pdv_iv"],
        marker="o",
        label=f"PDV baseline ~{pdv_dte:.1f} DTE",
    )

    plt.xlabel("log(K / F)")
    plt.ylabel("Implied volatility")
    plt.title("PDV Baseline vs Market IV Smile")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    print("Loaded PDV benchmark:", pdv_path)
    print("Loaded market surface:", market_path)
    print("PDV estimated DTE:", pdv_dte)
    print("Matched market DTE:", matched_dte)
    print("Market rows in matched slice:", len(market_slice))
    print("Rows used for error metrics:", len(error_slice))
    print()
    print("Error metrics on overlapping log-moneyness grid:")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("Bias PDV minus market:", bias)
    print()
    print("Wrote plot:", plot_path)
    print("Wrote error CSV:", csv_path)


if __name__ == "__main__":
    main()
