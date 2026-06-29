import json
import time
from pathlib import Path
import sys
from datetime import timedelta

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration.torch_montecarlo import (
    TorchMonteCarloExponentialModel,
    initialize_R,
    identity,
    squared,
)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_spx_history(asofdate, outdir):
    cached = Path("/users/4/trest017/urop_pdv/benchmarks/msi_baseline/spx_history_yfinance.csv")

    if cached.exists():
        print("Loading cached SPX history:", cached)
        spx_df = pd.read_csv(cached)
        date_col = spx_df.columns[0]
        spx_df[date_col] = pd.to_datetime(spx_df[date_col])
        spx = pd.Series(spx_df["Close"].values, index=spx_df[date_col])
        spx.index = pd.to_datetime(spx.index.date)
        return spx

    load_from = asofdate - timedelta(days=4 * 365)

    print("Cached SPX history not found. Loading SPX history from yfinance...")
    spx = yf.Ticker("^GSPC").history(
        start=load_from,
        end=asofdate + timedelta(days=1),
    )["Close"]

    spx.index = pd.to_datetime(spx.index.date)
    spx.to_csv(outdir / "spx_history_yfinance.csv", header=["Close"])

    return spx


def summarize_slice(df):
    atm_row = df.iloc[(df["log_moneyness"] - 0.0).abs().argsort()].iloc[0]
    left_row = df.iloc[(df["log_moneyness"] + 0.05).abs().argsort()].iloc[0]
    right_row = df.iloc[(df["log_moneyness"] - 0.05).abs().argsort()].iloc[0]

    near_atm = df[df["log_moneyness"].abs() <= 0.03].copy()
    skew_slope, skew_intercept = np.polyfit(
        near_atm["log_moneyness"].to_numpy(),
        near_atm["pdv_iv"].to_numpy(),
        1,
    )

    return {
        "dte": int(df["dte"].iloc[0]),
        "option_maturity": float(df["option_maturity"].iloc[0]),
        "future": float(df["future"].iloc[0]),
        "grid_points": len(df),
        "atm_iv": float(atm_row["pdv_iv"]),
        "left_5pct_iv": float(left_row["pdv_iv"]),
        "right_5pct_iv": float(right_row["pdv_iv"]),
        "wing_iv_gap_left_minus_right": float(left_row["pdv_iv"] - right_row["pdv_iv"]),
        "atm_skew_slope": float(skew_slope),
        "atm_skew_intercept": float(skew_intercept),
    }


def main():
    outdir = Path("/users/4/trest017/urop_pdv/benchmarks/pdv_market_grid")
    outdir.mkdir(parents=True, exist_ok=True)

    seed = 123456
    torch.manual_seed(seed)
    np.random.seed(seed)

    asofdate = pd.to_datetime("2021-06-02")
    spx = load_spx_history(asofdate, outdir)

    lam1 = torch.tensor([55, 10])
    lam2 = torch.tensor([20, 3])
    betas = torch.tensor([0.04, -0.13, 0.65])
    theta1 = 0.25
    theta2 = 0.5
    parabolic = 0
    parabolic_offset = 0

    R_init1 = initialize_R(lam1, past_prices=spx, transform=identity)
    R_init2 = initialize_R(lam2, past_prices=spx, transform=squared)

    N = 10000
    vix_N = 1000
    timestep_per_day = 10

    # Must be at least as long as largest option maturity.
    maturity = 0.30

    target_dtes = [7, 14, 21, 30, 44, 90]
    option_maturities = [dte / 365 for dte in target_dtes]

    log_moneyness = np.linspace(-0.06, 0.06, 49)
    strikes = np.exp(log_moneyness)

    print("Constructing PDV model...")
    torch_mc = TorchMonteCarloExponentialModel(
        lam1=lam1,
        lam2=lam2,
        betas=betas,
        R_init1=R_init1,
        R_init2=R_init2,
        theta1=theta1,
        theta2=theta2,
        N=N,
        vix_N=vix_N,
        maturity=maturity,
        parabolic=parabolic,
        parabolic_offset=parabolic_offset,
        device="cpu",
    )

    print("Running Monte Carlo simulation...")
    t0 = time.perf_counter()
    torch_mc.simulate()
    simulate_seconds = time.perf_counter() - t0

    all_rows = []
    summaries = []

    for dte, option_maturity in zip(target_dtes, option_maturities):
        print(f"Computing PDV implied vols for {dte} DTE...")

        t1 = time.perf_counter()
        future, _, implied_vol, option_prices = torch_mc.compute_implied_vol(
            strikes=strikes,
            option_maturity=option_maturity,
        )
        pricing_seconds = time.perf_counter() - t1

        df = pd.DataFrame({
            "dte": dte,
            "option_maturity": option_maturity,
            "future": float(future),
            "strike": to_numpy(strikes),
            "moneyness": to_numpy(strikes),
            "log_moneyness": log_moneyness,
            "pdv_option_price": to_numpy(option_prices),
            "pdv_iv": to_numpy(implied_vol),
            "pricing_seconds": pricing_seconds,
        })

        all_rows.append(df)

        summary = summarize_slice(df)
        summary["pricing_seconds"] = pricing_seconds
        summaries.append(summary)

    result = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    csv_path = outdir / "pdv_market_grid_multi_maturity.csv"
    summary_csv_path = outdir / "pdv_market_grid_multi_maturity_summary.csv"
    summary_json_path = outdir / "pdv_market_grid_multi_maturity_summary.json"
    plot_path = outdir / "pdv_market_grid_multi_maturity.png"

    result.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    metadata = {
        "asofdate": str(asofdate.date()),
        "seed": seed,
        "N": N,
        "vix_N": vix_N,
        "timestep_per_day": timestep_per_day,
        "simulation_maturity": maturity,
        "target_dtes": target_dtes,
        "log_moneyness_min": float(log_moneyness.min()),
        "log_moneyness_max": float(log_moneyness.max()),
        "grid_points_per_maturity": len(log_moneyness),
        "simulate_seconds": simulate_seconds,
        "total_pricing_seconds": float(summary_df["pricing_seconds"].sum()),
        "total_seconds": float(simulate_seconds + summary_df["pricing_seconds"].sum()),
        "lam1": to_numpy(lam1).tolist(),
        "lam2": to_numpy(lam2).tolist(),
        "betas": to_numpy(betas).tolist(),
        "theta1": theta1,
        "theta2": theta2,
        "parabolic": parabolic,
        "parabolic_offset": parabolic_offset,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "summary_by_dte": summaries,
    }

    with open(summary_json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    plt.figure()
    for dte, group in result.groupby("dte"):
        group = group.sort_values("log_moneyness")
        plt.plot(group["log_moneyness"], group["pdv_iv"], marker="o", label=f"{int(dte)} DTE")

    plt.xlabel("log(K / F)")
    plt.ylabel("PDV implied volatility")
    plt.title("PDV Market-Aligned Multi-Maturity IV Smiles")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    print("Done.")
    print("Wrote:", csv_path)
    print("Wrote:", summary_csv_path)
    print("Wrote:", summary_json_path)
    print("Wrote:", plot_path)
    print()
    print("Summary by DTE:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print(summary_df)
    print()
    print("Timing:")
    print(json.dumps({
        "simulate_seconds": simulate_seconds,
        "total_pricing_seconds": float(summary_df["pricing_seconds"].sum()),
        "total_seconds": float(simulate_seconds + summary_df["pricing_seconds"].sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
