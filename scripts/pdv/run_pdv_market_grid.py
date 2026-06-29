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

    # Keep model simulation horizon longer than the option maturity.
    maturity = 0.25

    # Market-aligned comparison maturity.
    option_maturity = 30 / 365

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

    print("Computing market-grid SPX implied vols...")
    t1 = time.perf_counter()
    future, _, implied_vol, option_prices = torch_mc.compute_implied_vol(
        strikes=strikes,
        option_maturity=option_maturity,
    )
    pricing_seconds = time.perf_counter() - t1

    result = pd.DataFrame({
        "dte": int(round(option_maturity * 365)),
        "option_maturity": option_maturity,
        "strike": to_numpy(strikes),
        "moneyness": to_numpy(strikes),
        "log_moneyness": log_moneyness,
        "pdv_option_price": to_numpy(option_prices),
        "pdv_iv": to_numpy(implied_vol),
    })

    csv_path = outdir / "pdv_market_grid_30d.csv"
    result.to_csv(csv_path, index=False)

    atm_row = result.iloc[(result["log_moneyness"] - 0.0).abs().argsort()].iloc[0]
    left_row = result.iloc[(result["log_moneyness"] + 0.05).abs().argsort()].iloc[0]
    right_row = result.iloc[(result["log_moneyness"] - 0.05).abs().argsort()].iloc[0]

    near_atm = result[result["log_moneyness"].abs() <= 0.03].copy()
    skew_slope, skew_intercept = np.polyfit(
        near_atm["log_moneyness"].to_numpy(),
        near_atm["pdv_iv"].to_numpy(),
        1,
    )

    summary = {
        "asofdate": str(asofdate.date()),
        "seed": seed,
        "N": N,
        "vix_N": vix_N,
        "timestep_per_day": timestep_per_day,
        "simulation_maturity": maturity,
        "option_maturity": option_maturity,
        "dte": int(round(option_maturity * 365)),
        "log_moneyness_min": float(log_moneyness.min()),
        "log_moneyness_max": float(log_moneyness.max()),
        "grid_points": len(result),
        "future": float(future),
        "atm_iv": float(atm_row["pdv_iv"]),
        "left_5pct_iv": float(left_row["pdv_iv"]),
        "right_5pct_iv": float(right_row["pdv_iv"]),
        "wing_iv_gap_left_minus_right": float(left_row["pdv_iv"] - right_row["pdv_iv"]),
        "atm_skew_slope": float(skew_slope),
        "atm_skew_intercept": float(skew_intercept),
        "simulate_seconds": simulate_seconds,
        "pricing_seconds": pricing_seconds,
        "total_seconds": simulate_seconds + pricing_seconds,
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
    }

    summary_path = outdir / "pdv_market_grid_30d_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot_path = outdir / "pdv_market_grid_30d.png"

    plt.figure()
    plt.plot(result["log_moneyness"], result["pdv_iv"], marker="o")
    plt.xlabel("log(K / F)")
    plt.ylabel("PDV implied volatility")
    plt.title("PDV Market-Aligned 30D IV Smile")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    print("Done.")
    print("Wrote:", csv_path)
    print("Wrote:", summary_path)
    print("Wrote:", plot_path)
    print()
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
