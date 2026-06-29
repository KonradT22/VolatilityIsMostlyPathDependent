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


def load_spx_history(asofdate):
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
    print("Loading SPX history from yfinance...")
    spx = yf.Ticker("^GSPC").history(
        start=load_from,
        end=asofdate + timedelta(days=1),
    )["Close"]
    spx.index = pd.to_datetime(spx.index.date)
    return spx


def run_case(seed, N, option_dte, log_moneyness_min, log_moneyness_max, grid_points):
    torch.manual_seed(seed)
    np.random.seed(seed)

    asofdate = pd.to_datetime("2021-06-02")
    spx = load_spx_history(asofdate)

    lam1 = torch.tensor([55, 10])
    lam2 = torch.tensor([20, 3])
    betas = torch.tensor([0.04, -0.13, 0.65])
    theta1 = 0.25
    theta2 = 0.5

    R_init1 = initialize_R(lam1, past_prices=spx, transform=identity)
    R_init2 = initialize_R(lam2, past_prices=spx, transform=squared)

    maturity = 0.25
    option_maturity = option_dte / 365

    log_moneyness = np.linspace(log_moneyness_min, log_moneyness_max, grid_points)
    strikes = np.exp(log_moneyness)

    print()
    print("Running case:")
    print("seed:", seed)
    print("N:", N)
    print("option_dte:", option_dte)
    print("log_moneyness:", log_moneyness_min, "to", log_moneyness_max)
    print("grid_points:", grid_points)

    torch_mc = TorchMonteCarloExponentialModel(
        lam1=lam1,
        lam2=lam2,
        betas=betas,
        R_init1=R_init1,
        R_init2=R_init2,
        theta1=theta1,
        theta2=theta2,
        N=N,
        vix_N=1000,
        maturity=maturity,
        parabolic=0,
        parabolic_offset=0,
        device="cpu",
    )

    t0 = time.perf_counter()
    torch_mc.simulate()
    simulate_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    future, _, implied_vol, option_prices = torch_mc.compute_implied_vol(
        strikes=strikes,
        option_maturity=option_maturity,
    )
    pricing_seconds = time.perf_counter() - t1

    df = pd.DataFrame({
        "seed": seed,
        "N": N,
        "dte": option_dte,
        "option_maturity": option_maturity,
        "future": float(future),
        "strike": strikes,
        "moneyness": strikes,
        "log_moneyness": log_moneyness,
        "pdv_option_price": to_numpy(option_prices),
        "pdv_iv": to_numpy(implied_vol),
        "simulate_seconds": simulate_seconds,
        "pricing_seconds": pricing_seconds,
    })

    atm = df.iloc[(df["log_moneyness"] - 0.0).abs().argsort()].iloc[0]
    left = df.iloc[(df["log_moneyness"] + 0.05).abs().argsort()].iloc[0]
    right = df.iloc[(df["log_moneyness"] - 0.05).abs().argsort()].iloc[0]

    near_atm = df[df["log_moneyness"].abs() <= 0.03].copy()
    slope, intercept = np.polyfit(
        near_atm["log_moneyness"].to_numpy(),
        near_atm["pdv_iv"].to_numpy(),
        1,
    )

    summary = {
        "seed": seed,
        "N": N,
        "dte": option_dte,
        "log_moneyness_min": log_moneyness_min,
        "log_moneyness_max": log_moneyness_max,
        "grid_points": grid_points,
        "future": float(future),
        "atm_iv": float(atm["pdv_iv"]),
        "left_5pct_iv": float(left["pdv_iv"]),
        "right_5pct_iv": float(right["pdv_iv"]),
        "wing_iv_gap_left_minus_right": float(left["pdv_iv"] - right["pdv_iv"]),
        "atm_skew_slope": float(slope),
        "atm_skew_intercept": float(intercept),
        "min_iv": float(df["pdv_iv"].min()),
        "max_iv": float(df["pdv_iv"].max()),
        "simulate_seconds": simulate_seconds,
        "pricing_seconds": pricing_seconds,
        "total_seconds": simulate_seconds + pricing_seconds,
    }

    return df, summary


def main():
    outdir = Path("/users/4/trest017/urop_pdv/benchmarks/pdv_stability")
    outdir.mkdir(parents=True, exist_ok=True)

    cases = [
        {"seed": 123456, "N": 10000, "option_dte": 30, "log_moneyness_min": -0.06, "log_moneyness_max": 0.06, "grid_points": 49},
        {"seed": 777777, "N": 10000, "option_dte": 30, "log_moneyness_min": -0.06, "log_moneyness_max": 0.06, "grid_points": 49},
        {"seed": 123456, "N": 20000, "option_dte": 30, "log_moneyness_min": -0.06, "log_moneyness_max": 0.06, "grid_points": 49},
        {"seed": 123456, "N": 10000, "option_dte": 30, "log_moneyness_min": -0.04, "log_moneyness_max": 0.04, "grid_points": 33},
    ]

    all_rows = []
    summaries = []

    for case in cases:
        df, summary = run_case(**case)
        all_rows.append(df)
        summaries.append(summary)

    result = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    result_path = outdir / "pdv_right_wing_stability_rows.csv"
    summary_path = outdir / "pdv_right_wing_stability_summary.csv"
    json_path = outdir / "pdv_right_wing_stability_summary.json"

    result.to_csv(result_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print()
    print("Stability summary:")
    print(summary_df)
    print()
    print("Wrote:", result_path)
    print("Wrote:", summary_path)
    print("Wrote:", json_path)


if __name__ == "__main__":
    main()
