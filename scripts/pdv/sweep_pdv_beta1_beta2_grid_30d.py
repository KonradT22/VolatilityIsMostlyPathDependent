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
        spx_df = pd.read_csv(cached)
        date_col = spx_df.columns[0]
        spx_df[date_col] = pd.to_datetime(spx_df[date_col])
        spx = pd.Series(spx_df["Close"].values, index=spx_df[date_col])
        spx.index = pd.to_datetime(spx.index.date)
        return spx

    load_from = asofdate - timedelta(days=4 * 365)
    spx = yf.Ticker("^GSPC").history(
        start=load_from,
        end=asofdate + timedelta(days=1),
    )["Close"]
    spx.index = pd.to_datetime(spx.index.date)
    return spx


def load_market_30d_target():
    path = Path("/users/4/trest017/urop_pdv/data/processed/thetadata/spxw_2021-06-02_155500_market_targets.json")

    with open(path) as f:
        targets = json.load(f)

    rows = targets["targets_by_dte"]
    row = min(rows, key=lambda x: abs(x["dte"] - 30))

    return row


def summarize_pdv_slice(df):
    atm = df.iloc[(df["log_moneyness"] - 0.0).abs().argsort()].iloc[0]
    left = df.iloc[(df["log_moneyness"] + 0.05).abs().argsort()].iloc[0]
    right = df.iloc[(df["log_moneyness"] - 0.05).abs().argsort()].iloc[0]

    near_atm = df[df["log_moneyness"].abs() <= 0.03].copy()
    slope, intercept = np.polyfit(
        near_atm["log_moneyness"].to_numpy(),
        near_atm["pdv_iv"].to_numpy(),
        1,
    )

    return {
        "pdv_atm_iv": float(atm["pdv_iv"]),
        "pdv_left_5pct_iv": float(left["pdv_iv"]),
        "pdv_right_5pct_iv": float(right["pdv_iv"]),
        "pdv_wing_iv_gap_left_minus_right": float(left["pdv_iv"] - right["pdv_iv"]),
        "pdv_atm_skew_slope": float(slope),
        "pdv_atm_skew_intercept": float(intercept),
        "pdv_min_iv": float(df["pdv_iv"].min()),
        "pdv_max_iv": float(df["pdv_iv"].max()),
    }


def run_case(case_name, betas_values, spx, R_init1, R_init2, market):
    seed = 123456
    torch.manual_seed(seed)
    np.random.seed(seed)

    lam1 = torch.tensor([55, 10])
    lam2 = torch.tensor([20, 3])
    betas = torch.tensor(betas_values)

    theta1 = 0.25
    theta2 = 0.5

    N = 10000
    maturity = 0.25
    option_dte = 30
    option_maturity = option_dte / 365

    log_moneyness = np.linspace(-0.06, 0.06, 49)
    strikes = np.exp(log_moneyness)

    print()
    print("Running case:", case_name)
    print("betas:", betas_values)

    t0 = time.perf_counter()

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

    t1 = time.perf_counter()
    torch_mc.simulate()
    simulate_seconds = time.perf_counter() - t1

    t2 = time.perf_counter()
    future, _, implied_vol, option_prices = torch_mc.compute_implied_vol(
        strikes=strikes,
        option_maturity=option_maturity,
    )
    pricing_seconds = time.perf_counter() - t2

    total_seconds = time.perf_counter() - t0

    df = pd.DataFrame({
        "case_name": case_name,
        "beta0": betas_values[0],
        "beta1": betas_values[1],
        "beta2": betas_values[2],
        "dte": option_dte,
        "future": float(future),
        "strike": strikes,
        "moneyness": strikes,
        "log_moneyness": log_moneyness,
        "pdv_option_price": to_numpy(option_prices),
        "pdv_iv": to_numpy(implied_vol),
    })

    summary = summarize_pdv_slice(df)

    row = {
        "case_name": case_name,
        "beta0": betas_values[0],
        "beta1": betas_values[1],
        "beta2": betas_values[2],
        "dte": option_dte,
        "future": float(future),
        "market_expiration": market["expiration"],
        "market_atm_iv": float(market["atm_iv"]),
        "market_left_5pct_iv": float(market["left_5pct_iv"]),
        "market_right_5pct_iv": float(market["right_5pct_iv"]),
        "market_wing_iv_gap_left_minus_right": float(market["wing_iv_gap_left_minus_right"]),
        "market_atm_skew_slope": float(market["atm_skew_slope"]),
        "simulate_seconds": simulate_seconds,
        "pricing_seconds": pricing_seconds,
        "total_seconds": total_seconds,
    }

    row.update(summary)

    row["atm_iv_error_pdv_minus_market"] = row["pdv_atm_iv"] - row["market_atm_iv"]
    row["left_5pct_iv_error_pdv_minus_market"] = row["pdv_left_5pct_iv"] - row["market_left_5pct_iv"]
    row["right_5pct_iv_error_pdv_minus_market"] = row["pdv_right_5pct_iv"] - row["market_right_5pct_iv"]
    row["wing_gap_error_pdv_minus_market"] = (
        row["pdv_wing_iv_gap_left_minus_right"]
        - row["market_wing_iv_gap_left_minus_right"]
    )
    row["skew_slope_error_pdv_minus_market"] = (
        row["pdv_atm_skew_slope"]
        - row["market_atm_skew_slope"]
    )

    # Smaller score is better. Put extra weight on right wing because that is the main miss.
    row["objective_score"] = (
        abs(row["atm_iv_error_pdv_minus_market"])
        + abs(row["left_5pct_iv_error_pdv_minus_market"])
        + 2.0 * abs(row["right_5pct_iv_error_pdv_minus_market"])
        + abs(row["wing_gap_error_pdv_minus_market"])
    )

    return df, row


def main():
    outdir = Path("/users/4/trest017/urop_pdv/benchmarks/pdv_sweeps")
    outdir.mkdir(parents=True, exist_ok=True)

    asofdate = pd.to_datetime("2021-06-02")
    spx = load_spx_history(asofdate)

    lam1 = torch.tensor([55, 10])
    lam2 = torch.tensor([20, 3])
    R_init1 = initialize_R(lam1, past_prices=spx, transform=identity)
    R_init2 = initialize_R(lam2, past_prices=spx, transform=squared)

    market = load_market_30d_target()

    base = [0.04, -0.13, 0.65]

    cases = []

    beta1_values = [-0.10, -0.09, -0.08, -0.07, -0.06]
    beta2_values = [0.65, 0.70, 0.75, 0.80]

    for beta1 in beta1_values:
        for beta2 in beta2_values:
            cases.append((
                f"beta1_{beta1:+.2f}_beta2_{beta2:.2f}",
                [base[0], beta1, beta2],
            ))

    all_rows = []
    summary_rows = []

    for case_name, betas_values in cases:
        df, row = run_case(case_name, betas_values, spx, R_init1, R_init2, market)
        all_rows.append(df)
        summary_rows.append(row)

    result = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("objective_score")

    rows_path = outdir / "pdv_beta1_beta2_grid_30d_rows.csv"
    summary_path = outdir / "pdv_beta1_beta2_grid_30d_summary.csv"

    result.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    print()
    print("Market 30D target:")
    print(json.dumps(market, indent=2))

    print()
    print("2D beta1-beta2 grid summary sorted by objective_score:")
    print(summary)

    print()
    print("Wrote:", rows_path)
    print("Wrote:", summary_path)


if __name__ == "__main__":
    main()
