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
    spx = yf.Ticker("^GSPC").history(
        start=load_from,
        end=asofdate + timedelta(days=1),
    )["Close"]
    spx.index = pd.to_datetime(spx.index.date)
    return spx


def load_market_targets():
    path = Path("/users/4/trest017/urop_pdv/data/processed/thetadata/spxw_2021-06-02_155500_market_targets.json")
    with open(path) as f:
        targets = json.load(f)

    market = pd.DataFrame(targets["targets_by_dte"])
    return market


def summarize_slice(df):
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


def run_case(case_name, betas_values, spx, R_init1, R_init2, target_dtes):
    seed = 123456
    torch.manual_seed(seed)
    np.random.seed(seed)

    lam1 = torch.tensor([55, 10])
    lam2 = torch.tensor([20, 3])
    betas = torch.tensor(betas_values)

    theta1 = 0.25
    theta2 = 0.5

    N = 10000
    vix_N = 1000
    maturity = 0.30

    log_moneyness = np.linspace(-0.06, 0.06, 49)
    strikes = np.exp(log_moneyness)

    print()
    print("Running candidate:", case_name)
    print("betas:", betas_values)

    model = TorchMonteCarloExponentialModel(
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
        parabolic=0,
        parabolic_offset=0,
        device="cpu",
    )

    t0 = time.perf_counter()
    model.simulate()
    simulate_seconds = time.perf_counter() - t0

    all_rows = []
    summaries = []

    for dte in target_dtes:
        print("Computing DTE:", dte)
        option_maturity = dte / 365

        t1 = time.perf_counter()
        future, _, implied_vol, option_prices = model.compute_implied_vol(
            strikes=strikes,
            option_maturity=option_maturity,
        )
        pricing_seconds = time.perf_counter() - t1

        df = pd.DataFrame({
            "case_name": case_name,
            "beta0": betas_values[0],
            "beta1": betas_values[1],
            "beta2": betas_values[2],
            "dte": dte,
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

        summary = {
            "case_name": case_name,
            "beta0": betas_values[0],
            "beta1": betas_values[1],
            "beta2": betas_values[2],
            "dte": dte,
            "future": float(future),
            "simulate_seconds": simulate_seconds,
            "pricing_seconds": pricing_seconds,
        }
        summary.update(summarize_slice(df))

        all_rows.append(df)
        summaries.append(summary)

    return pd.concat(all_rows, ignore_index=True), pd.DataFrame(summaries)


def add_errors(summary, market):
    merged = pd.merge(
        summary,
        market,
        on="dte",
        how="inner",
        suffixes=("_pdv", "_market"),
    )

    merged["atm_iv_error_pdv_minus_market"] = merged["pdv_atm_iv"] - merged["atm_iv"]
    merged["left_5pct_iv_error_pdv_minus_market"] = merged["pdv_left_5pct_iv"] - merged["left_5pct_iv"]
    merged["right_5pct_iv_error_pdv_minus_market"] = merged["pdv_right_5pct_iv"] - merged["right_5pct_iv"]
    merged["wing_gap_error_pdv_minus_market"] = (
        merged["pdv_wing_iv_gap_left_minus_right"]
        - merged["wing_iv_gap_left_minus_right"]
    )
    merged["skew_slope_error_pdv_minus_market"] = (
        merged["pdv_atm_skew_slope"]
        - merged["atm_skew_slope"]
    )

    merged["objective_no_slope"] = (
        abs(merged["atm_iv_error_pdv_minus_market"])
        + abs(merged["left_5pct_iv_error_pdv_minus_market"])
        + 2.0 * abs(merged["right_5pct_iv_error_pdv_minus_market"])
        + abs(merged["wing_gap_error_pdv_minus_market"])
    )

    return merged


def aggregate_errors(comparison):
    rows = []

    for case_name, g in comparison.groupby("case_name"):
        rows.append({
            "case_name": case_name,
            "beta0": g["beta0"].iloc[0],
            "beta1": g["beta1"].iloc[0],
            "beta2": g["beta2"].iloc[0],
            "mean_objective_no_slope": g["objective_no_slope"].mean(),
            "mean_abs_atm_iv_error": g["atm_iv_error_pdv_minus_market"].abs().mean(),
            "mean_abs_left_5pct_iv_error": g["left_5pct_iv_error_pdv_minus_market"].abs().mean(),
            "mean_abs_right_5pct_iv_error": g["right_5pct_iv_error_pdv_minus_market"].abs().mean(),
            "mean_abs_wing_gap_error": g["wing_gap_error_pdv_minus_market"].abs().mean(),
            "mean_abs_skew_slope_error": g["skew_slope_error_pdv_minus_market"].abs().mean(),
            "max_abs_right_5pct_iv_error": g["right_5pct_iv_error_pdv_minus_market"].abs().max(),
            "max_abs_atm_iv_error": g["atm_iv_error_pdv_minus_market"].abs().max(),
        })

    return pd.DataFrame(rows).sort_values("mean_objective_no_slope")


def main():
    outdir = Path("/users/4/trest017/urop_pdv/benchmarks/pdv_candidate_betas")
    outdir.mkdir(parents=True, exist_ok=True)

    asofdate = pd.to_datetime("2021-06-02")
    spx = load_spx_history(asofdate)

    lam1 = torch.tensor([55, 10])
    lam2 = torch.tensor([20, 3])
    R_init1 = initialize_R(lam1, past_prices=spx, transform=identity)
    R_init2 = initialize_R(lam2, past_prices=spx, transform=squared)

    market = load_market_targets()
    target_dtes = [7, 14, 21, 30, 44, 90]

    candidates = [
        ("baseline", [0.04, -0.13, 0.65]),
        ("best_30d_beta1_m009_beta2_075", [0.04, -0.09, 0.75]),
        ("slope_balanced_beta1_m010_beta2_075", [0.04, -0.10, 0.75]),
    ]

    all_rows = []
    all_summaries = []

    for case_name, betas_values in candidates:
        rows, summary = run_case(case_name, betas_values, spx, R_init1, R_init2, target_dtes)
        all_rows.append(rows)
        all_summaries.append(summary)

    rows = pd.concat(all_rows, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)
    comparison = add_errors(summary, market)
    aggregate = aggregate_errors(comparison)

    rows_path = outdir / "pdv_candidate_betas_multi_maturity_rows.csv"
    summary_path = outdir / "pdv_candidate_betas_multi_maturity_summary.csv"
    comparison_path = outdir / "pdv_candidate_betas_multi_maturity_vs_market.csv"
    aggregate_path = outdir / "pdv_candidate_betas_multi_maturity_aggregate.csv"

    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    print()
    print("Aggregate candidate ranking:")
    print(aggregate)

    print()
    print("Per-maturity comparison:")
    keep_cols = [
        "case_name",
        "dte",
        "pdv_atm_iv",
        "atm_iv",
        "atm_iv_error_pdv_minus_market",
        "pdv_left_5pct_iv",
        "left_5pct_iv",
        "left_5pct_iv_error_pdv_minus_market",
        "pdv_right_5pct_iv",
        "right_5pct_iv",
        "right_5pct_iv_error_pdv_minus_market",
        "pdv_wing_iv_gap_left_minus_right",
        "wing_iv_gap_left_minus_right",
        "wing_gap_error_pdv_minus_market",
        "pdv_atm_skew_slope",
        "atm_skew_slope",
        "skew_slope_error_pdv_minus_market",
        "objective_no_slope",
    ]
    print(comparison[keep_cols].sort_values(["case_name", "dte"]))

    print()
    print("Wrote:", rows_path)
    print("Wrote:", summary_path)
    print("Wrote:", comparison_path)
    print("Wrote:", aggregate_path)


if __name__ == "__main__":
    main()
