import os
import json
import time
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from datetime import timedelta

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

def main():
    outdir = Path("/users/4/trest017/urop_pdv/benchmarks/msi_baseline")
    outdir.mkdir(parents=True, exist_ok=True)

    seed = 123456
    torch.manual_seed(seed)
    np.random.seed(seed)

    asofdate = pd.to_datetime("2021-06-02")
    load_from = asofdate - timedelta(days=4 * 365)

    print("Loading SPX history from yfinance...")
    spx = yf.Ticker("^GSPC").history(
        start=load_from,
        end=asofdate + timedelta(days=1),
    )["Close"]

    spx.index = pd.to_datetime(spx.index.date)

    spx.to_csv(outdir / "spx_history_yfinance.csv", header=["Close"])

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
    maturity = 0.25

    print("Constructing model...")
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

    option_maturity = 1 / 12
    strikes = np.arange(0.9, 1.05, 0.01)

    print("Computing SPX implied vols and option prices...")
    t1 = time.perf_counter()
    future, _, implied_vol, option_prices = torch_mc.compute_implied_vol(
        strikes=strikes,
        option_maturity=option_maturity,
    )
    spx_seconds = time.perf_counter() - t1

    spx_benchmark = pd.DataFrame({
        "strike": to_numpy(strikes),
        "option_price": to_numpy(option_prices),
        "implied_vol": to_numpy(implied_vol),
    })

    spx_benchmark.to_csv(outdir / "spx_option_benchmark.csv", index=False)

    print("Computing VIX implied vols and option prices...")
    t2 = time.perf_counter()
    vix_future, vix_strikes, vix_implied_vol, vix_option_price = torch_mc.compute_vix_implied_vol(
        vix_maturity=option_maturity,
        strikes=None,
    )
    vix_seconds = time.perf_counter() - t2

    vix_benchmark = pd.DataFrame({
        "strike": to_numpy(vix_strikes),
        "vix_option_price": to_numpy(vix_option_price),
        "vix_implied_vol": to_numpy(vix_implied_vol),
    })

    vix_benchmark.to_csv(outdir / "vix_option_benchmark.csv", index=False)

    metadata = {
        "asofdate": str(asofdate.date()),
        "load_from": str(load_from.date()),
        "seed": seed,
        "N": N,
        "vix_N": vix_N,
        "timestep_per_day": timestep_per_day,
        "maturity": maturity,
        "option_maturity": option_maturity,
        "lam1": to_numpy(lam1).tolist(),
        "lam2": to_numpy(lam2).tolist(),
        "betas": to_numpy(betas).tolist(),
        "theta1": theta1,
        "theta2": theta2,
        "parabolic": parabolic,
        "parabolic_offset": parabolic_offset,
        "R_init1": to_numpy(R_init1).tolist(),
        "R_init2": to_numpy(R_init2).tolist(),
        "future": float(future),
        "vix_future": float(vix_future),
        "simulate_seconds": simulate_seconds,
        "spx_pricing_seconds": spx_seconds,
        "vix_pricing_seconds": vix_seconds,
        "total_seconds": simulate_seconds + spx_seconds + vix_seconds,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }

    with open(outdir / "baseline_parameters.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")
    print(f"Wrote outputs to: {outdir}")
    print(f"simulate_seconds={simulate_seconds:.3f}")
    print(f"spx_pricing_seconds={spx_seconds:.3f}")
    print(f"vix_pricing_seconds={vix_seconds:.3f}")
    print(f"total_seconds={metadata['total_seconds']:.3f}")

if __name__ == "__main__":
    main()
