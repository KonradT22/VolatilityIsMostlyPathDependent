import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from calibration.torch_montecarlo import (
    TorchMonteCarloExponentialModel,
    identity,
    initialize_R,
    squared,
)


TARGET_DTES = [7, 14, 21, 30, 44, 90]
LOG_MONEYNESS = np.linspace(-0.04, 0.04, 17)
STRIKES = np.exp(LOG_MONEYNESS)

PARAMETER_RANGES = {
    "beta0": [0.02, 0.06],
    "beta1": [-0.16, -0.06],
    "beta2": [0.55, 0.85],
    "theta1": [0.15, 0.35],
    "theta2": [0.35, 0.65],
}


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_spx_history(asofdate):
    cached = Path(
        "/users/4/trest017/urop_pdv/benchmarks/"
        "msi_baseline/spx_history_yfinance.csv"
    )

    if cached.exists():
        frame = pd.read_csv(cached)
        date_col = frame.columns[0]
        frame[date_col] = pd.to_datetime(frame[date_col])

        series = pd.Series(
            frame["Close"].to_numpy(),
            index=frame[date_col],
        )
        series.index = pd.to_datetime(series.index.date)
        return series

    load_from = asofdate - timedelta(days=4 * 365)

    series = yf.Ticker("^GSPC").history(
        start=load_from,
        end=asofdate + timedelta(days=1),
    )["Close"]

    series.index = pd.to_datetime(series.index.date)
    return series


def sample_parameters(rng, base_r1, base_r2):
    beta0 = rng.uniform(*PARAMETER_RANGES["beta0"])
    beta1 = rng.uniform(*PARAMETER_RANGES["beta1"])
    beta2 = rng.uniform(*PARAMETER_RANGES["beta2"])
    theta1 = rng.uniform(*PARAMETER_RANGES["theta1"])
    theta2 = rng.uniform(*PARAMETER_RANGES["theta2"])

    r1_scale = np.maximum(np.abs(base_r1) * 0.20, 0.01)
    sampled_r1 = base_r1 + rng.normal(0.0, r1_scale)

    sampled_r2 = base_r2 * np.exp(rng.normal(0.0, 0.20, size=2))
    sampled_r2 = np.maximum(sampled_r2, 1e-8)

    return {
        "beta0": float(beta0),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "theta1": float(theta1),
        "theta2": float(theta2),
        "r1_0": float(sampled_r1[0]),
        "r1_1": float(sampled_r1[1]),
        "r2_0": float(sampled_r2[0]),
        "r2_1": float(sampled_r2[1]),
    }


def generate_surface(parameters, seed_root, n_paths):
    model = TorchMonteCarloExponentialModel(
        lam1=torch.tensor([55.0, 10.0]),
        lam2=torch.tensor([20.0, 3.0]),
        R_init1=torch.tensor([
            parameters["r1_0"],
            parameters["r1_1"],
        ]),
        R_init2=torch.tensor([
            parameters["r2_0"],
            parameters["r2_1"],
        ]),
        betas=torch.tensor([
            parameters["beta0"],
            parameters["beta1"],
            parameters["beta2"],
        ]),
        theta1=parameters["theta1"],
        theta2=parameters["theta2"],
        maturity=0.30,
        timestep_per_day=5,
        N=n_paths,
        vix_N=1000,
        fixed_seed=True,
        seed_root=int(seed_root),
        parabolic=0.0,
        parabolic_offset=0.0,
        device="cpu",
    )

    start = time.perf_counter()
    model.simulate()
    simulate_seconds = time.perf_counter() - start

    row = dict(parameters)
    row["seed_root"] = int(seed_root)
    row["simulate_seconds"] = float(simulate_seconds)

    all_ivs = []

    for dte in TARGET_DTES:
        maturity = dte / 365.0

        pricing_start = time.perf_counter()

        future, _, implied_vol, option_prices = model.compute_implied_vol(
            strikes=STRIKES,
            option_maturity=maturity,
        )

        pricing_seconds = time.perf_counter() - pricing_start

        iv = to_numpy(implied_vol).astype(float)
        prices = to_numpy(option_prices).astype(float)

        if iv.shape != (len(LOG_MONEYNESS),):
            raise ValueError(f"Unexpected IV shape for {dte} DTE: {iv.shape}")

        if not np.all(np.isfinite(iv)):
            raise ValueError(f"Non-finite IV values at {dte} DTE")

        if not np.all(np.isfinite(prices)):
            raise ValueError(f"Non-finite option prices at {dte} DTE")

        if np.any(iv <= 0.005) or np.any(iv >= 1.50):
            raise ValueError(
                f"IV outside accepted range at {dte} DTE: "
                f"min={iv.min():.6f}, max={iv.max():.6f}"
            )

        row[f"future_dte_{dte}"] = float(future)
        row[f"pricing_seconds_dte_{dte}"] = float(pricing_seconds)

        for index, value in enumerate(iv):
            row[f"iv_dte_{dte}_m{index:02d}"] = float(value)

        all_ivs.extend(iv.tolist())

    row["surface_iv_min"] = float(np.min(all_ivs))
    row["surface_iv_max"] = float(np.max(all_ivs))
    row["surface_iv_mean"] = float(np.mean(all_ivs))

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--samples-per-shard", type=int, default=64)
    parser.add_argument("--n-paths", type=int, default=4000)
    parser.add_argument("--base-seed", type=int, default=20260728)

    parser.add_argument(
        "--out-dir",
        default=(
            "/users/4/trest017/urop_pdv/"
            "data/processed/surrogate/pilot_surface_dataset"
        ),
    )

    args = parser.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng_seed = args.base_seed + args.shard_id
    rng = np.random.default_rng(rng_seed)

    asofdate = pd.Timestamp("2021-06-02")
    spx = load_spx_history(asofdate)

    lam1 = torch.tensor([55.0, 10.0])
    lam2 = torch.tensor([20.0, 3.0])

    base_r1 = to_numpy(
        initialize_R(lam1, past_prices=spx, transform=identity)
    ).astype(float)

    base_r2 = to_numpy(
        initialize_R(lam2, past_prices=spx, transform=squared)
    ).astype(float)

    print("Shard:", args.shard_id)
    print("Samples requested:", args.samples_per_shard)
    print("N paths:", args.n_paths)
    print("Base R1:", base_r1)
    print("Base R2:", base_r2)

    rows = []
    rejected = 0
    attempts = 0
    max_attempts = args.samples_per_shard * 10

    shard_start = time.perf_counter()

    while len(rows) < args.samples_per_shard:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Too many rejected samples: accepted={len(rows)}, "
                f"rejected={rejected}"
            )

        parameters = sample_parameters(rng, base_r1, base_r2)

        case_number = args.shard_id * args.samples_per_shard + len(rows)
        seed_root = (
            args.base_seed * 1_000_000
            + args.shard_id * 100_000
            + attempts
        )

        try:
            row = generate_surface(
                parameters=parameters,
                seed_root=seed_root,
                n_paths=args.n_paths,
            )

            row["case_id"] = int(case_number)
            row["shard_id"] = int(args.shard_id)
            row["accepted_index"] = int(len(rows))
            row["attempt_number"] = int(attempts)

            rows.append(row)

            if len(rows) % 8 == 0:
                print(
                    f"Accepted {len(rows)}/{args.samples_per_shard}; "
                    f"rejected={rejected}"
                )

        except Exception as error:
            rejected += 1
            print(
                f"Rejected attempt {attempts}: "
                f"{type(error).__name__}: {error}"
            )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("case_id")

    csv_path = outdir / f"pdv_surface_shard_{args.shard_id:02d}.csv"
    metadata_path = outdir / f"pdv_surface_shard_{args.shard_id:02d}.json"

    frame.to_csv(csv_path, index=False)

    elapsed = time.perf_counter() - shard_start

    metadata = {
        "shard_id": args.shard_id,
        "accepted_samples": len(frame),
        "rejected_attempts": rejected,
        "total_attempts": attempts,
        "n_paths": args.n_paths,
        "base_seed": args.base_seed,
        "rng_seed": rng_seed,
        "target_dtes": TARGET_DTES,
        "log_moneyness": LOG_MONEYNESS.tolist(),
        "strikes": STRIKES.tolist(),
        "surface_output_size": len(TARGET_DTES) * len(LOG_MONEYNESS),
        "parameter_ranges": PARAMETER_RANGES,
        "base_r1": base_r1.tolist(),
        "base_r2": base_r2.tolist(),
        "elapsed_seconds": elapsed,
        "output_csv": str(csv_path),
    }

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2)

    print()
    print("Done.")
    print("Rows:", len(frame))
    print("Columns:", len(frame.columns))
    print("Rejected:", rejected)
    print("Elapsed seconds:", elapsed)
    print("Wrote:", csv_path)
    print("Wrote:", metadata_path)


if __name__ == "__main__":
    main()
