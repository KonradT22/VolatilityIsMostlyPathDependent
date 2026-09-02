"""Price each anchor-jitter candidate's full 77-coordinate option grid at 10,000
Monte Carlo paths, producing the inverse-learning dataset consumed by the ANN
training pipeline. Run as SLURM array chunks (see generate_pdv_inverse_dataset.slurm);
candidates with nonfinite/nonpositive volatility or option prices are rejected and
logged rather than kept. Also defines the canonical 77-coordinate grid used
throughout the surrogate/repricing pipeline."""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.calibration.calibrate_pdv_market_5param_fixed_grid import (
    BATCH_SIZE,
    build_model,
    to_numpy,
)


PRIOR_DIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_market_5param_fixed_grid/"
    "robust_historical_prior"
)

INPUT_PATH = (
    PRIOR_DIR
    / "anchor_jitter_samples.csv"
)

PILOT_SUMMARY_PATH = (
    PRIOR_DIR
    / "anchor_jitter_validity_pilot"
    / "pilot_candidate_summary.csv"
)

DEFAULT_OUTDIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "inverse_ann/pdv_anchor_jitter_10k"
)

PARAMS = [
    "beta0",
    "beta1",
    "beta2",
    "theta1",
    "theta2",
]

STATE_COLS = [
    "R1_fast",
    "R1_slow",
    "R2_fast",
    "R2_slow",
]

GRID_SPEC = {
    7: (-0.040, 0.010),
    14: (-0.050, 0.020),
    21: (-0.050, 0.025),
    30: (-0.040, 0.035),
    45: (-0.040, 0.040),
    60: (-0.040, 0.040),
    90: (-0.035, 0.040),
}

POINTS_PER_TENOR = 11
TIMESTEP_PER_DAY = 5

VOL_CAP = 1.5
CAP_TOL = 1e-7

# One-day cushion beyond longest canonical maturity.
SIMULATION_MATURITY = 91 / 365.0

EXPECTED_FAILURES = {
    "nonfinite_volatility",
    "nonpositive_volatility",
    "nonfinite_option_prices",
    "nonpositive_option_prices",
}


def canonical_grid():
    rows = []
    feature_index = 0

    for dte, (low, high) in GRID_SPEC.items():
        values = np.linspace(
            low,
            high,
            POINTS_PER_TENOR,
        )

        for grid_index, logm in enumerate(values):
            moneyness = float(np.exp(logm))

            if moneyness < 1.0 - 1e-12:
                side = "PUT"
            else:
                side = "CALL"

            column = (
                f"price_dte{dte:03d}_"
                f"grid{grid_index:02d}"
            )

            rows.append({
                "feature_index": feature_index,
                "column": column,
                "target_dte": dte,
                "grid_index": grid_index,
                "log_moneyness": float(logm),
                "moneyness": moneyness,
                "otm_right": side,
            })

            feature_index += 1

    result = pd.DataFrame(rows)

    if len(result) != 77:
        raise RuntimeError(
            f"Expected 77 coordinates, found {len(result)}"
        )

    return result


def known_invalid_sample_ids():
    if not PILOT_SUMMARY_PATH.exists():
        return set()

    pilot = pd.read_csv(
        PILOT_SUMMARY_PATH
    )

    valid = (
        pilot["all_seeds_valid"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    return set(
        pilot.loc[
            ~valid,
            "sample_id",
        ].astype(int)
    )


def evaluate_candidate(
    row,
    grid,
    n_paths,
    seed_root,
):
    params = np.array(
        [
            row.beta0,
            row.beta1,
            row.beta2,
            row.theta1,
            row.theta2,
        ],
        dtype=float,
    )

    r1 = torch.tensor(
        [
            row.R1_fast,
            row.R1_slow,
        ],
        dtype=torch.float64,
    )

    r2 = torch.tensor(
        [
            row.R2_fast,
            row.R2_slow,
        ],
        dtype=torch.float64,
    )

    model = build_model(
        params=params,
        R_init1=r1,
        R_init2=r2,
        simulation_maturity=SIMULATION_MATURITY,
        n_paths=n_paths,
        seed_root=seed_root,
    )

    model.simulate(
        save_R=False
    )

    vol = to_numpy(
        model.vol_array
    )

    if not np.all(
        np.isfinite(vol)
    ):
        raise ValueError(
            "nonfinite_volatility"
        )

    min_vol = float(
        np.min(vol)
    )

    max_vol = float(
        np.max(vol)
    )

    cap_fraction = float(
        np.mean(
            vol >= VOL_CAP - CAP_TOL
        )
    )

    if min_vol <= 0:
        raise ValueError(
            "nonpositive_volatility"
        )

    prices_by_column = {}

    for dte, g in grid.groupby(
        "target_dte",
        sort=True,
    ):
        maturity = (
            int(dte) / 365.0
        )

        index = int(
            torch.ceil(
                torch.tensor(
                    maturity,
                    dtype=model.timestep.dtype,
                    device=model.device,
                )
                / model.timestep
            )
        )

        if index >= len(
            model.S_array
        ):
            raise RuntimeError(
                f"Pricing index {index} "
                f"outside S_array length "
                f"{len(model.S_array)}"
            )

        model_future = float(
            model.S_array[index]
            .mean()
            .detach()
            .cpu()
        )

        if (
            not np.isfinite(model_future)
            or model_future <= 0
        ):
            raise ValueError(
                "invalid_model_future"
            )

        moneyness = (
            g["moneyness"]
            .to_numpy(dtype=float)
        )

        strikes = (
            model_future
            * moneyness
        )

        chunks = []

        for start in range(
            0,
            len(strikes),
            BATCH_SIZE,
        ):
            stop = min(
                start + BATCH_SIZE,
                len(strikes),
            )

            _, _, prices = (
                model.compute_option_price(
                    strikes=strikes[start:stop],
                    option_maturity=maturity,
                    return_future=True,
                    var_reduction=True,
                )
            )

            chunks.append(
                to_numpy(prices)
                .astype(float)
            )

        call_prices = np.concatenate(
            chunks
        )

        normalized_calls = (
            call_prices
            / model_future
        )

        normalized_puts = (
            normalized_calls
            - (1.0 - moneyness)
        )

        sides = (
            g["otm_right"]
            .astype(str)
            .to_numpy()
        )

        otm_prices = np.where(
            sides == "PUT",
            normalized_puts,
            normalized_calls,
        )

        if not np.all(
            np.isfinite(otm_prices)
        ):
            raise ValueError(
                "nonfinite_option_prices"
            )

        if np.any(
            otm_prices <= 0
        ):
            raise ValueError(
                "nonpositive_option_prices"
            )

        for column, price in zip(
            g["column"],
            otm_prices,
        ):
            prices_by_column[
                column
            ] = float(price)

    if len(prices_by_column) != 77:
        raise RuntimeError(
            f"Expected 77 generated prices, "
            f"found {len(prices_by_column)}"
        )

    return {
        "min_vol": min_vol,
        "max_vol": max_vol,
        "cap_fraction": cap_fraction,
        "prices": prices_by_column,
    }


def save_outputs(
    valid_rows,
    failure_rows,
    outdir,
    chunk_id,
):
    valid_path = (
        outdir
        / f"inverse_dataset_chunk_{chunk_id:02d}.csv"
    )

    failure_path = (
        outdir
        / f"inverse_dataset_failures_{chunk_id:02d}.csv"
    )

    pd.DataFrame(
        valid_rows
    ).to_csv(
        valid_path,
        index=False,
    )

    pd.DataFrame(
        failure_rows
    ).to_csv(
        failure_path,
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--count",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--n-paths",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--seed-base",
        type=int,
        default=202608190000,
    )

    parser.add_argument(
        "--chunk-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = pd.read_csv(
        INPUT_PATH
    )

    start = args.start_row

    stop = min(
        start + args.count,
        len(samples),
    )

    subset = (
        samples.iloc[
            start:stop
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(subset) == 0:
        raise RuntimeError(
            "Selected candidate subset is empty"
        )

    grid = canonical_grid()

    # Grid is identical across all array tasks.
    grid_path = (
        args.output_dir
        / "canonical_77_price_grid.csv"
    )

    temp_grid = (
        args.output_dir
        / f".canonical_grid_{os.getpid()}.tmp"
    )

    grid.to_csv(
        temp_grid,
        index=False,
    )

    os.replace(
        temp_grid,
        grid_path,
    )

    known_invalid = (
        known_invalid_sample_ids()
    )

    print("=" * 78)
    print("PDV INVERSE DATASET GENERATOR")
    print("=" * 78)
    print(
        "chunk:",
        args.chunk_id,
    )
    print(
        "rows:",
        f"{start}:{stop}",
    )
    print(
        "candidates:",
        len(subset),
    )
    print(
        "paths:",
        args.n_paths,
    )
    print(
        "known pilot-invalid sample IDs:",
        sorted(known_invalid),
    )
    print()

    valid_rows = []
    failure_rows = []

    total = len(subset)

    for i, row in enumerate(
        subset.itertuples(
            index=False
        ),
        start=1,
    ):
        sample_id = int(
            row.sample_id
        )

        seed_root = (
            args.seed_base
            + sample_id
        )

        if sample_id in known_invalid:
            failure_rows.append({
                "sample_id": sample_id,
                "anchor_date":
                    str(row.anchor_date),
                "seed_root": seed_root,
                "failure_reason":
                    "failed_independent_validity_pilot",
            })

            continue

        try:
            result = evaluate_candidate(
                row=row,
                grid=grid,
                n_paths=args.n_paths,
                seed_root=seed_root,
            )

        except ValueError as exc:
            reason = str(exc)

            if (
                reason
                not in EXPECTED_FAILURES
            ):
                raise

            failure_rows.append({
                "sample_id": sample_id,
                "anchor_date":
                    str(row.anchor_date),
                "seed_root": seed_root,
                "failure_reason": reason,
            })

            print(
                f"sample={sample_id} "
                f"INVALID "
                f"reason={reason}",
                flush=True,
            )

            continue

        output = {
            "sample_id":
                sample_id,
            "anchor_date":
                str(row.anchor_date),
            "seed_root":
                seed_root,
            "N_paths":
                args.n_paths,
            "beta0":
                float(row.beta0),
            "beta1":
                float(row.beta1),
            "beta2":
                float(row.beta2),
            "theta1":
                float(row.theta1),
            "theta2":
                float(row.theta2),
            "R1_fast":
                float(row.R1_fast),
            "R1_slow":
                float(row.R1_slow),
            "R2_fast":
                float(row.R2_fast),
            "R2_slow":
                float(row.R2_slow),
            "min_vol":
                result["min_vol"],
            "max_vol":
                result["max_vol"],
            "cap_fraction":
                result["cap_fraction"],
        }

        output.update(
            result["prices"]
        )

        valid_rows.append(
            output
        )

        if (
            i == 1
            or i % 25 == 0
        ):
            print(
                f"chunk={args.chunk_id:02d} "
                f"candidate={i:4d}/{total} "
                f"valid={len(valid_rows):4d} "
                f"failed={len(failure_rows):3d}",
                flush=True,
            )

        if i % 25 == 0:
            save_outputs(
                valid_rows,
                failure_rows,
                args.output_dir,
                args.chunk_id,
            )

    save_outputs(
        valid_rows,
        failure_rows,
        args.output_dir,
        args.chunk_id,
    )

    print()
    print("=" * 78)
    print("CHUNK COMPLETE")
    print("=" * 78)
    print(
        "valid:",
        len(valid_rows),
    )
    print(
        "failed:",
        len(failure_rows),
    )
    print(
        "output:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
