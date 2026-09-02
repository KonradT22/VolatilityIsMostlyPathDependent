"""Reprice held-out (bound-projected) ANN parameter predictions through the full
PDV Monte Carlo model on each scenario's original seed, and compare against the
true 77-coordinate surface. Reports RMSE/MAE in basis points of forward, per
scenario and by maturity. Run as chunked SLURM array tasks."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibration.generate_pdv_inverse_dataset import (
    EXPECTED_FAILURES,
    canonical_grid,
    evaluate_candidate,
)


DEFAULT_SPLIT_DIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "inverse_ann/pdv_anchor_jitter_10k/"
    "splits_interpolation"
)

DEFAULT_PREDICTIONS = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "inverse_ann/pdv_anchor_jitter_10k/"
    "models/mlp_interpolation/test_evaluation/"
    "test_predictions.csv"
)

DEFAULT_OUTDIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "inverse_ann/pdv_anchor_jitter_10k/"
    "models/mlp_interpolation/test_evaluation/"
    "repricing"
)

TARGETS = [
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )

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
        "--chunk-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--n-paths",
        type=int,
        default=10000,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    test = pd.read_csv(
        args.split_dir / "test.csv"
    )

    predictions = pd.read_csv(
        args.predictions
    )

    if len(test) != 959:
        raise RuntimeError(
            f"Expected 959 test rows, found {len(test)}"
        )

    if len(predictions) != 959:
        raise RuntimeError(
            f"Expected 959 predictions, found {len(predictions)}"
        )

    if test["sample_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate sample_id in test set"
        )

    if predictions["sample_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate sample_id in prediction set"
        )

    prediction_cols = (
        ["sample_id"]
        + [
            f"clipped_{x}"
            for x in TARGETS
        ]
    )

    merged = test.merge(
        predictions[prediction_cols],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 959:
        raise RuntimeError(
            f"Merge produced {len(merged)} rows"
        )

    merged = (
        merged.sort_values("sample_id")
        .reset_index(drop=True)
    )

    start = args.start_row
    stop = min(
        start + args.count,
        len(merged),
    )

    subset = (
        merged.iloc[start:stop]
        .copy()
        .reset_index(drop=True)
    )

    if len(subset) == 0:
        raise RuntimeError(
            "Selected subset is empty"
        )

    grid = canonical_grid()

    price_cols = (
        grid["column"]
        .astype(str)
        .tolist()
    )

    if len(price_cols) != 77:
        raise RuntimeError(
            f"Expected 77 price columns, "
            f"found {len(price_cols)}"
        )

    missing = [
        c for c in price_cols
        if c not in merged.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing true price columns: {missing[:5]}"
        )

    print("=" * 88)
    print("PDV INVERSE ANN TEST REPRICING")
    print("=" * 88)
    print("chunk:", args.chunk_id)
    print("rows:", f"{start}:{stop}")
    print("scenarios:", len(subset))
    print("paths:", args.n_paths)
    print("pricing coordinates:", len(price_cols))
    print("seed mode: original per-scenario seed_root / common random numbers")
    print()

    scenario_rows = []
    residual_rows = []
    failure_rows = []

    for i, row in enumerate(
        subset.itertuples(index=False),
        start=1,
    ):
        sample_id = int(
            row.sample_id
        )

        candidate = SimpleNamespace(
            beta0=float(row.clipped_beta0),
            beta1=float(row.clipped_beta1),
            beta2=float(row.clipped_beta2),
            theta1=float(row.clipped_theta1),
            theta2=float(row.clipped_theta2),
            R1_fast=float(row.R1_fast),
            R1_slow=float(row.R1_slow),
            R2_fast=float(row.R2_fast),
            R2_slow=float(row.R2_slow),
        )

        seed_root = int(
            row.seed_root
        )

        try:
            result = evaluate_candidate(
                row=candidate,
                grid=grid,
                n_paths=args.n_paths,
                seed_root=seed_root,
            )

        except ValueError as exc:
            reason = str(exc)

            if reason not in EXPECTED_FAILURES:
                raise

            failure_rows.append({
                "sample_id": sample_id,
                "anchor_date": str(row.anchor_date),
                "seed_root": seed_root,
                "failure_reason": reason,
            })

            print(
                f"sample={sample_id} "
                f"FAILED reason={reason}",
                flush=True,
            )

            continue

        predicted = np.array(
            [
                result["prices"][c]
                for c in price_cols
            ],
            dtype=float,
        )

        truth = np.array(
            [
                getattr(row, c)
                for c in price_cols
            ],
            dtype=float,
        )

        error = predicted - truth

        rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        mae = float(
            np.mean(
                np.abs(error)
            )
        )

        max_abs = float(
            np.max(
                np.abs(error)
            )
        )

        scenario_rows.append({
            "sample_id": sample_id,
            "anchor_date": str(row.anchor_date),
            "seed_root": seed_root,
            "price_rmse": rmse,
            "price_mae": mae,
            "price_max_abs_error": max_abs,
            "price_rmse_bps_forward": rmse * 10000.0,
            "price_mae_bps_forward": mae * 10000.0,
            "min_vol": result["min_vol"],
            "max_vol": result["max_vol"],
            "cap_fraction": result["cap_fraction"],
        })

        for grid_row, true_price, pred_price, err in zip(
            grid.itertuples(index=False),
            truth,
            predicted,
            error,
        ):
            residual_rows.append({
                "sample_id": sample_id,
                "anchor_date": str(row.anchor_date),
                "target_dte": int(grid_row.target_dte),
                "grid_index": int(grid_row.grid_index),
                "log_moneyness": float(grid_row.log_moneyness),
                "moneyness": float(grid_row.moneyness),
                "otm_right": str(grid_row.otm_right),
                "true_price": float(true_price),
                "repriced_price": float(pred_price),
                "error": float(err),
                "abs_error": float(abs(err)),
                "squared_error": float(err ** 2),
            })

        if (
            i == 1
            or i % 5 == 0
            or i == len(subset)
        ):
            print(
                f"scenario={i:3d}/{len(subset)} "
                f"sample={sample_id:5d} "
                f"rmse={rmse:.8f} "
                f"({rmse * 10000:.3f} bp)",
                flush=True,
            )

    scenario_df = pd.DataFrame(
        scenario_rows
    )

    residual_df = pd.DataFrame(
        residual_rows
    )

    failure_df = pd.DataFrame(
        failure_rows
    )

    scenario_path = (
        args.output_dir
        / f"repricing_scenarios_chunk_{args.chunk_id:02d}.csv"
    )

    residual_path = (
        args.output_dir
        / f"repricing_residuals_chunk_{args.chunk_id:02d}.csv"
    )

    failure_path = (
        args.output_dir
        / f"repricing_failures_chunk_{args.chunk_id:02d}.csv"
    )

    scenario_df.to_csv(
        scenario_path,
        index=False,
    )

    residual_df.to_csv(
        residual_path,
        index=False,
    )

    failure_df.to_csv(
        failure_path,
        index=False,
    )

    print()
    print("=" * 88)
    print("CHUNK SUMMARY")
    print("=" * 88)
    print("successful:", len(scenario_df))
    print("failed:", len(failure_df))

    if len(scenario_df):
        print(
            "mean scenario RMSE:",
            f"{scenario_df['price_rmse'].mean():.8f}",
        )
        print(
            "median scenario RMSE:",
            f"{scenario_df['price_rmse'].median():.8f}",
        )
        print(
            "mean scenario RMSE (bp forward):",
            f"{scenario_df['price_rmse_bps_forward'].mean():.4f}",
        )

    print()
    print("Wrote:", scenario_path)
    print("Wrote:", residual_path)
    print("Wrote:", failure_path)


if __name__ == "__main__":
    main()
