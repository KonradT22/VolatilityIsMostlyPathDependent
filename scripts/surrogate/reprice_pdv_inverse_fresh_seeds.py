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


BASE = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "inverse_ann/pdv_anchor_jitter_10k"
)

DEFAULT_TEST = (
    BASE / "splits_interpolation/test.csv"
)

DEFAULT_PREDICTIONS = (
    BASE
    / "models/mlp_interpolation/test_evaluation/"
    "test_predictions.csv"
)

DEFAULT_OUTDIR = (
    BASE
    / "models/mlp_interpolation/test_evaluation/"
    "fresh_seed_repricing"
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

DEFAULT_SEEDS = [
    2026081840,
    2026081841,
    2026081842,
]

SELECTION_SEED = 2026081843
SCENARIOS_PER_ANCHOR = 4


def build_candidate(row, prefix):
    return SimpleNamespace(
        beta0=float(
            getattr(row, f"{prefix}beta0")
        ),
        beta1=float(
            getattr(row, f"{prefix}beta1")
        ),
        beta2=float(
            getattr(row, f"{prefix}beta2")
        ),
        theta1=float(
            getattr(row, f"{prefix}theta1")
        ),
        theta2=float(
            getattr(row, f"{prefix}theta2")
        ),
        R1_fast=float(row.R1_fast),
        R1_slow=float(row.R1_slow),
        R2_fast=float(row.R2_fast),
        R2_slow=float(row.R2_slow),
    )


def evaluate_surface(
    candidate,
    grid,
    n_paths,
    seed,
):
    return evaluate_candidate(
        row=candidate,
        grid=grid,
        n_paths=n_paths,
        seed_root=seed,
    )


def select_balanced_scenarios(
    test,
    predictions,
):
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

    if len(merged) != len(test):
        raise RuntimeError(
            "Prediction/test merge lost rows"
        )

    selected_parts = []

    for anchor, group in merged.groupby(
        "anchor_date",
        sort=True,
    ):
        if len(group) < SCENARIOS_PER_ANCHOR:
            raise RuntimeError(
                f"Anchor {anchor} has only "
                f"{len(group)} rows"
            )

        part = group.sample(
            n=SCENARIOS_PER_ANCHOR,
            random_state=(
                SELECTION_SEED
                + int(
                    str(anchor)
                    .replace("-", "")
                )
            ),
        )

        selected_parts.append(part)

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected = (
        selected.sort_values(
            [
                "anchor_date",
                "sample_id",
            ]
        )
        .reset_index(drop=True)
    )

    if len(selected) != 64:
        raise RuntimeError(
            f"Expected 64 selected scenarios, "
            f"found {len(selected)}"
        )

    if (
        selected["anchor_date"].nunique()
        != 16
    ):
        raise RuntimeError(
            "Expected all 16 anchors"
        )

    return selected


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_TEST,
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

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    test = pd.read_csv(
        args.test
    )

    predictions = pd.read_csv(
        args.predictions
    )

    selected = select_balanced_scenarios(
        test,
        predictions,
    )

    # Only chunk 0 writes the shared deterministic
    # scenario-selection manifest. This avoids concurrent
    # array tasks writing the same file.
    if args.chunk_id == 0:
        selected.to_csv(
            args.output_dir
            / "selected_balanced_scenarios.csv",
            index=False,
        )

    start = args.start_row
    stop = min(
        start + args.count,
        len(selected),
    )

    subset = (
        selected.iloc[start:stop]
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

    print("=" * 88)
    print("PDV INVERSE ANN — FRESH-SEED REPRICING")
    print("=" * 88)
    print("chunk:", args.chunk_id)
    print("rows:", f"{start}:{stop}")
    print("scenarios:", len(subset))
    print("paths:", args.n_paths)
    print("fresh seeds:", args.seeds)
    print(
        "paired simulations:",
        len(subset)
        * len(args.seeds),
    )
    print()

    pair_rows = []
    residual_rows = []
    failure_rows = []

    for scenario_i, row in enumerate(
        subset.itertuples(index=False),
        start=1,
    ):
        sample_id = int(row.sample_id)

        true_candidate = build_candidate(
            row,
            "",
        )

        pred_candidate = build_candidate(
            row,
            "clipped_",
        )

        for seed in args.seeds:
            try:
                true_result = evaluate_surface(
                    candidate=true_candidate,
                    grid=grid,
                    n_paths=args.n_paths,
                    seed=seed,
                )

            except ValueError as exc:
                reason = str(exc)

                if reason not in EXPECTED_FAILURES:
                    raise

                failure_rows.append({
                    "sample_id":
                        sample_id,
                    "anchor_date":
                        str(row.anchor_date),
                    "seed":
                        int(seed),
                    "side":
                        "true",
                    "failure_reason":
                        reason,
                })

                continue

            try:
                pred_result = evaluate_surface(
                    candidate=pred_candidate,
                    grid=grid,
                    n_paths=args.n_paths,
                    seed=seed,
                )

            except ValueError as exc:
                reason = str(exc)

                if reason not in EXPECTED_FAILURES:
                    raise

                failure_rows.append({
                    "sample_id":
                        sample_id,
                    "anchor_date":
                        str(row.anchor_date),
                    "seed":
                        int(seed),
                    "side":
                        "predicted",
                    "failure_reason":
                        reason,
                })

                continue

            true_prices = np.array(
                [
                    true_result["prices"][c]
                    for c in price_cols
                ],
                dtype=float,
            )

            pred_prices = np.array(
                [
                    pred_result["prices"][c]
                    for c in price_cols
                ],
                dtype=float,
            )

            error = (
                pred_prices
                - true_prices
            )

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

            pair_rows.append({
                "sample_id":
                    sample_id,
                "anchor_date":
                    str(row.anchor_date),
                "seed":
                    int(seed),
                "price_rmse":
                    rmse,
                "price_mae":
                    mae,
                "price_max_abs_error":
                    max_abs,
                "price_rmse_bps_forward":
                    rmse * 10000.0,
                "price_mae_bps_forward":
                    mae * 10000.0,
                "true_min_vol":
                    true_result["min_vol"],
                "pred_min_vol":
                    pred_result["min_vol"],
                "true_cap_fraction":
                    true_result[
                        "cap_fraction"
                    ],
                "pred_cap_fraction":
                    pred_result[
                        "cap_fraction"
                    ],
            })

            for (
                grid_row,
                true_price,
                pred_price,
                err,
            ) in zip(
                grid.itertuples(
                    index=False
                ),
                true_prices,
                pred_prices,
                error,
            ):
                residual_rows.append({
                    "sample_id":
                        sample_id,
                    "anchor_date":
                        str(row.anchor_date),
                    "seed":
                        int(seed),
                    "target_dte":
                        int(
                            grid_row.target_dte
                        ),
                    "grid_index":
                        int(
                            grid_row.grid_index
                        ),
                    "log_moneyness":
                        float(
                            grid_row.log_moneyness
                        ),
                    "true_price":
                        float(true_price),
                    "predicted_price":
                        float(pred_price),
                    "error":
                        float(err),
                    "abs_error":
                        float(abs(err)),
                    "squared_error":
                        float(err ** 2),
                })

        if (
            scenario_i == 1
            or scenario_i % 2 == 0
            or scenario_i
            == len(subset)
        ):
            print(
                f"scenario="
                f"{scenario_i:2d}/"
                f"{len(subset)} "
                f"sample={sample_id:5d} "
                f"pairs_done="
                f"{len(pair_rows):3d} "
                f"failures="
                f"{len(failure_rows):2d}",
                flush=True,
            )

    pair_df = pd.DataFrame(
        pair_rows
    )

    residual_df = pd.DataFrame(
        residual_rows
    )

    failure_df = pd.DataFrame(
        failure_rows
    )

    pair_path = (
        args.output_dir
        / (
            "fresh_seed_pairs_chunk_"
            f"{args.chunk_id:02d}.csv"
        )
    )

    residual_path = (
        args.output_dir
        / (
            "fresh_seed_residuals_chunk_"
            f"{args.chunk_id:02d}.csv"
        )
    )

    failure_path = (
        args.output_dir
        / (
            "fresh_seed_failures_chunk_"
            f"{args.chunk_id:02d}.csv"
        )
    )

    pair_df.to_csv(
        pair_path,
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
    print(
        "successful pairs:",
        len(pair_df),
    )
    print(
        "failures:",
        len(failure_df),
    )

    if len(pair_df):
        print(
            "mean paired RMSE:",
            f"{pair_df['price_rmse'].mean():.8f}",
        )

        print(
            "median paired RMSE:",
            f"{pair_df['price_rmse'].median():.8f}",
        )

        print(
            "mean paired RMSE bp:",
            f"{pair_df['price_rmse_bps_forward'].mean():.4f}",
        )

    print()
    print("Wrote:", pair_path)
    print("Wrote:", residual_path)
    print("Wrote:", failure_path)


if __name__ == "__main__":
    main()
