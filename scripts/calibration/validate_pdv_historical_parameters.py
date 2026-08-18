import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from calibration.torch_montecarlo import (
    initialize_R,
    identity,
    squared,
)

from scripts.calibration.calibrate_pdv_market_5param_fixed_grid import (
    BASE_OUTDIR,
    LAM1,
    LAM2,
    evaluate_surface,
    load_state_history,
    load_targets,
)


SEQUENCE = (
    BASE_OUTDIR
    / "historical_sequence_summary.csv"
)

OUTDIR = (
    BASE_OUTDIR
    / "historical_independent_validation"
)

RESULTS_PATH = (
    OUTDIR
    / "independent_seed_results.csv"
)

PER_DATE_PATH = (
    OUTDIR
    / "independent_validation_by_date.csv"
)

N_PATHS = 20000

SEEDS = [
    2026080720,
    2026080721,
    2026080722,
]

EXPECTED_MODEL_FAILURES = {
    "nonpositive_volatility",
    "nonfinite_volatility",
    "nonfinite_model_option_price",
}


def build_state(trade_date):
    history, _ = load_state_history(
        trade_date
    )

    max_delta = min(
        1000,
        len(history) - 1,
    )

    lam1 = torch.tensor(
        LAM1,
        dtype=torch.float32,
    )

    lam2 = torch.tensor(
        LAM2,
        dtype=torch.float32,
    )

    r1 = initialize_R(
        lam1,
        past_prices=history,
        max_delta=max_delta,
        transform=identity,
    )

    r2 = initialize_R(
        lam2,
        past_prices=history,
        max_delta=max_delta,
        transform=squared,
    )

    return r1, r2


def save_checkpoint(rows):
    pd.DataFrame(rows).to_csv(
        RESULTS_PATH,
        index=False,
    )


def build_per_date(results):
    rows = []

    for date, g in results.groupby(
        "trade_date",
        sort=True,
    ):
        valid = g[
            g["valid"]
        ].copy()

        calibration_rmse = float(
            g["calibration_4k_rmse"].iloc[0]
        )

        if len(valid):
            mean_rmse = float(
                valid[
                    "validation_20k_rmse"
                ].mean()
            )

            std_rmse = (
                float(
                    valid[
                        "validation_20k_rmse"
                    ].std()
                )
                if len(valid) > 1
                else 0.0
            )

            min_rmse = float(
                valid[
                    "validation_20k_rmse"
                ].min()
            )

            max_rmse = float(
                valid[
                    "validation_20k_rmse"
                ].max()
            )

            difference = (
                mean_rmse
                - calibration_rmse
            )

            ratio = (
                mean_rmse
                / calibration_rmse
            )

        else:
            mean_rmse = np.nan
            std_rmse = np.nan
            min_rmse = np.nan
            max_rmse = np.nan
            difference = np.nan
            ratio = np.nan

        rows.append({
            "trade_date": date,
            "calibration_4k_rmse":
                calibration_rmse,
            "valid_seed_count":
                int(len(valid)),
            "invalid_seed_count":
                int((~g["valid"]).sum()),
            "all_seeds_valid":
                bool(g["valid"].all()),
            "validation_20k_mean_rmse":
                mean_rmse,
            "validation_20k_std_rmse":
                std_rmse,
            "validation_20k_min_rmse":
                min_rmse,
            "validation_20k_max_rmse":
                max_rmse,
            "validation_minus_calibration":
                difference,
            "validation_over_calibration":
                ratio,
        })

    return pd.DataFrame(rows)


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    historical = pd.read_csv(
        SEQUENCE
    )

    if len(historical) != 20:
        raise RuntimeError(
            f"Expected 20 historical rows, "
            f"found {len(historical)}"
        )

    if not historical["success"].all():
        raise RuntimeError(
            "Historical sequence contains "
            "unsuccessful calibrations"
        )

    rows = []

    for day_number, row in enumerate(
        historical.itertuples(
            index=False
        ),
        start=1,
    ):
        date = str(row.trade_date)

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

        market = load_targets(
            date
        )

        r1, r2 = build_state(
            date
        )

        print()
        print("=" * 78)
        print(
            f"{day_number:02d}/20  {date}"
        )
        print("=" * 78)

        for seed in SEEDS:
            valid = True
            failure_reason = ""
            score = np.nan
            equal_tenor_score = np.nan

            try:
                result = evaluate_surface(
                    params=params,
                    market=market,
                    R_init1=r1,
                    R_init2=r2,
                    n_paths=N_PATHS,
                    seed_root=seed,
                )

                score = float(
                    result[
                        "global_rmse"
                    ]
                )

                equal_tenor_score = float(
                    result[
                        "equal_tenor_rmse"
                    ]
                )

            except ValueError as exc:
                failure_reason = str(exc)

                if (
                    failure_reason
                    not in EXPECTED_MODEL_FAILURES
                ):
                    raise

                valid = False

            rows.append({
                "trade_date": date,
                "seed_root": seed,
                "N_paths": N_PATHS,
                "beta0": params[0],
                "beta1": params[1],
                "beta2": params[2],
                "theta1": params[3],
                "theta2": params[4],
                "calibration_4k_rmse":
                    float(
                        row.final_global_rmse
                    ),
                "valid": valid,
                "failure_reason":
                    failure_reason,
                "validation_20k_rmse":
                    score,
                "validation_equal_tenor_rmse":
                    equal_tenor_score,
            })

            # Preserve every completed seed immediately.
            save_checkpoint(rows)

            if valid:
                print(
                    f"seed={seed} "
                    f"VALID "
                    f"rmse={score:.10f}",
                    flush=True,
                )
            else:
                print(
                    f"seed={seed} "
                    f"INVALID "
                    f"reason={failure_reason}",
                    flush=True,
                )

    results = pd.DataFrame(
        rows
    )

    per_date = build_per_date(
        results
    )

    per_date.to_csv(
        PER_DATE_PATH,
        index=False,
    )

    valid_results = results[
        results["valid"]
    ]

    invalid_results = results[
        ~results["valid"]
    ]

    valid_dates = per_date[
        per_date["all_seeds_valid"]
    ]

    summary = {
        "n_dates": 20,
        "N_paths": N_PATHS,
        "seeds": SEEDS,
        "total_evaluations":
            int(len(results)),
        "valid_evaluations":
            int(len(valid_results)),
        "invalid_evaluations":
            int(len(invalid_results)),
        "dates_all_seeds_valid":
            int(
                per_date[
                    "all_seeds_valid"
                ].sum()
            ),
        "dates_with_invalid_seed":
            int(
                (
                    ~per_date[
                        "all_seeds_valid"
                    ]
                ).sum()
            ),
        "mean_calibration_4k_rmse":
            float(
                per_date[
                    "calibration_4k_rmse"
                ].mean()
            ),
        "mean_validation_20k_rmse_valid_only":
            float(
                valid_results[
                    "validation_20k_rmse"
                ].mean()
            ),
        "median_validation_20k_rmse_valid_only":
            float(
                valid_results[
                    "validation_20k_rmse"
                ].median()
            ),
        "mean_validation_20k_rmse_fully_valid_dates":
            (
                float(
                    valid_dates[
                        "validation_20k_mean_rmse"
                    ].mean()
                )
                if len(valid_dates)
                else None
            ),
        "worst_validation_20k_mean_rmse_fully_valid_dates":
            (
                float(
                    valid_dates[
                        "validation_20k_mean_rmse"
                    ].max()
                )
                if len(valid_dates)
                else None
            ),
    }

    with open(
        OUTDIR
        / "validation_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print("INDEPENDENT VALIDATION SUMMARY")
    print("=" * 78)

    print(
        per_date.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.10f}",
        )
    )

    print()
    print("INVALID EVALUATIONS")
    print("-" * 78)

    if len(invalid_results):
        print(
            invalid_results[
                [
                    "trade_date",
                    "seed_root",
                    "beta0",
                    "beta1",
                    "beta2",
                    "theta1",
                    "theta2",
                    "failure_reason",
                ]
            ].to_string(
                index=False
            )
        )
    else:
        print("None")

    print()
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("Wrote:", OUTDIR)


if __name__ == "__main__":
    main()
