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

N_PATHS = 20000

SEEDS = [
    2026080720,
    2026080721,
    2026080722,
]


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

        seed_scores = []

        for seed in SEEDS:
            result = evaluate_surface(
                params=params,
                market=market,
                R_init1=r1,
                R_init2=r2,
                n_paths=N_PATHS,
                seed_root=seed,
            )

            score = result[
                "global_rmse"
            ]

            seed_scores.append(
                score
            )

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
                "validation_20k_rmse":
                    float(score),
                "validation_equal_tenor_rmse":
                    float(
                        result[
                            "equal_tenor_rmse"
                        ]
                    ),
            })

            print(
                f"seed={seed} "
                f"rmse={score:.10f}",
                flush=True,
            )

        mean_score = float(
            np.mean(seed_scores)
        )

        print(
            "mean independent RMSE:",
            f"{mean_score:.10f}",
            flush=True,
        )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUTDIR
        / "independent_seed_results.csv",
        index=False,
    )

    per_date = (
        results.groupby(
            "trade_date",
            as_index=False,
        )
        .agg(
            calibration_4k_rmse=(
                "calibration_4k_rmse",
                "first",
            ),
            validation_20k_mean_rmse=(
                "validation_20k_rmse",
                "mean",
            ),
            validation_20k_std_rmse=(
                "validation_20k_rmse",
                "std",
            ),
            validation_20k_min_rmse=(
                "validation_20k_rmse",
                "min",
            ),
            validation_20k_max_rmse=(
                "validation_20k_rmse",
                "max",
            ),
        )
    )

    per_date[
        "validation_minus_calibration"
    ] = (
        per_date[
            "validation_20k_mean_rmse"
        ]
        - per_date[
            "calibration_4k_rmse"
        ]
    )

    per_date[
        "validation_over_calibration"
    ] = (
        per_date[
            "validation_20k_mean_rmse"
        ]
        / per_date[
            "calibration_4k_rmse"
        ]
    )

    per_date.to_csv(
        OUTDIR
        / "independent_validation_by_date.csv",
        index=False,
    )

    summary = {
        "n_dates": 20,
        "N_paths": N_PATHS,
        "seeds": SEEDS,
        "mean_calibration_4k_rmse":
            float(
                per_date[
                    "calibration_4k_rmse"
                ].mean()
            ),
        "mean_validation_20k_rmse":
            float(
                per_date[
                    "validation_20k_mean_rmse"
                ].mean()
            ),
        "median_validation_20k_rmse":
            float(
                per_date[
                    "validation_20k_mean_rmse"
                ].median()
            ),
        "mean_validation_over_calibration":
            float(
                per_date[
                    "validation_over_calibration"
                ].mean()
            ),
        "worst_validation_20k_mean_rmse":
            float(
                per_date[
                    "validation_20k_mean_rmse"
                ].max()
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
