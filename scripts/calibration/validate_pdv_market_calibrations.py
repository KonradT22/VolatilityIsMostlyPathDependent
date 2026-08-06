import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

import scripts.calibration.calibrate_pdv_market_multi_maturity as market

from calibration.torch_montecarlo import (
    initialize_R,
    identity,
    squared,
)


OUTDIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_market_validation"
)

CANDIDATES = {
    "baseline": np.array(
        [0.04, -0.13, 0.65],
        dtype=float,
    ),
    "bounded_calibration": np.array(
        [
            0.05996444574148981,
            -0.14386034079634102,
            0.6180842038647475,
        ],
        dtype=float,
    ),
    "expanded_beta0_calibration": np.array(
        [
            0.06806084199821262,
            -0.14746539744666748,
            0.5658674736127569,
        ],
        dtype=float,
    ),
}

SEEDS = [
    2026080601,
    2026080602,
    2026080603,
    2026080604,
    2026080605,
]

VALIDATION_N = 20000


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    market_df = market.load_market()

    spx = market.load_spx_history()

    lam1 = torch.tensor(
        market.LAM1,
        dtype=torch.float32,
    )

    lam2 = torch.tensor(
        market.LAM2,
        dtype=torch.float32,
    )

    R_init1 = initialize_R(
        lam1,
        past_prices=spx,
        transform=identity,
    )

    R_init2 = initialize_R(
        lam2,
        past_prices=spx,
        transform=squared,
    )

    original_n = market.N_PATHS
    original_seed = market.SEED_ROOT

    market.N_PATHS = VALIDATION_N

    rows = []
    detail_frames = []

    print(
        f"Validation paths per run: {VALIDATION_N}"
    )
    print(
        f"Seeds: {SEEDS}"
    )
    print()

    try:
        for name, betas in CANDIDATES.items():
            print("=" * 72)
            print(name)
            print("betas:", betas)
            print("=" * 72)

            for seed in SEEDS:
                market.SEED_ROOT = seed

                result = market.evaluate_surface(
                    betas,
                    market_df,
                    R_init1,
                    R_init2,
                )

                row = {
                    "candidate": name,
                    "seed": seed,
                    "N_paths": VALIDATION_N,
                    "beta0": betas[0],
                    "beta1": betas[1],
                    "beta2": betas[2],
                    "equal_maturity_rmse": (
                        result[
                            "equal_maturity_rmse"
                        ]
                    ),
                    "global_rmse": (
                        result["global_rmse"]
                    ),
                }

                for dte, value in (
                    result[
                        "per_dte_rmse"
                    ].items()
                ):
                    row[
                        f"rmse_{int(dte)}d"
                    ] = value

                rows.append(row)

                detail = (
                    result["rows"]
                    .copy()
                )

                detail[
                    "candidate"
                ] = name

                detail[
                    "seed"
                ] = seed

                detail_frames.append(
                    detail
                )

                print(
                    f"seed={seed} "
                    f"equal="
                    f"{result['equal_maturity_rmse']:.8f} "
                    f"global="
                    f"{result['global_rmse']:.8f}"
                )

            print()

    finally:
        market.N_PATHS = original_n
        market.SEED_ROOT = original_seed

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTDIR
        / "validation_runs.csv",
        index=False,
    )

    details = pd.concat(
        detail_frames,
        ignore_index=True,
    )

    details.to_csv(
        OUTDIR
        / "validation_quote_errors.csv",
        index=False,
    )

    summary = (
        results.groupby(
            "candidate"
        )
        .agg(
            mean_equal_rmse=(
                "equal_maturity_rmse",
                "mean",
            ),
            std_equal_rmse=(
                "equal_maturity_rmse",
                "std",
            ),
            min_equal_rmse=(
                "equal_maturity_rmse",
                "min",
            ),
            max_equal_rmse=(
                "equal_maturity_rmse",
                "max",
            ),
            mean_global_rmse=(
                "global_rmse",
                "mean",
            ),
            std_global_rmse=(
                "global_rmse",
                "std",
            ),
        )
        .sort_values(
            "mean_equal_rmse"
        )
    )

    summary.to_csv(
        OUTDIR
        / "validation_summary.csv"
    )

    print()
    print("=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)
    print(summary.to_string())


if __name__ == "__main__":
    main()
