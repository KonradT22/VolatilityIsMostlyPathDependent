import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


BASE = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_market_5param_fixed_grid"
)

HISTORICAL_PATH = (
    BASE / "historical_sequence_summary.csv"
)

VALIDATION_PATH = (
    BASE
    / "historical_independent_validation"
    / "independent_validation_by_date.csv"
)

OUTDIR = (
    BASE
    / "robust_historical_prior"
)

PARAMS = [
    "beta0",
    "beta1",
    "beta2",
    "theta1",
    "theta2",
]


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    historical = pd.read_csv(
        HISTORICAL_PATH
    )

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    merged = historical.merge(
        validation[
            [
                "trade_date",
                "all_seeds_valid",
                "valid_seed_count",
                "invalid_seed_count",
                "validation_20k_mean_rmse",
            ]
        ],
        on="trade_date",
        how="left",
        validate="one_to_one",
    )

    if len(merged) != 20:
        raise RuntimeError(
            f"Expected 20 dates, found {len(merged)}"
        )

    robust = merged[
        merged["all_seeds_valid"] == True
    ].copy()

    excluded = merged[
        merged["all_seeds_valid"] != True
    ].copy()

    if len(robust) != 16:
        raise RuntimeError(
            f"Expected 16 robust dates, "
            f"found {len(robust)}"
        )

    X = robust[
        PARAMS
    ].to_numpy(dtype=float)

    mean = X.mean(axis=0)

    std = X.std(
        axis=0,
        ddof=1,
    )

    if np.any(std <= 0):
        raise RuntimeError(
            "Non-positive parameter standard deviation"
        )

    Z = (
        X - mean
    ) / std

    raw_standardized_covariance = np.cov(
        Z,
        rowvar=False,
        ddof=1,
    )

    raw_correlation = np.corrcoef(
        X,
        rowvar=False,
    )

    lw = LedoitWolf().fit(
        Z
    )

    shrunk_standardized_covariance = (
        lw.covariance_
    )

    scale = np.diag(
        std
    )

    shrunk_parameter_covariance = (
        scale
        @ shrunk_standardized_covariance
        @ scale
    )

    shrunk_standardized_correlation = (
        shrunk_standardized_covariance
        / np.sqrt(
            np.outer(
                np.diag(
                    shrunk_standardized_covariance
                ),
                np.diag(
                    shrunk_standardized_covariance
                ),
            )
        )
    )

    raw_condition = float(
        np.linalg.cond(
            raw_standardized_covariance
        )
    )

    shrunk_condition = float(
        np.linalg.cond(
            shrunk_standardized_covariance
        )
    )

    robust.to_csv(
        OUTDIR
        / "robust_historical_dates.csv",
        index=False,
    )

    excluded.to_csv(
        OUTDIR
        / "excluded_unstable_dates.csv",
        index=False,
    )

    pd.DataFrame({
        "parameter": PARAMS,
        "mean": mean,
        "std": std,
        "min": X.min(axis=0),
        "median": np.median(
            X,
            axis=0,
        ),
        "max": X.max(axis=0),
    }).to_csv(
        OUTDIR
        / "robust_parameter_summary.csv",
        index=False,
    )

    pd.DataFrame(
        raw_correlation,
        index=PARAMS,
        columns=PARAMS,
    ).to_csv(
        OUTDIR
        / "robust_raw_correlation.csv"
    )

    pd.DataFrame(
        raw_standardized_covariance,
        index=PARAMS,
        columns=PARAMS,
    ).to_csv(
        OUTDIR
        / "robust_standardized_covariance_raw.csv"
    )

    pd.DataFrame(
        shrunk_standardized_covariance,
        index=PARAMS,
        columns=PARAMS,
    ).to_csv(
        OUTDIR
        / "robust_standardized_covariance_shrunk.csv"
    )

    pd.DataFrame(
        shrunk_standardized_correlation,
        index=PARAMS,
        columns=PARAMS,
    ).to_csv(
        OUTDIR
        / "robust_standardized_correlation_shrunk.csv"
    )

    pd.DataFrame(
        shrunk_parameter_covariance,
        index=PARAMS,
        columns=PARAMS,
    ).to_csv(
        OUTDIR
        / "robust_parameter_covariance_shrunk.csv"
    )

    summary = {
        "n_historical_dates": int(
            len(merged)
        ),
        "n_robust_dates": int(
            len(robust)
        ),
        "n_excluded_dates": int(
            len(excluded)
        ),
        "robust_dates": (
            robust["trade_date"]
            .astype(str)
            .tolist()
        ),
        "excluded_dates": (
            excluded["trade_date"]
            .astype(str)
            .tolist()
        ),
        "parameters": PARAMS,
        "parameter_mean": {
            p: float(v)
            for p, v
            in zip(PARAMS, mean)
        },
        "parameter_std": {
            p: float(v)
            for p, v
            in zip(PARAMS, std)
        },
        "ledoit_wolf_shrinkage":
            float(lw.shrinkage_),
        "raw_standardized_condition_number":
            raw_condition,
        "shrunk_standardized_condition_number":
            shrunk_condition,
    }

    with open(
        OUTDIR / "robust_prior_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("=" * 76)
    print("ROBUST HISTORICAL PDV PRIOR")
    print("=" * 76)

    print(
        f"Historical dates: {len(merged)}"
    )
    print(
        f"Robust dates:     {len(robust)}"
    )
    print(
        f"Excluded dates:   {len(excluded)}"
    )

    print("\nEXCLUDED")
    print(
        excluded[
            [
                "trade_date",
                "beta0",
                "beta1",
                "beta2",
                "theta1",
                "theta2",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nROBUST PARAMETER SUMMARY")
    print(
        pd.DataFrame({
            "parameter": PARAMS,
            "mean": mean,
            "std": std,
            "min": X.min(axis=0),
            "median": np.median(
                X,
                axis=0,
            ),
            "max": X.max(axis=0),
        }).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8f}",
        )
    )

    print("\nRAW ROBUST CORRELATION")
    print(
        pd.DataFrame(
            raw_correlation,
            index=PARAMS,
            columns=PARAMS,
        ).to_string(
            float_format=lambda x:
                f"{x: .4f}"
        )
    )

    print(
        "\nLedoit-Wolf shrinkage:",
        f"{lw.shrinkage_:.6f}",
    )

    print(
        "Raw standardized condition number:",
        f"{raw_condition:.4f}",
    )

    print(
        "Shrunk standardized condition number:",
        f"{shrunk_condition:.4f}",
    )

    print("\nWrote:", OUTDIR)


if __name__ == "__main__":
    main()
