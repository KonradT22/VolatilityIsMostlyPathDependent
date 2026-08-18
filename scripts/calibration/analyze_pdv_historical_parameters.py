import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "/users/4/trest017/urop_pdv/benchmarks/calibration/"
    "pdv_market_5param_fixed_grid/"
    "historical_sequence_summary.csv"
)

OUTDIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/calibration/"
    "pdv_market_5param_fixed_grid/"
    "historical_parameter_analysis"
)

PARAMS = [
    "beta0",
    "beta1",
    "beta2",
    "theta1",
    "theta2",
]

BOUNDS = {
    "beta0": (0.02, 0.06),
    "beta1": (-0.16, -0.06),
    "beta2": (0.55, 0.85),
    "theta1": (0.0, 1.0),
    "theta2": (0.0, 1.0),
}


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT)

    if len(df) != 20:
        raise RuntimeError(
            f"Expected 20 dates, found {len(df)}"
        )

    if not df["success"].all():
        raise RuntimeError(
            "Not all historical calibrations succeeded"
        )

    X = df[PARAMS].to_numpy(dtype=float)

    mean = X.mean(axis=0)
    covariance = np.cov(
        X,
        rowvar=False,
        ddof=1,
    )

    correlation = np.corrcoef(
        X,
        rowvar=False,
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(covariance)
    )

    condition_number = float(
        eigenvalues.max()
        / eigenvalues.min()
    )

    stats = (
        df[PARAMS]
        .describe()
        .T
    )

    stats.to_csv(
        OUTDIR / "parameter_summary.csv"
    )

    pd.DataFrame(
        covariance,
        index=PARAMS,
        columns=PARAMS,
    ).to_csv(
        OUTDIR / "sample_covariance.csv"
    )

    pd.DataFrame(
        correlation,
        index=PARAMS,
        columns=PARAMS,
    ).to_csv(
        OUTDIR / "sample_correlation.csv"
    )

    pd.DataFrame({
        "component": np.arange(
            1,
            len(PARAMS) + 1,
        ),
        "eigenvalue": eigenvalues,
        "fraction_total_variance":
            eigenvalues
            / eigenvalues.sum(),
    }).to_csv(
        OUTDIR / "covariance_eigenvalues.csv",
        index=False,
    )

    pd.DataFrame(
        eigenvectors,
        index=PARAMS,
        columns=[
            f"component_{i}"
            for i in range(
                1,
                len(PARAMS) + 1,
            )
        ],
    ).to_csv(
        OUTDIR / "covariance_eigenvectors.csv"
    )

    boundary_rows = []

    for param in PARAMS:
        lo, hi = BOUNDS[param]
        width = hi - lo

        values = df[param].to_numpy(
            dtype=float
        )

        lower_distance = (
            values - lo
        ) / width

        upper_distance = (
            hi - values
        ) / width

        boundary_rows.append({
            "parameter": param,
            "lower_bound": lo,
            "upper_bound": hi,
            "within_1pct_lower":
                int(
                    np.sum(
                        lower_distance <= 0.01
                    )
                ),
            "within_1pct_upper":
                int(
                    np.sum(
                        upper_distance <= 0.01
                    )
                ),
            "within_5pct_lower":
                int(
                    np.sum(
                        lower_distance <= 0.05
                    )
                ),
            "within_5pct_upper":
                int(
                    np.sum(
                        upper_distance <= 0.05
                    )
                ),
        })

    boundary_df = pd.DataFrame(
        boundary_rows
    )

    boundary_df.to_csv(
        OUTDIR / "boundary_diagnostics.csv",
        index=False,
    )

    summary = {
        "n_dates": int(len(df)),
        "parameters": PARAMS,
        "mean": {
            p: float(v)
            for p, v in zip(
                PARAMS,
                mean,
            )
        },
        "covariance_condition_number":
            condition_number,
        "minimum_covariance_eigenvalue":
            float(eigenvalues.min()),
        "maximum_covariance_eigenvalue":
            float(eigenvalues.max()),
        "rmse": {
            "mean": float(
                df["final_global_rmse"].mean()
            ),
            "median": float(
                df["final_global_rmse"].median()
            ),
            "min": float(
                df["final_global_rmse"].min()
            ),
            "max": float(
                df["final_global_rmse"].max()
            ),
        },
    }

    with open(
        OUTDIR / "analysis_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("=" * 72)
    print("HISTORICAL PDV PARAMETER ANALYSIS")
    print("=" * 72)

    print("\nPARAMETER SUMMARY")
    print(
        stats[
            [
                "mean",
                "std",
                "min",
                "50%",
                "max",
            ]
        ].to_string()
    )

    print("\nCORRELATION")
    print(
        pd.DataFrame(
            correlation,
            index=PARAMS,
            columns=PARAMS,
        ).to_string(
            float_format=lambda x:
                f"{x: .4f}"
        )
    )

    print("\nCOVARIANCE EIGENVALUES")
    for i, value in enumerate(
        eigenvalues,
        start=1,
    ):
        print(
            f"{i}: {value:.12g}"
        )

    print(
        "\nCovariance condition number:",
        f"{condition_number:.4f}",
    )

    print("\nBOUNDARY DIAGNOSTICS")
    print(
        boundary_df.to_string(
            index=False
        )
    )

    print("\nRMSE")
    print(
        df["final_global_rmse"]
        .describe()
        .to_string()
    )

    print("\nWrote:", OUTDIR)


if __name__ == "__main__":
    main()
