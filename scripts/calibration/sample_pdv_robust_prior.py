import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_market_5param_fixed_grid/"
    "robust_historical_prior"
)

SUMMARY_PATH = (
    BASE / "robust_prior_summary.json"
)

COV_PATH = (
    BASE / "robust_parameter_covariance_shrunk.csv"
)

HIST_PATH = (
    BASE / "robust_historical_dates.csv"
)

OUT_PATH = (
    BASE / "bounded_parameter_samples.csv"
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

RNG_SEED = 2026081801
N_ACCEPTED = 10000
BATCH_SIZE = 25000


def inside_bounds(x):
    mask = np.ones(
        len(x),
        dtype=bool,
    )

    for j, p in enumerate(PARAMS):
        lo, hi = BOUNDS[p]

        mask &= (
            (x[:, j] >= lo)
            & (x[:, j] <= hi)
        )

    return mask


def describe(frame):
    return (
        frame[PARAMS]
        .describe(
            percentiles=[
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
            ]
        )
        .T
    )


def main():
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    mean = np.array(
        [
            summary[
                "parameter_mean"
            ][p]
            for p in PARAMS
        ],
        dtype=float,
    )

    covariance = (
        pd.read_csv(
            COV_PATH,
            index_col=0,
        )
        .loc[PARAMS, PARAMS]
        .to_numpy(dtype=float)
    )

    historical = pd.read_csv(
        HIST_PATH
    )

    rng = np.random.default_rng(
        RNG_SEED
    )

    accepted_chunks = []

    total_drawn = 0
    total_accepted = 0

    while total_accepted < N_ACCEPTED:
        candidates = (
            rng.multivariate_normal(
                mean=mean,
                cov=covariance,
                size=BATCH_SIZE,
            )
        )

        total_drawn += len(candidates)

        mask = inside_bounds(
            candidates
        )

        accepted = candidates[
            mask
        ]

        accepted_chunks.append(
            accepted
        )

        total_accepted += len(
            accepted
        )

        print(
            f"drawn={total_drawn:,} "
            f"accepted={total_accepted:,} "
            f"running_rate="
            f"{total_accepted / total_drawn:.4f}",
            flush=True,
        )

    X = np.vstack(
        accepted_chunks
    )[:N_ACCEPTED]

    samples = pd.DataFrame(
        X,
        columns=PARAMS,
    )

    samples.insert(
        0,
        "sample_id",
        np.arange(
            1,
            len(samples) + 1,
        ),
    )

    samples.to_csv(
        OUT_PATH,
        index=False,
    )

    acceptance_rate = (
        total_accepted
        / total_drawn
    )

    hist_desc = describe(
        historical
    )

    sample_desc = describe(
        samples
    )

    comparison = pd.DataFrame({
        "historical_mean":
            hist_desc["mean"],
        "sample_mean":
            sample_desc["mean"],
        "historical_std":
            hist_desc["std"],
        "sample_std":
            sample_desc["std"],
        "historical_median":
            hist_desc["50%"],
        "sample_median":
            sample_desc["50%"],
        "historical_min":
            hist_desc["min"],
        "sample_p05":
            sample_desc["5%"],
        "historical_max":
            hist_desc["max"],
        "sample_p95":
            sample_desc["95%"],
    })

    comparison.to_csv(
        BASE
        / "bounded_sample_diagnostics.csv"
    )

    sample_corr = (
        samples[PARAMS]
        .corr()
    )

    sample_corr.to_csv(
        BASE
        / "bounded_sample_correlation.csv"
    )

    result_summary = {
        "rng_seed": RNG_SEED,
        "requested_accepted_samples":
            N_ACCEPTED,
        "total_candidates_drawn":
            int(total_drawn),
        "total_candidates_accepted_before_trim":
            int(total_accepted),
        "acceptance_rate":
            float(acceptance_rate),
        "parameter_bounds": {
            p: list(BOUNDS[p])
            for p in PARAMS
        },
    }

    with open(
        BASE
        / "bounded_sample_summary.json",
        "w",
    ) as f:
        json.dump(
            result_summary,
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print("BOUNDED ROBUST PRIOR SAMPLING")
    print("=" * 78)

    print(
        f"Accepted sample count: "
        f"{len(samples):,}"
    )

    print(
        f"Observed acceptance rate: "
        f"{acceptance_rate:.4f}"
    )

    print(
        "\nHISTORICAL VS SYNTHETIC"
    )

    print(
        comparison.to_string(
            float_format=lambda x:
                f"{x:.8f}",
        )
    )

    print(
        "\nSYNTHETIC CORRELATION"
    )

    print(
        sample_corr.to_string(
            float_format=lambda x:
                f"{x: .4f}",
        )
    )

    print(
        "\nWrote:",
        OUT_PATH,
    )


if __name__ == "__main__":
    main()
