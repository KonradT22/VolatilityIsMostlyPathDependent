"""Generate 10,000 synthetic PDV parameter candidates for inverse-learning-dataset
construction: sample one of the 16 robust historical anchors uniformly, then perturb
its five parameters with independent Gaussian jitter in bounded logit space so
candidates stay within documented parameter bounds. Each candidate keeps its
anchor's path-dependent state (R1_fast, R1_slow, R2_fast, R2_slow)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_market_5param_fixed_grid"
)

PRIOR_DIR = (
    BASE / "robust_historical_prior"
)

ROBUST_PATH = (
    PRIOR_DIR
    / "robust_historical_dates.csv"
)

ANCHOR_PATH = (
    PRIOR_DIR
    / "robust_anchor_state_table.csv"
)

SAMPLES_PATH = (
    PRIOR_DIR
    / "anchor_jitter_samples.csv"
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

BOUNDS = {
    "beta0": (0.02, 0.06),
    "beta1": (-0.16, -0.06),
    "beta2": (0.55, 0.85),
    "theta1": (0.0, 1.0),
    "theta2": (0.0, 1.0),
}

RNG_SEED = 2026081802
N_SAMPLES = 10000

# Local perturbation in bounded/logit space.
# The historical anchor distribution supplies the large-scale
# variation; this only fills neighborhoods around those anchors.
JITTER_LOGIT_SD = 0.30

EPS = 1e-7


def logit(x):
    return np.log(
        x / (1.0 - x)
    )


def sigmoid(x):
    return 1.0 / (
        1.0 + np.exp(-x)
    )


def to_unit(x):
    result = np.empty_like(
        x,
        dtype=float,
    )

    for j, p in enumerate(PARAMS):
        lo, hi = BOUNDS[p]

        result[:, j] = (
            x[:, j] - lo
        ) / (hi - lo)

    return np.clip(
        result,
        EPS,
        1.0 - EPS,
    )


def from_unit(u):
    result = np.empty_like(
        u,
        dtype=float,
    )

    for j, p in enumerate(PARAMS):
        lo, hi = BOUNDS[p]

        result[:, j] = (
            lo
            + u[:, j]
            * (hi - lo)
        )

    return result


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


def build_anchor_table():
    robust = pd.read_csv(
        ROBUST_PATH
    )

    if len(robust) != 16:
        raise RuntimeError(
            f"Expected 16 robust dates, "
            f"found {len(robust)}"
        )

    rows = []

    for row in robust.itertuples(
        index=False
    ):
        date = str(
            row.trade_date
        )

        summary_path = (
            BASE
            / date
            / "calibration_summary.json"
        )

        with open(summary_path) as f:
            summary = json.load(f)

        r1 = summary["R_init1"]
        r2 = summary["R_init2"]

        if (
            len(r1) != 2
            or len(r2) != 2
        ):
            raise RuntimeError(
                f"{date}: unexpected R_init dimensions"
            )

        rows.append({
            "trade_date": date,
            "beta0": float(row.beta0),
            "beta1": float(row.beta1),
            "beta2": float(row.beta2),
            "theta1": float(row.theta1),
            "theta2": float(row.theta2),
            "R1_fast": float(r1[0]),
            "R1_slow": float(r1[1]),
            "R2_fast": float(r2[0]),
            "R2_slow": float(r2[1]),
            "validation_20k_mean_rmse":
                float(
                    row.validation_20k_mean_rmse
                ),
        })

    anchors = pd.DataFrame(
        rows
    )

    anchors.to_csv(
        ANCHOR_PATH,
        index=False,
    )

    return anchors


def main():
    anchors = build_anchor_table()

    rng = np.random.default_rng(
        RNG_SEED
    )

    anchor_indices = rng.integers(
        low=0,
        high=len(anchors),
        size=N_SAMPLES,
    )

    anchor_params = (
        anchors.iloc[
            anchor_indices
        ][PARAMS]
        .to_numpy(dtype=float)
    )

    anchor_states = (
        anchors.iloc[
            anchor_indices
        ][STATE_COLS]
        .to_numpy(dtype=float)
    )

    anchor_dates = (
        anchors.iloc[
            anchor_indices
        ]["trade_date"]
        .astype(str)
        .to_numpy()
    )

    unit = to_unit(
        anchor_params
    )

    latent = logit(
        unit
    )

    jitter = rng.normal(
        loc=0.0,
        scale=JITTER_LOGIT_SD,
        size=latent.shape,
    )

    sampled_unit = sigmoid(
        latent + jitter
    )

    sampled_params = from_unit(
        sampled_unit
    )

    samples = pd.DataFrame(
        sampled_params,
        columns=PARAMS,
    )

    samples.insert(
        0,
        "anchor_date",
        anchor_dates,
    )

    samples.insert(
        0,
        "sample_id",
        np.arange(
            1,
            N_SAMPLES + 1,
        ),
    )

    for j, col in enumerate(
        STATE_COLS
    ):
        samples[col] = (
            anchor_states[:, j]
        )

    samples.to_csv(
        SAMPLES_PATH,
        index=False,
    )

    hist_desc = describe(
        anchors
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
        PRIOR_DIR
        / "anchor_jitter_diagnostics.csv"
    )

    hist_corr = (
        anchors[PARAMS]
        .corr()
    )

    sample_corr = (
        samples[PARAMS]
        .corr()
    )

    hist_corr.to_csv(
        PRIOR_DIR
        / "anchor_historical_correlation.csv"
    )

    sample_corr.to_csv(
        PRIOR_DIR
        / "anchor_jitter_correlation.csv"
    )

    anchor_counts = (
        samples[
            "anchor_date"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "anchor_date"
        )
        .reset_index(
            name="sample_count"
        )
    )

    anchor_counts.to_csv(
        PRIOR_DIR
        / "anchor_jitter_anchor_counts.csv",
        index=False,
    )

    summary = {
        "rng_seed": RNG_SEED,
        "n_samples": N_SAMPLES,
        "n_anchors": int(
            len(anchors)
        ),
        "jitter_logit_sd":
            JITTER_LOGIT_SD,
        "sampling_method":
            "uniform historical anchor plus "
            "independent local Gaussian jitter "
            "in bounded logit parameter space",
        "parameter_bounds": {
            p: list(BOUNDS[p])
            for p in PARAMS
        },
        "state_features":
            STATE_COLS,
    }

    with open(
        PRIOR_DIR
        / "anchor_jitter_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("=" * 78)
    print("PDV EMPIRICAL ANCHOR + JITTER PRIOR")
    print("=" * 78)

    print(
        f"Robust historical anchors: "
        f"{len(anchors)}"
    )

    print(
        f"Synthetic candidates: "
        f"{len(samples):,}"
    )

    print(
        f"Logit jitter SD: "
        f"{JITTER_LOGIT_SD:.3f}"
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
        "\nHISTORICAL CORRELATION"
    )

    print(
        hist_corr.to_string(
            float_format=lambda x:
                f"{x: .4f}",
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
        "\nSTATE SUMMARY"
    )

    print(
        anchors[
            STATE_COLS
        ]
        .describe()
        .T
        .to_string(
            float_format=lambda x:
                f"{x:.8f}",
        )
    )

    print(
        "\nANCHOR COUNTS"
    )

    print(
        anchor_counts.to_string(
            index=False
        )
    )

    print(
        "\nWrote:",
        SAMPLES_PATH,
    )


if __name__ == "__main__":
    main()
