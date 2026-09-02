"""Fit and evaluate the primary OOD guardrail: squared Mahalanobis distance with
Ledoit-Wolf covariance shrinkage, over both the full 81-D input space (77 prices +
4 state variables) and a state-only 4-D variant. Threshold is fixed on a held-out
in-domain calibration split only, then evaluated for separation against later,
chronologically held-out dates (see docs/GUARDRAILS.md)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


SEED = 2026081805

BASE = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "inverse_ann/pdv_anchor_jitter_10k"
)

SPLIT_DIR = BASE / "splits"

OUTDIR = (
    BASE
    / "guardrails"
    / "mahalanobis_chronological"
)

STATE_COLS = [
    "R1_fast",
    "R1_slow",
    "R2_fast",
    "R2_slow",
]


def stratified_fit_calibration_split(
    df,
    calibration_fraction=0.20,
):
    rng = np.random.default_rng(SEED)

    fit_parts = []
    cal_parts = []

    for anchor, group in df.groupby(
        "anchor_date",
        sort=True,
    ):
        group = (
            group.copy()
            .reset_index(drop=True)
        )

        order = rng.permutation(
            len(group)
        )

        n_cal = max(
            1,
            int(
                round(
                    calibration_fraction
                    * len(group)
                )
            ),
        )

        cal_idx = order[:n_cal]
        fit_idx = order[n_cal:]

        fit_parts.append(
            group.iloc[fit_idx]
        )

        cal_parts.append(
            group.iloc[cal_idx]
        )

    fit = pd.concat(
        fit_parts,
        ignore_index=True,
    )

    calibration = pd.concat(
        cal_parts,
        ignore_index=True,
    )

    return fit, calibration


def fit_detector(
    fit_df,
    feature_cols,
):
    x = fit_df[
        feature_cols
    ].to_numpy(dtype=float)

    mean = x.mean(axis=0)

    std = x.std(
        axis=0,
        ddof=0,
    )

    std = np.where(
        std > 1e-12,
        std,
        1.0,
    )

    z = (
        x - mean
    ) / std

    covariance = LedoitWolf().fit(z)

    return {
        "mean": mean,
        "std": std,
        "model": covariance,
    }


def distances(
    detector,
    df,
    feature_cols,
):
    x = df[
        feature_cols
    ].to_numpy(dtype=float)

    z = (
        x - detector["mean"]
    ) / detector["std"]

    return detector[
        "model"
    ].mahalanobis(z)


def summarize_detector(
    name,
    feature_cols,
    fit_df,
    calibration_df,
    val_df,
    test_df,
):
    detector = fit_detector(
        fit_df,
        feature_cols,
    )

    d_fit = distances(
        detector,
        fit_df,
        feature_cols,
    )

    d_cal = distances(
        detector,
        calibration_df,
        feature_cols,
    )

    d_val = distances(
        detector,
        val_df,
        feature_cols,
    )

    d_test = distances(
        detector,
        test_df,
        feature_cols,
    )

    #
    # Threshold fixed using ID calibration data only.
    #
    threshold_95 = float(
        np.quantile(
            d_cal,
            0.95,
        )
    )

    threshold_99 = float(
        np.quantile(
            d_cal,
            0.99,
        )
    )

    threshold_995 = float(
        np.quantile(
            d_cal,
            0.995,
        )
    )

    #
    # AUROC: ID calibration rows = 0;
    # chronological unseen-state rows = 1.
    #
    ood_dist = np.concatenate(
        [
            d_val,
            d_test,
        ]
    )

    labels = np.concatenate(
        [
            np.zeros(
                len(d_cal),
                dtype=int,
            ),
            np.ones(
                len(ood_dist),
                dtype=int,
            ),
        ]
    )

    scores = np.concatenate(
        [
            d_cal,
            ood_dist,
        ]
    )

    auroc = float(
        roc_auc_score(
            labels,
            scores,
        )
    )

    result = {
        "name": name,
        "feature_count":
            len(feature_cols),
        "fit_rows":
            len(fit_df),
        "calibration_rows":
            len(calibration_df),
        "validation_rows":
            len(val_df),
        "test_rows":
            len(test_df),
        "auroc_id_vs_chronological_ood":
            auroc,
        "thresholds": {
            "p95": threshold_95,
            "p99": threshold_99,
            "p995": threshold_995,
        },
        "rates": {},
    }

    print()
    print("=" * 88)
    print(name)
    print("=" * 88)

    print(
        "features:",
        len(feature_cols),
    )

    print(
        "Ledoit-Wolf shrinkage:",
        f"{detector['model'].shrinkage_:.8f}",
    )

    print(
        "ID-vs-OOD AUROC:",
        f"{auroc:.6f}",
    )

    for label, d in [
        ("fit", d_fit),
        ("id_calibration", d_cal),
        ("chronological_validation", d_val),
        ("chronological_test", d_test),
    ]:
        print()
        print(label)

        print(
            pd.Series(d)
            .describe(
                percentiles=[
                    .50,
                    .90,
                    .95,
                    .99,
                ]
            )
            .to_string()
        )

    print()
    print("THRESHOLD PERFORMANCE")
    print("-" * 88)

    for threshold_name, threshold in [
        ("p95", threshold_95),
        ("p99", threshold_99),
        ("p995", threshold_995),
    ]:
        cal_flag = float(
            np.mean(
                d_cal > threshold
            )
        )

        val_flag = float(
            np.mean(
                d_val > threshold
            )
        )

        test_flag = float(
            np.mean(
                d_test > threshold
            )
        )

        result["rates"][
            threshold_name
        ] = {
            "threshold":
                threshold,
            "id_calibration_flag_rate":
                cal_flag,
            "chronological_validation_flag_rate":
                val_flag,
            "chronological_test_flag_rate":
                test_flag,
        }

        print(
            f"{threshold_name:5s} "
            f"threshold={threshold:.6f} "
            f"ID false-positive="
            f"{cal_flag:.2%} "
            f"val OOD flagged="
            f"{val_flag:.2%} "
            f"test OOD flagged="
            f"{test_flag:.2%}"
        )

    #
    # Per-anchor OOD distances.
    #
    combined = pd.concat(
        [
            calibration_df.assign(
                split="id_calibration"
            ),
            val_df.assign(
                split="chronological_validation"
            ),
            test_df.assign(
                split="chronological_test"
            ),
        ],
        ignore_index=True,
    )

    combined[
        "mahalanobis_distance"
    ] = distances(
        detector,
        combined,
        feature_cols,
    )

    combined[
        "flag_p99"
    ] = (
        combined[
            "mahalanobis_distance"
        ]
        > threshold_99
    )

    anchor_summary = (
        combined.groupby(
            [
                "split",
                "anchor_date",
            ]
        )
        .agg(
            rows=(
                "sample_id",
                "count",
            ),
            mean_distance=(
                "mahalanobis_distance",
                "mean",
            ),
            median_distance=(
                "mahalanobis_distance",
                "median",
            ),
            p95_distance=(
                "mahalanobis_distance",
                lambda x:
                    float(
                        np.quantile(
                            x,
                            .95,
                        )
                    ),
            ),
            p99_flag_rate=(
                "flag_p99",
                "mean",
            ),
        )
        .reset_index()
    )

    print()
    print("PER-ANCHOR P99 RESULTS")
    print("-" * 88)

    print(
        anchor_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    combined[
        [
            "sample_id",
            "anchor_date",
            "split",
            "mahalanobis_distance",
            "flag_p99",
        ]
    ].to_csv(
        OUTDIR
        / f"{safe_name}_row_scores.csv",
        index=False,
    )

    anchor_summary.to_csv(
        OUTDIR
        / f"{safe_name}_anchor_summary.csv",
        index=False,
    )

    return result


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = pd.read_csv(
        SPLIT_DIR / "train.csv"
    )

    val = pd.read_csv(
        SPLIT_DIR / "validation.csv"
    )

    test = pd.read_csv(
        SPLIT_DIR / "test.csv"
    )

    with open(
        SPLIT_DIR
        / "feature_columns.json"
    ) as f:
        feature_info = json.load(f)

    all_features = feature_info[
        "all_features"
    ]

    fit_df, calibration_df = (
        stratified_fit_calibration_split(
            train
        )
    )

    print("=" * 88)
    print("PDV MAHALANOBIS OOD GUARDRAIL")
    print("=" * 88)

    print(
        "chronological training rows:",
        len(train),
    )

    print(
        "detector fit rows:",
        len(fit_df),
    )

    print(
        "ID calibration rows:",
        len(calibration_df),
    )

    print(
        "chronological validation rows:",
        len(val),
    )

    print(
        "chronological test rows:",
        len(test),
    )

    print()
    print(
        "fit anchors:",
        fit_df[
            "anchor_date"
        ].nunique(),
    )

    print(
        "calibration anchors:",
        calibration_df[
            "anchor_date"
        ].nunique(),
    )

    results = {}

    results[
        "all_81_features"
    ] = summarize_detector(
        name="ALL 81 FEATURES",
        feature_cols=all_features,
        fit_df=fit_df,
        calibration_df=calibration_df,
        val_df=val,
        test_df=test,
    )

    results[
        "state_only_4_features"
    ] = summarize_detector(
        name="STATE ONLY 4 FEATURES",
        feature_cols=STATE_COLS,
        fit_df=fit_df,
        calibration_df=calibration_df,
        val_df=val,
        test_df=test,
    )

    with open(
        OUTDIR / "guardrail_summary.json",
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 88)
    print("COMPARISON")
    print("=" * 88)

    for key, result in results.items():
        p99 = result[
            "rates"
        ]["p99"]

        print(
            f"{key:24s} "
            f"AUROC="
            f"{result['auroc_id_vs_chronological_ood']:.6f} "
            f"ID_FP="
            f"{p99['id_calibration_flag_rate']:.2%} "
            f"VAL_RECALL="
            f"{p99['chronological_validation_flag_rate']:.2%} "
            f"TEST_RECALL="
            f"{p99['chronological_test_flag_rate']:.2%}"
        )

    print()
    print("Wrote:", OUTDIR)


if __name__ == "__main__":
    main()
