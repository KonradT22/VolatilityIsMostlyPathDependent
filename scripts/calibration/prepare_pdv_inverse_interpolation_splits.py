"""Build the interpolation train/validation/test split of the 9,667-row synthetic
inverse dataset, stratified by anchor date rather than chronologically. This is the
split behind the headline inverse-ANN test results (959-scenario test set)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 2026081804

BASE = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "inverse_ann/pdv_anchor_jitter_10k"
)

INPUT = BASE / "pdv_inverse_dataset_merged.csv"

OUTDIR = BASE / "splits_interpolation"

STATE_COLS = [
    "R1_fast",
    "R1_slow",
    "R2_fast",
    "R2_slow",
]

TARGET_COLS = [
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

    df = pd.read_csv(INPUT)

    price_cols = sorted(
        c
        for c in df.columns
        if c.startswith("price_dte")
    )

    feature_cols = price_cols + STATE_COLS

    if len(df) != 9667:
        raise RuntimeError(
            f"Expected 9667 rows, found {len(df)}"
        )

    if len(price_cols) != 77:
        raise RuntimeError(
            f"Expected 77 price features, found {len(price_cols)}"
        )

    if len(feature_cols) != 81:
        raise RuntimeError(
            f"Expected 81 total features, found {len(feature_cols)}"
        )

    check = df[
        feature_cols + TARGET_COLS
    ].to_numpy(dtype=float)

    if not np.isfinite(check).all():
        raise RuntimeError(
            "Found nonfinite feature/target values"
        )

    rng = np.random.default_rng(SEED)

    train_parts = []
    val_parts = []
    test_parts = []

    per_anchor = []

    for anchor in sorted(
        df["anchor_date"].astype(str).unique()
    ):
        group = (
            df[
                df["anchor_date"].astype(str)
                == anchor
            ]
            .copy()
            .reset_index(drop=True)
        )

        n = len(group)

        order = rng.permutation(n)

        n_val = int(
            np.floor(0.10 * n)
        )

        n_test = int(
            np.floor(0.10 * n)
        )

        n_train = (
            n - n_val - n_test
        )

        train_idx = (
            order[:n_train]
        )

        val_idx = (
            order[
                n_train:
                n_train + n_val
            ]
        )

        test_idx = (
            order[
                n_train + n_val:
            ]
        )

        train_part = (
            group.iloc[train_idx]
            .copy()
        )

        val_part = (
            group.iloc[val_idx]
            .copy()
        )

        test_part = (
            group.iloc[test_idx]
            .copy()
        )

        train_parts.append(
            train_part
        )

        val_parts.append(
            val_part
        )

        test_parts.append(
            test_part
        )

        per_anchor.append({
            "anchor_date": anchor,
            "total": n,
            "train": len(train_part),
            "validation": len(val_part),
            "test": len(test_part),
        })

    train = (
        pd.concat(
            train_parts,
            ignore_index=True,
        )
        .sample(
            frac=1.0,
            random_state=SEED,
        )
        .reset_index(drop=True)
    )

    val = (
        pd.concat(
            val_parts,
            ignore_index=True,
        )
        .sample(
            frac=1.0,
            random_state=SEED + 1,
        )
        .reset_index(drop=True)
    )

    test = (
        pd.concat(
            test_parts,
            ignore_index=True,
        )
        .sample(
            frac=1.0,
            random_state=SEED + 2,
        )
        .reset_index(drop=True)
    )

    train_ids = set(
        train["sample_id"].astype(int)
    )

    val_ids = set(
        val["sample_id"].astype(int)
    )

    test_ids = set(
        test["sample_id"].astype(int)
    )

    if train_ids & val_ids:
        raise RuntimeError(
            "Train/validation overlap"
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "Train/test overlap"
        )

    if val_ids & test_ids:
        raise RuntimeError(
            "Validation/test overlap"
        )

    union = (
        train_ids
        | val_ids
        | test_ids
    )

    if len(union) != len(df):
        raise RuntimeError(
            f"Assigned {len(union)} "
            f"of {len(df)} rows"
        )

    for name, split in [
        ("train", train),
        ("validation", val),
        ("test", test),
    ]:
        anchors = set(
            split["anchor_date"]
            .astype(str)
            .unique()
        )

        if len(anchors) != 16:
            raise RuntimeError(
                f"{name} contains "
                f"{len(anchors)} anchors, "
                "expected 16"
            )

    train.to_csv(
        OUTDIR / "train.csv",
        index=False,
    )

    val.to_csv(
        OUTDIR / "validation.csv",
        index=False,
    )

    test.to_csv(
        OUTDIR / "test.csv",
        index=False,
    )

    per_anchor_df = pd.DataFrame(
        per_anchor
    )

    per_anchor_df.to_csv(
        OUTDIR / "per_anchor_counts.csv",
        index=False,
    )

    with open(
        OUTDIR / "feature_columns.json",
        "w",
    ) as f:
        json.dump(
            {
                "price_features":
                    price_cols,
                "state_features":
                    STATE_COLS,
                "all_features":
                    feature_cols,
                "targets":
                    TARGET_COLS,
            },
            f,
            indent=2,
        )

    summary = {
        "split_type":
            "within_anchor_interpolation",
        "seed":
            SEED,
        "total_rows":
            len(df),
        "train_rows":
            len(train),
        "validation_rows":
            len(val),
        "test_rows":
            len(test),
        "anchor_count":
            16,
        "anchors_in_each_split":
            True,
        "price_features":
            77,
        "state_features":
            4,
        "total_features":
            81,
        "targets":
            5,
    }

    with open(
        OUTDIR / "split_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("=" * 78)
    print("PDV INVERSE ANN WITHIN-ANCHOR INTERPOLATION SPLIT")
    print("=" * 78)
    print("total:", len(df))
    print(
        "train:",
        len(train),
        "rows",
    )
    print(
        "validation:",
        len(val),
        "rows",
    )
    print(
        "test:",
        len(test),
        "rows",
    )
    print()
    print(
        "anchors in train:",
        train["anchor_date"].nunique(),
    )
    print(
        "anchors in validation:",
        val["anchor_date"].nunique(),
    )
    print(
        "anchors in test:",
        test["anchor_date"].nunique(),
    )
    print()
    print("features:", len(feature_cols))
    print("targets:", len(TARGET_COLS))
    print()
    print("PER-ANCHOR COUNTS")
    print(
        per_anchor_df.to_string(
            index=False
        )
    )
    print()
    print("Wrote:", OUTDIR)


if __name__ == "__main__":
    main()
