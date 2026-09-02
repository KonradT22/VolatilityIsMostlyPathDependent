"""Build the chronological train/validation/test split of the 9,667-row synthetic
inverse dataset: 12 earliest anchor dates train, 2 middle dates (2021-05-27/28)
validation, 2 latest dates (2021-06-01/02) test. Used to test generalization to
later, unseen market states and to fit/evaluate the Mahalanobis OOD guardrail."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "inverse_ann/pdv_anchor_jitter_10k"
)

INPUT = (
    BASE / "pdv_inverse_dataset_merged.csv"
)

OUTDIR = (
    BASE / "splits"
)

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

TRAIN_ANCHORS = [
    "2021-05-05",
    "2021-05-06",
    "2021-05-07",
    "2021-05-10",
    "2021-05-11",
    "2021-05-12",
    "2021-05-13",
    "2021-05-14",
    "2021-05-17",
    "2021-05-18",
    "2021-05-19",
    "2021-05-20",
]

VAL_ANCHORS = [
    "2021-05-27",
    "2021-05-28",
]

TEST_ANCHORS = [
    "2021-06-01",
    "2021-06-02",
]


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT)

    price_cols = sorted(
        [
            c
            for c in df.columns
            if c.startswith("price_dte")
        ]
    )

    if len(df) != 9667:
        raise RuntimeError(
            f"Expected 9667 valid rows, found {len(df)}"
        )

    if len(price_cols) != 77:
        raise RuntimeError(
            f"Expected 77 price columns, found {len(price_cols)}"
        )

    feature_cols = (
        price_cols
        + STATE_COLS
    )

    if len(feature_cols) != 81:
        raise RuntimeError(
            f"Expected 81 ANN features, found {len(feature_cols)}"
        )

    values = df[
        feature_cols + TARGET_COLS
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Nonfinite feature/target values found"
        )

    train = df[
        df["anchor_date"].isin(
            TRAIN_ANCHORS
        )
    ].copy()

    val = df[
        df["anchor_date"].isin(
            VAL_ANCHORS
        )
    ].copy()

    test = df[
        df["anchor_date"].isin(
            TEST_ANCHORS
        )
    ].copy()

    assigned = set(
        train["sample_id"]
    ) | set(
        val["sample_id"]
    ) | set(
        test["sample_id"]
    )

    if len(assigned) != len(df):
        raise RuntimeError(
            f"Split assigned {len(assigned)} "
            f"of {len(df)} rows"
        )

    if (
        set(train["sample_id"])
        & set(val["sample_id"])
    ):
        raise RuntimeError(
            "Train/validation overlap"
        )

    if (
        set(train["sample_id"])
        & set(test["sample_id"])
    ):
        raise RuntimeError(
            "Train/test overlap"
        )

    if (
        set(val["sample_id"])
        & set(test["sample_id"])
    ):
        raise RuntimeError(
            "Validation/test overlap"
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
        "total_rows":
            int(len(df)),
        "price_feature_count":
            len(price_cols),
        "state_feature_count":
            len(STATE_COLS),
        "total_feature_count":
            len(feature_cols),
        "target_count":
            len(TARGET_COLS),
        "train": {
            "rows":
                int(len(train)),
            "anchors":
                TRAIN_ANCHORS,
        },
        "validation": {
            "rows":
                int(len(val)),
            "anchors":
                VAL_ANCHORS,
        },
        "test": {
            "rows":
                int(len(test)),
            "anchors":
                TEST_ANCHORS,
        },
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
    print("PDV INVERSE ANN SPLIT")
    print("=" * 78)
    print("total:", len(df))
    print()
    print(
        "train:",
        len(train),
        "rows /",
        train["anchor_date"].nunique(),
        "anchors",
    )
    print(
        "validation:",
        len(val),
        "rows /",
        val["anchor_date"].nunique(),
        "anchors",
    )
    print(
        "test:",
        len(test),
        "rows /",
        test["anchor_date"].nunique(),
        "anchors",
    )
    print()
    print("features:", len(feature_cols))
    print("  option prices:", len(price_cols))
    print("  state values:", len(STATE_COLS))
    print("targets:", len(TARGET_COLS))

    print("\nTRAIN ANCHORS")
    for x in TRAIN_ANCHORS:
        print(" ", x)

    print("\nVALIDATION ANCHORS")
    for x in VAL_ANCHORS:
        print(" ", x)

    print("\nTEST ANCHORS")
    for x in TEST_ANCHORS:
        print(" ", x)

    print("\nWrote:", OUTDIR)


if __name__ == "__main__":
    main()
