"""Leave-one-anchor-out diagnostic for the state-only (4-D) OOD detector: score each
training-era anchor state as if it were held out, to build an in-family reference
distribution, then compare later chronological states against it as a robustness
check on the Mahalanobis OOD guardrail's chronological-generalization claim."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


BASE = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "inverse_ann/pdv_anchor_jitter_10k"
)

SPLIT_DIR = BASE / "splits"

OUTDIR = (
    BASE
    / "guardrails"
    / "state_leave_one_anchor_out"
)

STATE_COLS = [
    "R1_fast",
    "R1_slow",
    "R2_fast",
    "R2_slow",
]


def unique_anchor_states(df):
    x = (
        df[
            ["anchor_date"] + STATE_COLS
        ]
        .drop_duplicates()
        .sort_values("anchor_date")
        .reset_index(drop=True)
    )

    counts = (
        df.groupby("anchor_date")
        .size()
    )

    if len(x) != df["anchor_date"].nunique():
        raise RuntimeError(
            "Expected exactly one state vector per anchor"
        )

    return x


def fit_state_detector(states):
    x = states[
        STATE_COLS
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

    model = LedoitWolf().fit(z)

    return mean, std, model


def score(mean, std, model, row):
    x = np.asarray(
        [
            row[c]
            for c in STATE_COLS
        ],
        dtype=float,
    )

    z = (
        x - mean
    ) / std

    return float(
        model.mahalanobis(
            z.reshape(1, -1)
        )[0]
    )


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

    train_states = unique_anchor_states(
        train
    )

    val_states = unique_anchor_states(
        val
    )

    test_states = unique_anchor_states(
        test
    )

    if len(train_states) != 12:
        raise RuntimeError(
            f"Expected 12 training states, "
            f"found {len(train_states)}"
        )

    print("=" * 88)
    print("PDV STATE GUARDRAIL — LEAVE-ONE-ANCHOR-OUT")
    print("=" * 88)

    print(
        "training anchor states:",
        len(train_states),
    )

    print(
        "chronological validation states:",
        len(val_states),
    )

    print(
        "chronological test states:",
        len(test_states),
    )

    #
    # Each historical training anchor is treated as if it
    # were an unseen state.
    #
    loo_rows = []

    for i in range(
        len(train_states)
    ):
        held = (
            train_states
            .iloc[i]
        )

        fit = (
            train_states
            .drop(index=i)
            .reset_index(drop=True)
        )

        mean, std, model = (
            fit_state_detector(fit)
        )

        d = score(
            mean,
            std,
            model,
            held,
        )

        loo_rows.append({
            "anchor_date":
                str(
                    held[
                        "anchor_date"
                    ]
                ),
            "loo_distance":
                d,
            "fit_anchor_count":
                len(fit),
            "shrinkage":
                float(
                    model.shrinkage_
                ),
        })

    loo = pd.DataFrame(
        loo_rows
    )

    #
    # Final detector fitted to all 12 earlier states.
    #
    mean, std, model = (
        fit_state_detector(
            train_states
        )
    )

    future_rows = []

    for split_name, states in [
        (
            "chronological_validation",
            val_states,
        ),
        (
            "chronological_test",
            test_states,
        ),
    ]:
        for _, row in states.iterrows():
            d = score(
                mean,
                std,
                model,
                row,
            )

            future_rows.append({
                "split":
                    split_name,
                "anchor_date":
                    str(
                        row[
                            "anchor_date"
                        ]
                    ),
                "distance":
                    d,
            })

    future = pd.DataFrame(
        future_rows
    )

    max_loo = float(
        loo["loo_distance"].max()
    )

    median_loo = float(
        loo["loo_distance"].median()
    )

    print()
    print("LEAVE-ONE-ANCHOR-OUT TRAINING-ERA DISTANCES")
    print("-" * 88)

    print(
        loo.sort_values(
            "loo_distance"
        ).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print("LOAO SUMMARY")
    print("-" * 88)

    print(
        loo["loo_distance"]
        .describe(
            percentiles=[
                .50,
                .75,
                .90,
                .95,
            ]
        )
        .to_string()
    )

    print()
    print(
        "maximum LOAO distance:",
        f"{max_loo:.6f}",
    )

    print()
    print("LATER CHRONOLOGICAL STATES")
    print("-" * 88)

    future[
        "multiple_of_max_loo"
    ] = (
        future["distance"]
        / max_loo
    )

    future[
        "multiple_of_median_loo"
    ] = (
        future["distance"]
        / median_loo
    )

    print(
        future.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print(
        "Final 12-state Ledoit-Wolf shrinkage:",
        f"{model.shrinkage_:.8f}",
    )

    print()
    print(
        "future states above maximum "
        "training-era LOAO distance:",
        int(
            (
                future["distance"]
                > max_loo
            ).sum()
        ),
        "/",
        len(future),
    )

    loo.to_csv(
        OUTDIR
        / "training_anchor_loo_scores.csv",
        index=False,
    )

    future.to_csv(
        OUTDIR
        / "future_state_scores.csv",
        index=False,
    )

    summary = {
        "training_unique_states":
            len(train_states),
        "future_unique_states":
            len(future),
        "max_training_loo_distance":
            max_loo,
        "median_training_loo_distance":
            median_loo,
        "future_above_max_loo":
            int(
                (
                    future["distance"]
                    > max_loo
                ).sum()
            ),
        "future_count":
            len(future),
        "final_detector_shrinkage":
            float(
                model.shrinkage_
            ),
    }

    with open(
        OUTDIR / "summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("Wrote:", OUTDIR)


if __name__ == "__main__":
    main()
