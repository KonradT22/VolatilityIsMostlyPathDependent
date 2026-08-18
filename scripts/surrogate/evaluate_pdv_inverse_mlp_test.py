import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.surrogate.train_pdv_inverse_mlp import InverseMLP


SPLIT_DIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "inverse_ann/pdv_anchor_jitter_10k/"
    "splits_interpolation"
)

MODEL_DIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "inverse_ann/pdv_anchor_jitter_10k/"
    "models/mlp_interpolation"
)

OUTDIR = MODEL_DIR / "test_evaluation"

CHECKPOINT = MODEL_DIR / "best_model.pt"

TARGETS = [
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


def metrics(y, p, y_std):
    result = {}

    standardized_error = (
        p - y
    ) / y_std

    result["aggregate_standardized_rmse"] = float(
        np.sqrt(
            np.mean(
                standardized_error ** 2
            )
        )
    )

    per_target = {}

    for j, name in enumerate(TARGETS):
        truth = y[:, j]
        pred = p[:, j]

        error = pred - truth

        rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        mae = float(
            np.mean(
                np.abs(error)
            )
        )

        ss_res = float(
            np.sum(
                error ** 2
            )
        )

        ss_tot = float(
            np.sum(
                (truth - truth.mean()) ** 2
            )
        )

        r2 = (
            float(1.0 - ss_res / ss_tot)
            if ss_tot > 0
            else float("nan")
        )

        low, high = BOUNDS[name]

        per_target[name] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "range_normalized_rmse": float(
                rmse / (high - low)
            ),
            "pred_min": float(pred.min()),
            "pred_max": float(pred.max()),
        }

    result["per_target"] = per_target

    return result


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    test = pd.read_csv(
        SPLIT_DIR / "test.csv"
    )

    with open(
        SPLIT_DIR / "feature_columns.json"
    ) as f:
        feature_info = json.load(f)

    features = feature_info[
        "all_features"
    ]

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint[
        "feature_columns"
    ] != features:
        raise RuntimeError(
            "Checkpoint feature order "
            "does not match test split"
        )

    if checkpoint[
        "target_columns"
    ] != TARGETS:
        raise RuntimeError(
            "Checkpoint target order mismatch"
        )

    x_mean = np.asarray(
        checkpoint["x_mean"],
        dtype=float,
    )

    x_std = np.asarray(
        checkpoint["x_std"],
        dtype=float,
    )

    y_mean = np.asarray(
        checkpoint["y_mean"],
        dtype=float,
    )

    y_std = np.asarray(
        checkpoint["y_std"],
        dtype=float,
    )

    x = test[
        features
    ].to_numpy(dtype=float)

    y = test[
        TARGETS
    ].to_numpy(dtype=float)

    if not np.isfinite(x).all():
        raise RuntimeError(
            "Nonfinite test features"
        )

    if not np.isfinite(y).all():
        raise RuntimeError(
            "Nonfinite test targets"
        )

    x_z = (
        x - x_mean
    ) / x_std

    x_tensor = torch.tensor(
        x_z,
        dtype=torch.float32,
    )

    model = InverseMLP(
        input_dim=len(features)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    with torch.no_grad():
        pred_z = (
            model(x_tensor)
            .cpu()
            .numpy()
        )

    raw = (
        pred_z * y_std
        + y_mean
    )

    clipped = raw.copy()

    violation_rows = np.zeros(
        len(test),
        dtype=bool,
    )

    violation_summary = {}

    for j, name in enumerate(
        TARGETS
    ):
        low, high = BOUNDS[name]

        below = raw[:, j] < low
        above = raw[:, j] > high

        violation_rows |= (
            below | above
        )

        amount = np.maximum(
            np.where(
                below,
                low - raw[:, j],
                0.0,
            ),
            np.where(
                above,
                raw[:, j] - high,
                0.0,
            ),
        )

        nonzero = amount[
            amount > 0
        ]

        violation_summary[name] = {
            "below_count":
                int(below.sum()),
            "above_count":
                int(above.sum()),
            "violation_count":
                int(
                    (below | above).sum()
                ),
            "max_violation":
                float(
                    nonzero.max()
                )
                if len(nonzero)
                else 0.0,
        }

        clipped[:, j] = np.clip(
            clipped[:, j],
            low,
            high,
        )

    raw_metrics = metrics(
        y,
        raw,
        y_std,
    )

    clipped_metrics = metrics(
        y,
        clipped,
        y_std,
    )

    #
    # CPU inference timing.
    #
    torch.set_num_threads(2)

    with torch.no_grad():
        for _ in range(20):
            model(x_tensor)

        timings = []

        for _ in range(100):
            start = time.perf_counter()

            model(x_tensor)

            timings.append(
                time.perf_counter()
                - start
            )

    median_batch_seconds = float(
        np.median(timings)
    )

    median_per_sample_seconds = (
        median_batch_seconds
        / len(test)
    )

    predictions = test[
        [
            "sample_id",
            "anchor_date",
        ]
    ].copy()

    for j, name in enumerate(
        TARGETS
    ):
        predictions[
            f"true_{name}"
        ] = y[:, j]

        predictions[
            f"raw_{name}"
        ] = raw[:, j]

        predictions[
            f"clipped_{name}"
        ] = clipped[:, j]

        predictions[
            f"clipped_error_{name}"
        ] = (
            clipped[:, j]
            - y[:, j]
        )

    predictions[
        "any_bound_violation"
    ] = violation_rows

    predictions.to_csv(
        OUTDIR / "test_predictions.csv",
        index=False,
    )

    #
    # Per-anchor clipped accuracy.
    #
    anchor_rows = []

    for anchor, idx in (
        predictions.groupby(
            "anchor_date"
        ).groups.items()
    ):
        idx = np.asarray(
            list(idx),
            dtype=int,
        )

        m = metrics(
            y[idx],
            clipped[idx],
            y_std,
        )

        anchor_rows.append({
            "anchor_date":
                str(anchor),
            "rows":
                len(idx),
            "standardized_rmse":
                m[
                    "aggregate_standardized_rmse"
                ],
            **{
                f"{name}_rmse":
                    m[
                        "per_target"
                    ][name]["rmse"]
                for name in TARGETS
            },
        })

    anchor_df = pd.DataFrame(
        anchor_rows
    )

    anchor_df.to_csv(
        OUTDIR
        / "test_metrics_by_anchor.csv",
        index=False,
    )

    summary = {
        "model":
            "81-256-256-128-5",
        "checkpoint_best_epoch":
            int(
                checkpoint[
                    "best_epoch"
                ]
            ),
        "test_rows":
            len(test),
        "anchors":
            int(
                test[
                    "anchor_date"
                ].nunique()
            ),
        "raw_metrics":
            raw_metrics,
        "clipped_metrics":
            clipped_metrics,
        "bound_violations":
            violation_summary,
        "rows_with_any_bound_violation":
            int(
                violation_rows.sum()
            ),
        "row_bound_violation_rate":
            float(
                violation_rows.mean()
            ),
        "inference_timing": {
            "median_batch_seconds":
                median_batch_seconds,
            "median_per_sample_seconds":
                median_per_sample_seconds,
            "median_per_sample_ms":
                median_per_sample_seconds
                * 1000.0,
        },
    }

    with open(
        OUTDIR / "test_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("=" * 88)
    print("PDV INVERSE ANN — FINAL INTERPOLATION TEST")
    print("=" * 88)

    print(
        "test rows:",
        len(test),
    )

    print(
        "anchors:",
        test[
            "anchor_date"
        ].nunique(),
    )

    print(
        "checkpoint best epoch:",
        checkpoint[
            "best_epoch"
        ],
    )

    print()
    print("STANDARDIZED AGGREGATE RMSE")
    print("-" * 88)

    print(
        "raw:",
        f"{raw_metrics['aggregate_standardized_rmse']:.8f}",
    )

    print(
        "clipped:",
        f"{clipped_metrics['aggregate_standardized_rmse']:.8f}",
    )

    print()
    print("CLIPPED TEST PARAMETER METRICS")
    print("-" * 88)

    for name in TARGETS:
        m = (
            clipped_metrics[
                "per_target"
            ][name]
        )

        print(
            f"{name:7s} "
            f"RMSE={m['rmse']:.8f} "
            f"MAE={m['mae']:.8f} "
            f"R2={m['r2']:.6f} "
            f"rangeNRMSE="
            f"{m['range_normalized_rmse']:.6f}"
        )

    print()
    print("BOUND VIOLATIONS BEFORE PROJECTION")
    print("-" * 88)

    for name in TARGETS:
        v = violation_summary[name]

        print(
            f"{name:7s} "
            f"below={v['below_count']:4d} "
            f"above={v['above_count']:4d} "
            f"max_amount="
            f"{v['max_violation']:.8f}"
        )

    print(
        "\nrows with >=1 violation:",
        int(violation_rows.sum()),
        "/",
        len(test),
        f"({violation_rows.mean():.2%})",
    )

    print()
    print("CPU INFERENCE")
    print("-" * 88)

    print(
        "median full-test batch:",
        f"{median_batch_seconds * 1000:.4f} ms",
    )

    print(
        "median per scenario:",
        f"{median_per_sample_seconds * 1000:.6f} ms",
    )

    print()
    print("PER-ANCHOR CLIPPED STANDARDIZED RMSE")
    print("-" * 88)

    print(
        anchor_df[
            [
                "anchor_date",
                "rows",
                "standardized_rmse",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.8f}",
        )
    )

    print()
    print(
        "Wrote:",
        OUTDIR,
    )


if __name__ == "__main__":
    main()
