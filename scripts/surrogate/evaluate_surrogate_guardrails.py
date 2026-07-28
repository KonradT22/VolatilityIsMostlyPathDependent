import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class SurfaceSurrogate(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.network(x)


def target_sort_key(column):
    match = re.fullmatch(r"iv_dte_(\d+)_m(\d+)", column)
    if match is None:
        raise ValueError(f"Unexpected target column: {column}")
    return int(match.group(1)), int(match.group(2))


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = SurfaceSurrogate(
        input_size=checkpoint["input_size"],
        output_size=checkpoint["output_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict_surface(model, checkpoint, raw_features):
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
    y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)

    scaled = (raw_features - x_mean) / x_std

    tensor = torch.tensor(
        scaled.reshape(1, -1),
        dtype=torch.float32,
    )

    with torch.no_grad():
        predicted_scaled = model(tensor).cpu().numpy()[0]

    prediction = predicted_scaled * y_std + y_mean
    return prediction, scaled


def check_surface_shape(prediction, target_columns):
    grouped = {}

    for index, column in enumerate(target_columns):
        dte, moneyness_index = target_sort_key(column)
        grouped.setdefault(dte, []).append(
            (moneyness_index, float(prediction[index]))
        )

    violations = []

    for dte, values in grouped.items():
        values = sorted(values)
        iv = np.asarray([value for _, value in values])

        if not np.all(np.isfinite(iv)):
            violations.append(f"dte_{dte}_nonfinite")
            continue

        if np.any(iv < 0.005):
            violations.append(f"dte_{dte}_iv_below_floor")

        if np.any(iv > 1.50):
            violations.append(f"dte_{dte}_iv_above_cap")

        first_difference = np.diff(iv)
        second_difference = np.diff(iv, n=2)

        if np.max(np.abs(first_difference)) > 0.08:
            violations.append(f"dte_{dte}_large_adjacent_jump")

        if np.min(second_difference) < -0.04:
            violations.append(f"dte_{dte}_strong_local_concavity")

    return violations


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=(
            "/users/4/trest017/urop_pdv/data/processed/"
            "surrogate/pilot_surface_dataset/pdv_surface_dataset.csv"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "/users/4/trest017/urop_pdv/benchmarks/"
            "surrogate/pilot_surface_ann/pdv_surface_surrogate.pt"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "/users/4/trest017/urop_pdv/benchmarks/"
            "surrogate/pilot_guardrails"
        ),
    )
    parser.add_argument("--ood-z-threshold", type=float, default=3.0)
    parser.add_argument("--error-threshold", type=float, default=0.02)

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    checkpoint_path = Path(args.checkpoint)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(dataset_path)
    model, checkpoint = load_model(checkpoint_path)

    feature_columns = checkpoint["feature_columns"]
    target_columns = sorted(
        checkpoint["target_columns"],
        key=target_sort_key,
    )

    test = (
        frame[frame["split"] == "test"]
        .sort_values("case_id")
        .reset_index(drop=True)
    )

    replay_rows = []
    prediction_latencies = []

    for sequence_index, row in test.iterrows():
        raw_features = row[feature_columns].to_numpy(dtype=np.float32)
        actual = row[target_columns].to_numpy(dtype=np.float32)

        start = time.perf_counter()
        prediction, scaled_features = predict_surface(
            model,
            checkpoint,
            raw_features,
        )
        latency_microseconds = (
            time.perf_counter() - start
        ) * 1e6

        prediction_latencies.append(latency_microseconds)

        max_abs_z = float(np.max(np.abs(scaled_features)))
        ood_reasons = []

        if max_abs_z > args.ood_z_threshold:
            ood_reasons.append("feature_zscore")

        surface_reasons = check_surface_shape(
            prediction,
            target_columns,
        )

        error = prediction - actual
        rmse = float(np.sqrt(np.mean(error ** 2)))
        mae = float(np.mean(np.abs(error)))
        max_abs_error = float(np.max(np.abs(error)))

        fallback_reasons = ood_reasons + surface_reasons
        fallback_triggered = len(fallback_reasons) > 0

        large_error = max_abs_error > args.error_threshold

        replay_rows.append({
            "sequence_index": sequence_index,
            "case_id": int(row["case_id"]),
            "latency_microseconds": latency_microseconds,
            "max_abs_feature_z": max_abs_z,
            "ood_triggered": len(ood_reasons) > 0,
            "surface_guardrail_triggered": len(surface_reasons) > 0,
            "fallback_triggered": fallback_triggered,
            "fallback_reasons": "|".join(fallback_reasons),
            "rmse_iv": rmse,
            "mae_iv": mae,
            "max_abs_error_iv": max_abs_error,
            "large_error": large_error,
            "large_error_caught": bool(
                large_error and fallback_triggered
            ),
        })

    replay = pd.DataFrame(replay_rows)

    large_error_count = int(replay["large_error"].sum())
    caught_count = int(replay["large_error_caught"].sum())

    guardrail_recall = (
        caught_count / large_error_count
        if large_error_count > 0
        else None
    )

    non_large = ~replay["large_error"]
    false_positive_count = int(
        (
            replay["fallback_triggered"]
            & non_large
        ).sum()
    )

    non_large_count = int(non_large.sum())
    false_positive_rate = (
        false_positive_count / non_large_count
        if non_large_count > 0
        else None
    )

    summary = {
        "test_surfaces": len(replay),
        "ood_z_threshold": args.ood_z_threshold,
        "large_error_threshold_iv": args.error_threshold,
        "fallback_count": int(
            replay["fallback_triggered"].sum()
        ),
        "fallback_rate": float(
            replay["fallback_triggered"].mean()
        ),
        "ood_count": int(replay["ood_triggered"].sum()),
        "surface_guardrail_count": int(
            replay["surface_guardrail_triggered"].sum()
        ),
        "large_error_count": large_error_count,
        "large_error_caught_count": caught_count,
        "large_error_guardrail_recall": guardrail_recall,
        "false_positive_count": false_positive_count,
        "false_positive_rate": false_positive_rate,
        "mean_rmse_iv": float(replay["rmse_iv"].mean()),
        "mean_mae_iv": float(replay["mae_iv"].mean()),
        "max_abs_error_iv": float(
            replay["max_abs_error_iv"].max()
        ),
        "mean_latency_microseconds": float(
            np.mean(prediction_latencies)
        ),
        "median_latency_microseconds": float(
            np.median(prediction_latencies)
        ),
        "p95_latency_microseconds": float(
            np.quantile(prediction_latencies, 0.95)
        ),
    }

    replay_path = outdir / "sequential_replay.csv"
    summary_path = outdir / "guardrail_summary.json"

    replay.to_csv(replay_path, index=False)

    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("Sequential replay complete.")
    print(json.dumps(summary, indent=2))

    print()
    print("Largest-error replay rows:")
    print(
        replay.sort_values(
            "max_abs_error_iv",
            ascending=False,
        ).head(10)
    )

    print()
    print("Wrote:", replay_path)
    print("Wrote:", summary_path)


if __name__ == "__main__":
    main()
