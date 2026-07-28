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


def load_ensemble(ensemble_dir):
    checkpoint_paths = sorted(
        ensemble_dir.glob("model_*/pdv_surface_surrogate.pt")
    )

    if len(checkpoint_paths) != 5:
        raise ValueError(
            f"Expected 5 checkpoints, found {len(checkpoint_paths)}"
        )

    ensemble = []

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model = SurfaceSurrogate(
            input_size=checkpoint["input_size"],
            output_size=checkpoint["output_size"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        ensemble.append((model, checkpoint, checkpoint_path))

    reference_features = ensemble[0][1]["feature_columns"]
    reference_targets = ensemble[0][1]["target_columns"]

    for _, checkpoint, path in ensemble[1:]:
        if checkpoint["feature_columns"] != reference_features:
            raise ValueError(f"Feature mismatch in {path}")

        if checkpoint["target_columns"] != reference_targets:
            raise ValueError(f"Target mismatch in {path}")

    return ensemble


def predict_batch(model, checkpoint, raw_features):
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float32)
    y_std = np.asarray(checkpoint["y_std"], dtype=np.float32)

    scaled = (raw_features - x_mean) / x_std
    tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.inference_mode():
        prediction_scaled = model(tensor).cpu().numpy()

    prediction = prediction_scaled * y_std + y_mean
    return prediction, scaled


def predict_one(ensemble, raw_features):
    predictions = []

    with torch.inference_mode():
        for model, checkpoint, _ in ensemble:
            x_mean = np.asarray(
                checkpoint["x_mean"],
                dtype=np.float32,
            )
            x_std = np.asarray(
                checkpoint["x_std"],
                dtype=np.float32,
            )
            y_mean = np.asarray(
                checkpoint["y_mean"],
                dtype=np.float32,
            )
            y_std = np.asarray(
                checkpoint["y_std"],
                dtype=np.float32,
            )

            scaled = (raw_features - x_mean) / x_std

            tensor = torch.tensor(
                scaled.reshape(1, -1),
                dtype=torch.float32,
            )

            prediction_scaled = model(tensor).cpu().numpy()[0]
            prediction = prediction_scaled * y_std + y_mean
            predictions.append(prediction)

    stacked = np.stack(predictions, axis=0)

    return stacked.mean(axis=0), stacked.std(axis=0)


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


def evaluate_split(
    split_frame,
    ensemble,
    feature_columns,
    target_columns,
    ood_z_threshold,
    large_error_threshold,
):
    raw_features = split_frame[
        feature_columns
    ].to_numpy(dtype=np.float32)

    actual = split_frame[
        target_columns
    ].to_numpy(dtype=np.float32)

    model_predictions = []
    reference_scaled = None

    for index, (model, checkpoint, _) in enumerate(ensemble):
        prediction, scaled = predict_batch(
            model,
            checkpoint,
            raw_features,
        )

        model_predictions.append(prediction)

        if index == 0:
            reference_scaled = scaled

    stacked = np.stack(model_predictions, axis=0)

    ensemble_prediction = stacked.mean(axis=0)
    ensemble_std = stacked.std(axis=0)

    error = ensemble_prediction - actual
    absolute_error = np.abs(error)

    rows = []

    for index in range(len(split_frame)):
        surface_reasons = check_surface_shape(
            ensemble_prediction[index],
            target_columns,
        )

        max_abs_z = float(
            np.max(np.abs(reference_scaled[index]))
        )

        ood_triggered = max_abs_z > ood_z_threshold

        base_reasons = []

        if ood_triggered:
            base_reasons.append("feature_zscore")

        base_reasons.extend(surface_reasons)

        rmse = float(
            np.sqrt(np.mean(error[index] ** 2))
        )
        mae = float(np.mean(absolute_error[index]))
        max_abs_error = float(np.max(absolute_error[index]))

        rows.append({
            "case_id": int(split_frame.iloc[index]["case_id"]),
            "rmse_iv": rmse,
            "mae_iv": mae,
            "max_abs_error_iv": max_abs_error,
            "large_error": (
                max_abs_error > large_error_threshold
            ),
            "max_abs_feature_z": max_abs_z,
            "ood_triggered": ood_triggered,
            "surface_guardrail_triggered": (
                len(surface_reasons) > 0
            ),
            "base_rule_triggered": len(base_reasons) > 0,
            "base_reasons": "|".join(base_reasons),
            "max_ensemble_std_iv": float(
                np.max(ensemble_std[index])
            ),
            "mean_ensemble_std_iv": float(
                np.mean(ensemble_std[index])
            ),
            "p95_ensemble_std_iv": float(
                np.quantile(ensemble_std[index], 0.95)
            ),
        })

    return pd.DataFrame(rows)


def guardrail_metrics(frame, fallback):
    large_error = frame["large_error"].to_numpy(dtype=bool)
    fallback = np.asarray(fallback, dtype=bool)

    true_positive = int(np.sum(fallback & large_error))
    false_positive = int(np.sum(fallback & ~large_error))
    false_negative = int(np.sum(~fallback & large_error))

    fallback_count = int(np.sum(fallback))
    positive_count = int(np.sum(large_error))
    negative_count = int(np.sum(~large_error))

    recall = (
        true_positive / positive_count
        if positive_count > 0
        else 0.0
    )

    precision = (
        true_positive / fallback_count
        if fallback_count > 0
        else 0.0
    )

    false_positive_rate = (
        false_positive / negative_count
        if negative_count > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(frame),
        "recall": recall,
        "precision": precision,
        "false_positive_rate": false_positive_rate,
        "f1": f1,
    }


def calibrate_threshold(validation, max_fallback_rate):
    scores = validation[
        "max_ensemble_std_iv"
    ].to_numpy(dtype=float)

    base_rule = validation[
        "base_rule_triggered"
    ].to_numpy(dtype=bool)

    epsilon = 1e-12

    candidates = np.unique(
        np.concatenate([
            scores,
            [scores.min() - epsilon],
            [scores.max() + epsilon],
        ])
    )

    results = []

    for threshold in candidates:
        fallback = base_rule | (scores >= threshold)
        metrics = guardrail_metrics(validation, fallback)

        results.append({
            "threshold": float(threshold),
            **metrics,
        })

    table = pd.DataFrame(results)

    feasible = table[
        table["fallback_rate"] <= max_fallback_rate
    ].copy()

    if feasible.empty:
        feasible = table.copy()

    feasible = feasible.sort_values(
        [
            "recall",
            "precision",
            "f1",
            "fallback_rate",
        ],
        ascending=[False, False, False, True],
    )

    selected = feasible.iloc[0].to_dict()

    return float(selected["threshold"]), table


def add_final_guardrail(frame, threshold):
    output = frame.copy()

    output["uncertainty_triggered"] = (
        output["max_ensemble_std_iv"] >= threshold
    )

    output["fallback_triggered"] = (
        output["base_rule_triggered"]
        | output["uncertainty_triggered"]
    )

    output["fallback_reasons"] = output.apply(
        lambda row: "|".join(
            reason
            for reason in [
                row["base_reasons"],
                (
                    "ensemble_disagreement"
                    if row["uncertainty_triggered"]
                    else ""
                ),
            ]
            if reason
        ),
        axis=1,
    )

    output["large_error_caught"] = (
        output["large_error"]
        & output["fallback_triggered"]
    )

    return output


def measure_sequential_latency(test, ensemble, feature_columns):
    raw_features = test[
        feature_columns
    ].to_numpy(dtype=np.float32)

    for _ in range(50):
        predict_one(ensemble, raw_features[0])

    latencies = []

    for row in raw_features:
        start = time.perf_counter()
        predict_one(ensemble, row)

        latencies.append(
            (time.perf_counter() - start) * 1e6
        )

    return np.asarray(latencies)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=(
            "/users/4/trest017/urop_pdv/data/processed/"
            "surrogate/pilot_surface_dataset/"
            "pdv_surface_dataset.csv"
        ),
    )

    parser.add_argument(
        "--ensemble-dir",
        default=(
            "/users/4/trest017/urop_pdv/benchmarks/"
            "surrogate/pilot_surface_ann_ensemble"
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "/users/4/trest017/urop_pdv/benchmarks/"
            "surrogate/pilot_ensemble_guardrails"
        ),
    )

    parser.add_argument(
        "--ood-z-threshold",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--large-error-threshold",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--max-validation-fallback-rate",
        type=float,
        default=0.20,
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    ensemble_dir = Path(args.ensemble_dir)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(dataset_path)
    ensemble = load_ensemble(ensemble_dir)

    feature_columns = ensemble[0][1]["feature_columns"]

    target_columns = sorted(
        ensemble[0][1]["target_columns"],
        key=target_sort_key,
    )

    validation_source = (
        frame[frame["split"] == "validation"]
        .sort_values("case_id")
        .reset_index(drop=True)
    )

    test_source = (
        frame[frame["split"] == "test"]
        .sort_values("case_id")
        .reset_index(drop=True)
    )

    validation = evaluate_split(
        validation_source,
        ensemble,
        feature_columns,
        target_columns,
        args.ood_z_threshold,
        args.large_error_threshold,
    )

    threshold, calibration_table = calibrate_threshold(
        validation,
        args.max_validation_fallback_rate,
    )

    validation = add_final_guardrail(
        validation,
        threshold,
    )

    test = evaluate_split(
        test_source,
        ensemble,
        feature_columns,
        target_columns,
        args.ood_z_threshold,
        args.large_error_threshold,
    )

    test = add_final_guardrail(
        test,
        threshold,
    )

    validation_metrics = guardrail_metrics(
        validation,
        validation["fallback_triggered"],
    )

    test_metrics = guardrail_metrics(
        test,
        test["fallback_triggered"],
    )

    base_test_metrics = guardrail_metrics(
        test,
        test["base_rule_triggered"],
    )

    latency = measure_sequential_latency(
        test_source,
        ensemble,
        feature_columns,
    )

    summary = {
        "ensemble_models": len(ensemble),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "large_error_threshold_iv": (
            args.large_error_threshold
        ),
        "ood_z_threshold": args.ood_z_threshold,
        "max_validation_fallback_rate": (
            args.max_validation_fallback_rate
        ),
        "selected_max_std_threshold_iv": threshold,
        "validation_guardrail_metrics": validation_metrics,
        "test_base_rule_metrics": base_test_metrics,
        "test_combined_guardrail_metrics": test_metrics,
        "test_mean_rmse_iv": float(test["rmse_iv"].mean()),
        "test_mean_mae_iv": float(test["mae_iv"].mean()),
        "test_max_abs_error_iv": float(
            test["max_abs_error_iv"].max()
        ),
        "test_mean_max_ensemble_std_iv": float(
            test["max_ensemble_std_iv"].mean()
        ),
        "ensemble_mean_latency_microseconds": float(
            latency.mean()
        ),
        "ensemble_median_latency_microseconds": float(
            np.median(latency)
        ),
        "ensemble_p95_latency_microseconds": float(
            np.quantile(latency, 0.95)
        ),
    }

    validation_path = outdir / "validation_guardrails.csv"
    test_path = outdir / "test_guardrails.csv"
    calibration_path = outdir / "threshold_calibration.csv"
    summary_path = outdir / "ensemble_guardrail_summary.json"

    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)
    calibration_table.to_csv(calibration_path, index=False)

    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    print("Ensemble guardrail evaluation complete.")
    print(json.dumps(summary, indent=2))

    print()
    print("Largest test errors:")
    print(
        test.sort_values(
            "max_abs_error_iv",
            ascending=False,
        ).head(10)
    )

    print()
    print("Wrote:", validation_path)
    print("Wrote:", test_path)
    print("Wrote:", calibration_path)
    print("Wrote:", summary_path)


if __name__ == "__main__":
    main()
