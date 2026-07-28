import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(frame, fallback, error_threshold):
    fallback = np.asarray(fallback, dtype=bool)
    large_error = (
        frame["max_abs_error_iv"].to_numpy(dtype=float)
        > error_threshold
    )

    true_positive = int(np.sum(fallback & large_error))
    false_positive = int(np.sum(fallback & ~large_error))
    false_negative = int(np.sum(~fallback & large_error))

    fallback_count = int(np.sum(fallback))
    positive_count = int(np.sum(large_error))
    negative_count = int(np.sum(~large_error))

    recall = (
        true_positive / positive_count
        if positive_count else 0.0
    )

    precision = (
        true_positive / fallback_count
        if fallback_count else 0.0
    )

    false_positive_rate = (
        false_positive / negative_count
        if negative_count else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    recall_ceiling = (
        min(fallback_count, positive_count) / positive_count
        if positive_count else 0.0
    )

    recall_efficiency = (
        recall / recall_ceiling
        if recall_ceiling else 0.0
    )

    return {
        "large_error_count": positive_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(frame),
        "recall": recall,
        "precision": precision,
        "false_positive_rate": false_positive_rate,
        "f1": f1,
        "recall_ceiling_given_fallback_count": recall_ceiling,
        "recall_efficiency_vs_ceiling": recall_efficiency,
    }


def calibrate(validation, error_threshold, fallback_budget):
    scores = validation[
        "max_ensemble_std_iv"
    ].to_numpy(dtype=float)

    base_rule = validation[
        "base_rule_triggered"
    ].to_numpy(dtype=bool)

    candidates = np.unique(
        np.concatenate([
            scores,
            [scores.min() - 1e-12],
            [scores.max() + 1e-12],
        ])
    )

    rows = []

    for threshold in candidates:
        fallback = base_rule | (scores >= threshold)
        result = metrics(
            validation,
            fallback,
            error_threshold,
        )

        rows.append({
            "threshold": float(threshold),
            **result,
        })

    table = pd.DataFrame(rows)

    feasible = table[
        table["fallback_rate"] <= fallback_budget
    ].copy()

    if feasible.empty:
        feasible = table.copy()

    feasible = feasible.sort_values(
        ["recall", "precision", "f1", "fallback_rate"],
        ascending=[False, False, False, True],
    )

    return float(feasible.iloc[0]["threshold"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validation",
        default=(
            "/users/4/trest017/urop_pdv/benchmarks/"
            "surrogate/pilot_ensemble_guardrails/"
            "validation_guardrails.csv"
        ),
    )

    parser.add_argument(
        "--test",
        default=(
            "/users/4/trest017/urop_pdv/benchmarks/"
            "surrogate/pilot_ensemble_guardrails/"
            "test_guardrails.csv"
        ),
    )

    parser.add_argument(
        "--dataset",
        default=(
            "/users/4/trest017/urop_pdv/data/processed/"
            "surrogate/pilot_surface_dataset/"
            "pdv_surface_dataset.csv"
        ),
    )

    parser.add_argument(
        "--ensemble-summary",
        default=(
            "/users/4/trest017/urop_pdv/benchmarks/"
            "surrogate/pilot_ensemble_guardrails/"
            "ensemble_guardrail_summary.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "/users/4/trest017/urop_pdv/benchmarks/"
            "surrogate/pilot_ensemble_guardrails/"
            "two_lane_operating_points.csv"
        ),
    )

    args = parser.parse_args()

    validation = pd.read_csv(args.validation)
    test = pd.read_csv(args.test)
    dataset = pd.read_csv(args.dataset)

    with open(args.ensemble_summary) as file:
        ensemble_summary = json.load(file)

    fast_seconds = (
        ensemble_summary["ensemble_mean_latency_microseconds"]
        / 1_000_000
    )

    pricing_columns = [
        column
        for column in dataset.columns
        if column.startswith("pricing_seconds_dte_")
    ]

    dataset["slow_lane_seconds"] = (
        dataset["simulate_seconds"]
        + dataset[pricing_columns].sum(axis=1)
    )

    test = test.merge(
        dataset[["case_id", "slow_lane_seconds"]],
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    if test["slow_lane_seconds"].isna().any():
        raise ValueError("Missing slow-lane timings after merge")

    rows = []

    error_thresholds = [0.02, 0.04, 0.06]
    fallback_budgets = [0.10, 0.20, 0.30]

    for error_threshold in error_thresholds:
        for fallback_budget in fallback_budgets:
            threshold = calibrate(
                validation,
                error_threshold,
                fallback_budget,
            )

            fallback = (
                test["base_rule_triggered"].to_numpy(dtype=bool)
                | (
                    test["max_ensemble_std_iv"].to_numpy(dtype=float)
                    >= threshold
                )
            )

            result = metrics(
                test,
                fallback,
                error_threshold,
            )

            hybrid_rmse_by_surface = np.where(
                fallback,
                0.0,
                test["rmse_iv"].to_numpy(dtype=float),
            )

            hybrid_mae_by_surface = np.where(
                fallback,
                0.0,
                test["mae_iv"].to_numpy(dtype=float),
            )

            hybrid_max_error = np.where(
                fallback,
                0.0,
                test["max_abs_error_iv"].to_numpy(dtype=float),
            )

            slow_seconds = test[
                "slow_lane_seconds"
            ].to_numpy(dtype=float)

            hybrid_seconds = (
                fast_seconds
                + fallback.astype(float) * slow_seconds
            )

            always_slow_mean = float(np.mean(slow_seconds))
            hybrid_mean = float(np.mean(hybrid_seconds))

            rows.append({
                "error_threshold_iv": error_threshold,
                "fallback_budget_validation": fallback_budget,
                "selected_disagreement_threshold_iv": threshold,
                **result,
                "ann_only_mean_surface_rmse_iv": float(
                    test["rmse_iv"].mean()
                ),
                "hybrid_mean_surface_rmse_iv": float(
                    hybrid_rmse_by_surface.mean()
                ),
                "ann_only_mean_surface_mae_iv": float(
                    test["mae_iv"].mean()
                ),
                "hybrid_mean_surface_mae_iv": float(
                    hybrid_mae_by_surface.mean()
                ),
                "ann_only_max_error_iv": float(
                    test["max_abs_error_iv"].max()
                ),
                "hybrid_max_error_iv": float(
                    hybrid_max_error.max()
                ),
                "ann_latency_milliseconds": fast_seconds * 1000,
                "always_slow_mean_seconds": always_slow_mean,
                "hybrid_mean_seconds": hybrid_mean,
                "speedup_vs_always_slow": (
                    always_slow_mean / hybrid_mean
                ),
            })

    output = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 320)

    print("Two-lane operating points:")
    print(output)

    print()
    print("Reference-lane timing:")
    print(test["slow_lane_seconds"].describe())

    print()
    print("Wrote:", output_path)


if __name__ == "__main__":
    main()
