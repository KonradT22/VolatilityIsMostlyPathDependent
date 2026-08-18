import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGET_COLS = [
    "beta0",
    "beta1",
    "beta2",
    "theta1",
    "theta2",
]

PARAM_BOUNDS = {
    "beta0": (0.02, 0.06),
    "beta1": (-0.16, -0.06),
    "beta2": (0.55, 0.85),
    "theta1": (0.0, 1.0),
    "theta2": (0.0, 1.0),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class InverseMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),

            nn.Linear(256, 256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, 5),
        )

    def forward(self, x):
        return self.net(x)


def fit_standardizer(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)

    std = np.where(
        std > 1e-12,
        std,
        1.0,
    )

    return mean, std


def regression_metrics(
    truth,
    pred,
    target_cols,
    y_std,
):
    result = {}

    standardized_error = (
        pred - truth
    ) / y_std

    result[
        "aggregate_standardized_rmse"
    ] = float(
        np.sqrt(
            np.mean(
                standardized_error ** 2
            )
        )
    )

    per_target = {}

    for j, name in enumerate(
        target_cols
    ):
        y = truth[:, j]
        p = pred[:, j]

        error = p - y

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
                (y - y.mean()) ** 2
            )
        )

        if ss_tot > 0:
            r2 = float(
                1.0
                - ss_res / ss_tot
            )
        else:
            r2 = float("nan")

        low, high = (
            PARAM_BOUNDS[name]
        )

        range_width = high - low

        normalized_rmse = (
            rmse / range_width
        )

        below = int(
            np.sum(p < low)
        )

        above = int(
            np.sum(p > high)
        )

        per_target[name] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "range_normalized_rmse":
                float(
                    normalized_rmse
                ),
            "pred_min":
                float(p.min()),
            "pred_max":
                float(p.max()),
            "below_bound_count":
                below,
            "above_bound_count":
                above,
        }

    result["per_target"] = per_target

    return result


def predict(
    model,
    x,
    y_mean,
    y_std,
):
    model.eval()

    with torch.no_grad():
        pred_z = (
            model(x)
            .cpu()
            .numpy()
        )

    return (
        pred_z * y_std
        + y_mean
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path(
            "/users/4/trest017/urop_pdv/"
            "benchmarks/inverse_ann/"
            "pdv_anchor_jitter_10k/splits"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/users/4/trest017/urop_pdv/"
            "benchmarks/inverse_ann/"
            "pdv_anchor_jitter_10k/"
            "models/mlp_baseline"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026081801,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(args.seed)

    torch.set_num_threads(
        max(
            1,
            min(
                2,
                torch.get_num_threads(),
            ),
        )
    )

    with open(
        args.split_dir
        / "feature_columns.json"
    ) as f:
        feature_info = json.load(f)

    feature_cols = (
        feature_info[
            "all_features"
        ]
    )

    target_cols = (
        feature_info[
            "targets"
        ]
    )

    if target_cols != TARGET_COLS:
        raise RuntimeError(
            "Unexpected target-column order: "
            f"{target_cols}"
        )

    if len(feature_cols) != 81:
        raise RuntimeError(
            f"Expected 81 features, "
            f"found {len(feature_cols)}"
        )

    train_df = pd.read_csv(
        args.split_dir
        / "train.csv"
    )

    val_df = pd.read_csv(
        args.split_dir
        / "validation.csv"
    )

    x_train = train_df[
        feature_cols
    ].to_numpy(
        dtype=np.float64
    )

    y_train = train_df[
        target_cols
    ].to_numpy(
        dtype=np.float64
    )

    x_val = val_df[
        feature_cols
    ].to_numpy(
        dtype=np.float64
    )

    y_val = val_df[
        target_cols
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        x_train
    ).all():
        raise RuntimeError(
            "Nonfinite training features"
        )

    if not np.isfinite(
        y_train
    ).all():
        raise RuntimeError(
            "Nonfinite training targets"
        )

    if not np.isfinite(
        x_val
    ).all():
        raise RuntimeError(
            "Nonfinite validation features"
        )

    if not np.isfinite(
        y_val
    ).all():
        raise RuntimeError(
            "Nonfinite validation targets"
        )

    # Fit ALL preprocessing on training data only.
    x_mean, x_std = (
        fit_standardizer(
            x_train
        )
    )

    y_mean, y_std = (
        fit_standardizer(
            y_train
        )
    )

    x_train_z = (
        x_train - x_mean
    ) / x_std

    y_train_z = (
        y_train - y_mean
    ) / y_std

    x_val_z = (
        x_val - x_mean
    ) / x_std

    y_val_z = (
        y_val - y_mean
    ) / y_std

    train_x_tensor = (
        torch.tensor(
            x_train_z,
            dtype=torch.float32,
        )
    )

    train_y_tensor = (
        torch.tensor(
            y_train_z,
            dtype=torch.float32,
        )
    )

    val_x_tensor = (
        torch.tensor(
            x_val_z,
            dtype=torch.float32,
        )
    )

    val_y_tensor = (
        torch.tensor(
            y_val_z,
            dtype=torch.float32,
        )
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        args.seed
    )

    loader = DataLoader(
        TensorDataset(
            train_x_tensor,
            train_y_tensor,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = InverseMLP(
        input_dim=len(
            feature_cols
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = None
    best_state = None
    patience_count = 0

    history = []

    print("=" * 78)
    print("PDV INVERSE MLP BASELINE")
    print("=" * 78)
    print(
        "training rows:",
        len(train_df),
    )
    print(
        "validation rows:",
        len(val_df),
    )
    print(
        "input features:",
        len(feature_cols),
    )
    print(
        "targets:",
        len(target_cols),
    )
    print(
        "architecture:",
        "81 -> 256 -> 256 -> 128 -> 5",
    )
    print(
        "seed:",
        args.seed,
    )
    print()

    start_time = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in loader:
            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(xb)

            loss = loss_fn(
                pred,
                yb,
            )

            loss.backward()

            optimizer.step()

            batch_n = len(xb)

            train_loss_sum += (
                float(loss.item())
                * batch_n
            )

            train_count += batch_n

        train_loss = (
            train_loss_sum
            / train_count
        )

        model.eval()

        with torch.no_grad():
            val_pred = model(
                val_x_tensor
            )

            val_loss = float(
                loss_fn(
                    val_pred,
                    val_y_tensor,
                ).item()
            )

        history.append({
            "epoch":
                epoch,
            "train_mse_standardized":
                train_loss,
            "validation_mse_standardized":
                val_loss,
        })

        improved = (
            val_loss
            < best_val_loss - 1e-8
        )

        if improved:
            best_val_loss = (
                val_loss
            )

            best_epoch = epoch

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            patience_count = 0

        else:
            patience_count += 1

        if (
            epoch == 1
            or epoch % 25 == 0
            or improved
            and epoch <= 10
        ):
            print(
                f"epoch={epoch:4d} "
                f"train={train_loss:.8f} "
                f"val={val_loss:.8f} "
                f"best={best_val_loss:.8f} "
                f"patience={patience_count}",
                flush=True,
            )

        if (
            patience_count
            >= args.patience
        ):
            print(
                f"\nEarly stopping at "
                f"epoch {epoch}.",
                flush=True,
            )
            break

    elapsed = (
        time.time()
        - start_time
    )

    if best_state is None:
        raise RuntimeError(
            "Training produced no "
            "best checkpoint"
        )

    model.load_state_dict(
        best_state
    )

    train_pred = predict(
        model,
        train_x_tensor,
        y_mean,
        y_std,
    )

    val_pred = predict(
        model,
        val_x_tensor,
        y_mean,
        y_std,
    )

    train_metrics = (
        regression_metrics(
            truth=y_train,
            pred=train_pred,
            target_cols=target_cols,
            y_std=y_std,
        )
    )

    val_metrics = (
        regression_metrics(
            truth=y_val,
            pred=val_pred,
            target_cols=target_cols,
            y_std=y_std,
        )
    )

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        args.output_dir
        / "training_history.csv",
        index=False,
    )

    val_predictions = val_df[
        [
            "sample_id",
            "anchor_date",
        ]
    ].copy()

    for j, name in enumerate(
        target_cols
    ):
        val_predictions[
            f"true_{name}"
        ] = y_val[:, j]

        val_predictions[
            f"pred_{name}"
        ] = val_pred[:, j]

        val_predictions[
            f"error_{name}"
        ] = (
            val_pred[:, j]
            - y_val[:, j]
        )

    val_predictions.to_csv(
        args.output_dir
        / "validation_predictions.csv",
        index=False,
    )

    np.savez(
        args.output_dir
        / "standardization.npz",
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),
        "seed":
            args.seed,
        "feature_columns":
            feature_cols,
        "target_columns":
            target_cols,
        "architecture":
            [81, 256, 256, 128, 5],
        "x_mean":
            x_mean,
        "x_std":
            x_std,
        "y_mean":
            y_mean,
        "y_std":
            y_std,
        "best_epoch":
            best_epoch,
        "best_validation_loss":
            best_val_loss,
    }

    torch.save(
        checkpoint,
        args.output_dir
        / "best_model.pt",
    )

    summary = {
        "seed":
            args.seed,
        "train_rows":
            len(train_df),
        "validation_rows":
            len(val_df),
        "input_features":
            len(feature_cols),
        "target_count":
            len(target_cols),
        "architecture":
            [81, 256, 256, 128, 5],
        "learning_rate":
            args.learning_rate,
        "weight_decay":
            args.weight_decay,
        "batch_size":
            args.batch_size,
        "epochs_requested":
            args.epochs,
        "epochs_completed":
            len(history),
        "best_epoch":
            best_epoch,
        "best_validation_mse_standardized":
            best_val_loss,
        "elapsed_seconds":
            elapsed,
        "train_metrics":
            train_metrics,
        "validation_metrics":
            val_metrics,
    }

    with open(
        args.output_dir
        / "training_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print("BEST MODEL")
    print("=" * 78)
    print(
        "best epoch:",
        best_epoch,
    )
    print(
        "best validation standardized MSE:",
        f"{best_val_loss:.8f}",
    )
    print(
        "validation standardized RMSE:",
        f"{val_metrics['aggregate_standardized_rmse']:.8f}",
    )
    print(
        "elapsed seconds:",
        f"{elapsed:.2f}",
    )

    print()
    print("VALIDATION PARAMETER METRICS")
    print("-" * 78)

    for name in target_cols:
        m = (
            val_metrics[
                "per_target"
            ][name]
        )

        print(
            f"{name:7s} "
            f"RMSE={m['rmse']:.8f} "
            f"MAE={m['mae']:.8f} "
            f"R2={m['r2']:.6f} "
            f"rangeNRMSE="
            f"{m['range_normalized_rmse']:.6f} "
            f"bounds="
            f"{m['below_bound_count']} below/"
            f"{m['above_bound_count']} above"
        )

    print()
    print(
        "Wrote:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
