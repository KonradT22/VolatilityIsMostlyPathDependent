import argparse
import json
import random
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_COLUMNS = [
    "beta0",
    "beta1",
    "beta2",
    "theta1",
    "theta2",
    "r1_0",
    "r1_1",
    "r2_0",
    "r2_1",
]


def target_sort_key(column):
    match = re.fullmatch(r"iv_dte_(\d+)_m(\d+)", column)
    if match is None:
        raise ValueError(f"Unexpected target column: {column}")
    return int(match.group(1)), int(match.group(2))


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


def make_loader(x, y, batch_size, shuffle, seed):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_rows = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)

            total_loss += loss.item() * len(x_batch)
            total_rows += len(x_batch)

    return total_loss / total_rows


def predict(model, x):
    model.eval()

    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32)
        return model(tensor).cpu().numpy()


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
        "--out-dir",
        default=(
            "/users/4/trest017/urop_pdv/"
            "benchmarks/surrogate/pilot_surface_ann"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    frame = pd.read_csv(dataset_path)

    target_columns = sorted(
        [
            column
            for column in frame.columns
            if column.startswith("iv_dte_")
        ],
        key=target_sort_key,
    )

    if len(target_columns) != 102:
        raise ValueError(
            f"Expected 102 IV targets, found {len(target_columns)}"
        )

    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One or more dataset splits are empty")

    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    x_validation = validation[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    x_test = test[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    y_train = train[target_columns].to_numpy(dtype=np.float32)
    y_validation = validation[target_columns].to_numpy(dtype=np.float32)
    y_test = test[target_columns].to_numpy(dtype=np.float32)

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    x_train_scaled = (x_train - x_mean) / x_std
    x_validation_scaled = (x_validation - x_mean) / x_std
    x_test_scaled = (x_test - x_mean) / x_std

    y_train_scaled = (y_train - y_mean) / y_std
    y_validation_scaled = (y_validation - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    train_loader = make_loader(
        x_train_scaled,
        y_train_scaled,
        args.batch_size,
        True,
        args.seed,
    )

    validation_loader = make_loader(
        x_validation_scaled,
        y_validation_scaled,
        args.batch_size,
        False,
        args.seed,
    )

    model = SurfaceSurrogate(
        input_size=len(FEATURE_COLUMNS),
        output_size=len(target_columns),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    criterion = nn.MSELoss()

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    best_state_path = outdir / "best_model_state.pt"

    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_train_loss = 0.0
        total_train_rows = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(x_batch)
            total_train_rows += len(x_batch)

        train_loss = total_train_loss / total_train_rows
        validation_loss = evaluate_loss(
            model,
            validation_loader,
            criterion,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })

        if validation_loss < best_validation_loss - 1e-7:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_state_path)
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoch={epoch:4d} "
                f"train={train_loss:.6f} "
                f"validation={validation_loss:.6f} "
                f"best_epoch={best_epoch}"
            )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}"
            )
            break

    training_seconds = time.perf_counter() - training_start

    model.load_state_dict(
        torch.load(best_state_path, map_location="cpu")
    )

    prediction_scaled = predict(model, x_test_scaled)
    prediction = prediction_scaled * y_std + y_mean

    error = prediction - y_test
    absolute_error = np.abs(error)

    test_rmse = float(np.sqrt(np.mean(error ** 2)))
    test_mae = float(np.mean(absolute_error))
    test_median_abs = float(np.median(absolute_error))
    test_p95_abs = float(np.quantile(absolute_error, 0.95))
    test_max_abs = float(np.max(absolute_error))

    denominator = np.sum((y_test - y_test.mean()) ** 2)
    r_squared = float(
        1.0 - np.sum(error ** 2) / denominator
    )

    mean_prediction = np.broadcast_to(
        y_train.mean(axis=0),
        y_test.shape,
    )
    mean_baseline_rmse = float(
        np.sqrt(np.mean((mean_prediction - y_test) ** 2))
    )

    per_dte_rows = []

    dtes = sorted({
        target_sort_key(column)[0]
        for column in target_columns
    })

    for dte in dtes:
        indices = [
            index
            for index, column in enumerate(target_columns)
            if target_sort_key(column)[0] == dte
        ]

        dte_error = error[:, indices]
        dte_absolute_error = np.abs(dte_error)

        per_dte_rows.append({
            "dte": dte,
            "rmse_iv": float(np.sqrt(np.mean(dte_error ** 2))),
            "mae_iv": float(np.mean(dte_absolute_error)),
            "p95_abs_error_iv": float(
                np.quantile(dte_absolute_error, 0.95)
            ),
            "max_abs_error_iv": float(
                np.max(dte_absolute_error)
            ),
        })

    per_dte = pd.DataFrame(per_dte_rows)

    sample = torch.tensor(
        x_test_scaled[:1],
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():
        for _ in range(200):
            model(sample)

        repeats = 10000
        latency_start = time.perf_counter()

        for _ in range(repeats):
            model(sample)

        latency_seconds = time.perf_counter() - latency_start

    single_surface_latency_microseconds = (
        latency_seconds / repeats * 1e6
    )

    checkpoint_path = outdir / "pdv_surface_surrogate.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": len(FEATURE_COLUMNS),
        "output_size": len(target_columns),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": target_columns,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "seed": args.seed,
        "best_epoch": best_epoch,
    }, checkpoint_path)

    history_frame = pd.DataFrame(history)
    history_path = outdir / "training_history.csv"
    history_frame.to_csv(history_path, index=False)

    per_dte_path = outdir / "test_metrics_by_dte.csv"
    per_dte.to_csv(per_dte_path, index=False)

    prediction_frame = pd.DataFrame({
        "case_id": test["case_id"].to_numpy(),
    })

    for index, column in enumerate(target_columns):
        prediction_frame[f"actual_{column}"] = y_test[:, index]
        prediction_frame[f"predicted_{column}"] = prediction[:, index]
        prediction_frame[f"error_{column}"] = error[:, index]

    predictions_path = outdir / "test_predictions.csv"
    prediction_frame.to_csv(predictions_path, index=False)

    metrics = {
        "dataset": str(dataset_path),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "input_size": len(FEATURE_COLUMNS),
        "output_size": len(target_columns),
        "best_epoch": best_epoch,
        "best_validation_loss_scaled": best_validation_loss,
        "training_seconds": training_seconds,
        "test_rmse_iv": test_rmse,
        "test_mae_iv": test_mae,
        "test_median_abs_error_iv": test_median_abs,
        "test_p95_abs_error_iv": test_p95_abs,
        "test_max_abs_error_iv": test_max_abs,
        "test_rmse_vol_points": test_rmse * 100,
        "test_mae_vol_points": test_mae * 100,
        "test_r_squared": r_squared,
        "mean_predictor_rmse_iv": mean_baseline_rmse,
        "single_surface_latency_microseconds": (
            single_surface_latency_microseconds
        ),
    }

    metrics_path = outdir / "test_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=2)

    plt.figure()
    plt.plot(
        history_frame["epoch"],
        history_frame["train_loss"],
        label="Train",
    )
    plt.plot(
        history_frame["epoch"],
        history_frame["validation_loss"],
        label="Validation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Scaled MSE")
    plt.title("PDV Surface Surrogate Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    loss_plot_path = outdir / "training_history.png"
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(
        per_dte["dte"],
        per_dte["rmse_iv"],
        marker="o",
    )
    plt.xlabel("DTE")
    plt.ylabel("Test RMSE in implied volatility")
    plt.title("Surrogate Test Error by Maturity")
    plt.grid(True)
    plt.tight_layout()
    dte_plot_path = outdir / "test_rmse_by_dte.png"
    plt.savefig(dte_plot_path, dpi=150)
    plt.close()

    print()
    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print()
    print("Metrics by DTE:")
    print(per_dte)
    print()
    print("Wrote:", checkpoint_path)
    print("Wrote:", metrics_path)
    print("Wrote:", history_path)
    print("Wrote:", per_dte_path)
    print("Wrote:", predictions_path)
    print("Wrote:", loss_plot_path)
    print("Wrote:", dte_plot_path)


if __name__ == "__main__":
    main()
