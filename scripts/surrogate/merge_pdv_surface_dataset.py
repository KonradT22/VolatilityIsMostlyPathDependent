import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default=(
            "/users/4/trest017/urop_pdv/"
            "data/processed/surrogate/pilot_surface_dataset"
        ),
    )
    parser.add_argument("--split-seed", type=int, default=20260728)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    shard_paths = sorted(input_dir.glob("pdv_surface_shard_*.csv"))

    if not shard_paths:
        raise FileNotFoundError(f"No shard CSVs found in {input_dir}")

    frames = [pd.read_csv(path) for path in shard_paths]
    dataset = pd.concat(frames, ignore_index=True)

    dataset = dataset.sort_values("case_id").reset_index(drop=True)

    if dataset["case_id"].duplicated().any():
        duplicates = dataset.loc[
            dataset["case_id"].duplicated(),
            "case_id",
        ].tolist()
        raise ValueError(f"Duplicate case IDs: {duplicates}")

    rng = np.random.default_rng(args.split_seed)
    permutation = rng.permutation(len(dataset))

    train_end = int(0.70 * len(dataset))
    validation_end = int(0.85 * len(dataset))

    split = np.empty(len(dataset), dtype=object)
    split[permutation[:train_end]] = "train"
    split[permutation[train_end:validation_end]] = "validation"
    split[permutation[validation_end:]] = "test"

    dataset["split"] = split

    full_path = input_dir / "pdv_surface_dataset.csv"
    train_path = input_dir / "pdv_surface_train.csv"
    validation_path = input_dir / "pdv_surface_validation.csv"
    test_path = input_dir / "pdv_surface_test.csv"
    metadata_path = input_dir / "pdv_surface_dataset_metadata.json"

    dataset.to_csv(full_path, index=False)
    dataset[dataset["split"] == "train"].to_csv(train_path, index=False)
    dataset[dataset["split"] == "validation"].to_csv(
        validation_path,
        index=False,
    )
    dataset[dataset["split"] == "test"].to_csv(test_path, index=False)

    iv_columns = sorted(
        [column for column in dataset.columns if column.startswith("iv_dte_")]
    )

    feature_columns = [
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

    metadata = {
        "rows": len(dataset),
        "shards": len(shard_paths),
        "feature_columns": feature_columns,
        "target_columns": iv_columns,
        "input_size": len(feature_columns),
        "output_size": len(iv_columns),
        "split_seed": args.split_seed,
        "split_counts": dataset["split"].value_counts().to_dict(),
        "source_shards": [str(path) for path in shard_paths],
    }

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2)

    print("Shards:", len(shard_paths))
    print("Rows:", len(dataset))
    print("Columns:", len(dataset.columns))
    print("Inputs:", len(feature_columns))
    print("Outputs:", len(iv_columns))
    print()
    print("Split counts:")
    print(dataset["split"].value_counts())
    print()
    print("IV range:")
    print(
        dataset[iv_columns]
        .to_numpy()
        .min(),
        dataset[iv_columns]
        .to_numpy()
        .max(),
    )
    print()
    print("Wrote:", full_path)
    print("Wrote:", train_path)
    print("Wrote:", validation_path)
    print("Wrote:", test_path)
    print("Wrote:", metadata_path)


if __name__ == "__main__":
    main()
