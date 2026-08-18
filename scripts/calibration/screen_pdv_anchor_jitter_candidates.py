import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.calibration.calibrate_pdv_market_5param_fixed_grid import (
    build_model,
)


BASE = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_market_5param_fixed_grid/"
    "robust_historical_prior"
)

INPUT_PATH = (
    BASE / "anchor_jitter_samples.csv"
)

OUTDIR = (
    BASE / "anchor_jitter_validity_pilot"
)

PARAMS = [
    "beta0",
    "beta1",
    "beta2",
    "theta1",
    "theta2",
]

STATE_COLS = [
    "R1_fast",
    "R1_slow",
    "R2_fast",
    "R2_slow",
]

SEEDS = [
    2026081820,
    2026081821,
    2026081822,
]

N_PATHS = 20000
PER_ANCHOR = 32
SELECTION_SEED = 2026081803

# Canonical ANN grid has maximum maturity 90D.
# Keep the same one-day simulation-horizon cushion
# used by the historical calibrator.
SIMULATION_MATURITY = 91 / 365.0


def select_pilot(samples):
    pieces = []

    for anchor_date, group in samples.groupby(
        "anchor_date",
        sort=True,
    ):
        if len(group) < PER_ANCHOR:
            raise RuntimeError(
                f"{anchor_date}: only {len(group)} "
                f"samples available"
            )

        # Deterministic but different selection per group
        # through pandas' shared fixed random state.
        chosen = group.sample(
            n=PER_ANCHOR,
            random_state=SELECTION_SEED,
        )

        pieces.append(chosen)

    pilot = pd.concat(
        pieces,
        ignore_index=True,
    )

    expected = (
        samples["anchor_date"].nunique()
        * PER_ANCHOR
    )

    if len(pilot) != expected:
        raise RuntimeError(
            f"Expected {expected} pilot rows, "
            f"found {len(pilot)}"
        )

    return pilot


def evaluate_candidate(row, seed):
    params = np.array(
        [
            row.beta0,
            row.beta1,
            row.beta2,
            row.theta1,
            row.theta2,
        ],
        dtype=float,
    )

    r1 = torch.tensor(
        [
            row.R1_fast,
            row.R1_slow,
        ],
        dtype=torch.float64,
    )

    r2 = torch.tensor(
        [
            row.R2_fast,
            row.R2_slow,
        ],
        dtype=torch.float64,
    )

    model = build_model(
        params=params,
        R_init1=r1,
        R_init2=r2,
        simulation_maturity=SIMULATION_MATURITY,
        n_paths=N_PATHS,
        seed_root=seed,
    )

    model.simulate(
        save_R=False
    )

    vol = (
        model.vol_array
        .detach()
        .cpu()
        .numpy()
    )

    if not np.all(
        np.isfinite(vol)
    ):
        return {
            "valid": False,
            "failure_reason":
                "nonfinite_volatility",
            "min_vol": np.nan,
            "max_vol": np.nan,
        }

    min_vol = float(
        np.min(vol)
    )

    max_vol = float(
        np.max(vol)
    )

    if min_vol <= 0:
        return {
            "valid": False,
            "failure_reason":
                "nonpositive_volatility",
            "min_vol": min_vol,
            "max_vol": max_vol,
        }

    return {
        "valid": True,
        "failure_reason": "",
        "min_vol": min_vol,
        "max_vol": max_vol,
    }


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = pd.read_csv(
        INPUT_PATH
    )

    pilot = select_pilot(
        samples
    )

    pilot.to_csv(
        OUTDIR
        / "pilot_candidates.csv",
        index=False,
    )

    print("=" * 78)
    print("PDV ANCHOR-JITTER VALIDITY PILOT")
    print("=" * 78)
    print(
        "Candidates:",
        len(pilot),
    )
    print(
        "Anchors:",
        pilot["anchor_date"].nunique(),
    )
    print(
        "Candidates per anchor:",
        PER_ANCHOR,
    )
    print(
        "Paths per seed:",
        N_PATHS,
    )
    print(
        "Seeds:",
        SEEDS,
    )
    print()

    rows = []

    total = len(pilot)

    for i, row in enumerate(
        pilot.itertuples(index=False),
        start=1,
    ):
        candidate_valid = True

        for seed in SEEDS:
            try:
                result = evaluate_candidate(
                    row,
                    seed,
                )

            except Exception as exc:
                result = {
                    "valid": False,
                    "failure_reason":
                        f"{type(exc).__name__}:{exc}",
                    "min_vol": np.nan,
                    "max_vol": np.nan,
                }

            rows.append({
                "sample_id":
                    int(row.sample_id),
                "anchor_date":
                    str(row.anchor_date),
                "seed_root":
                    int(seed),
                "N_paths":
                    N_PATHS,
                "beta0":
                    float(row.beta0),
                "beta1":
                    float(row.beta1),
                "beta2":
                    float(row.beta2),
                "theta1":
                    float(row.theta1),
                "theta2":
                    float(row.theta2),
                "R1_fast":
                    float(row.R1_fast),
                "R1_slow":
                    float(row.R1_slow),
                "R2_fast":
                    float(row.R2_fast),
                "R2_slow":
                    float(row.R2_slow),
                "valid":
                    bool(result["valid"]),
                "failure_reason":
                    result["failure_reason"],
                "min_vol":
                    result["min_vol"],
                "max_vol":
                    result["max_vol"],
            })

            if not result["valid"]:
                candidate_valid = False

        # Checkpoint after every candidate.
        pd.DataFrame(
            rows
        ).to_csv(
            OUTDIR
            / "pilot_seed_results.csv",
            index=False,
        )

        if (
            i == 1
            or i % 25 == 0
            or not candidate_valid
        ):
            valid_count = sum(
                x["valid"]
                for x in rows
            )

            print(
                f"candidate={i:4d}/{total} "
                f"sample_id={int(row.sample_id):5d} "
                f"anchor={row.anchor_date} "
                f"candidate_valid={candidate_valid} "
                f"seed_valid_rate="
                f"{valid_count / len(rows):.4f}",
                flush=True,
            )

    results = pd.DataFrame(
        rows
    )

    candidate_summary = (
        results.groupby(
            [
                "sample_id",
                "anchor_date",
            ],
            as_index=False,
        )
        .agg(
            valid_seed_count=(
                "valid",
                "sum",
            ),
            all_seeds_valid=(
                "valid",
                "all",
            ),
            minimum_vol=(
                "min_vol",
                "min",
            ),
            maximum_vol=(
                "max_vol",
                "max",
            ),
        )
    )

    candidate_summary[
        "invalid_seed_count"
    ] = (
        len(SEEDS)
        - candidate_summary[
            "valid_seed_count"
        ]
    )

    candidate_summary.to_csv(
        OUTDIR
        / "pilot_candidate_summary.csv",
        index=False,
    )

    anchor_summary = (
        candidate_summary.groupby(
            "anchor_date",
            as_index=False,
        )
        .agg(
            candidates=(
                "sample_id",
                "count",
            ),
            robust_candidates=(
                "all_seeds_valid",
                "sum",
            ),
            minimum_observed_vol=(
                "minimum_vol",
                "min",
            ),
        )
    )

    anchor_summary[
        "robust_rate"
    ] = (
        anchor_summary[
            "robust_candidates"
        ]
        / anchor_summary[
            "candidates"
        ]
    )

    anchor_summary.to_csv(
        OUTDIR
        / "pilot_anchor_summary.csv",
        index=False,
    )

    invalid = results[
        ~results["valid"]
    ].copy()

    overall_robust = int(
        candidate_summary[
            "all_seeds_valid"
        ].sum()
    )

    overall_rate = (
        overall_robust
        / len(candidate_summary)
    )

    summary = {
        "candidate_count":
            int(len(candidate_summary)),
        "anchor_count":
            int(
                candidate_summary[
                    "anchor_date"
                ].nunique()
            ),
        "candidates_per_anchor":
            PER_ANCHOR,
        "N_paths":
            N_PATHS,
        "seeds":
            SEEDS,
        "seed_evaluation_count":
            int(len(results)),
        "valid_seed_evaluations":
            int(
                results["valid"].sum()
            ),
        "invalid_seed_evaluations":
            int(
                (~results["valid"]).sum()
            ),
        "all_seed_valid_candidates":
            overall_robust,
        "candidate_robust_rate":
            float(overall_rate),
        "anchors_with_100pct_robust_rate":
            int(
                (
                    anchor_summary[
                        "robust_rate"
                    ]
                    == 1.0
                ).sum()
            ),
    }

    with open(
        OUTDIR
        / "pilot_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print("VALIDITY PILOT SUMMARY")
    print("=" * 78)
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("ANCHOR SUMMARY")
    print(
        anchor_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print("FAILURE REASONS")

    if len(invalid):
        print(
            invalid[
                "failure_reason"
            ]
            .value_counts()
            .to_string()
        )
    else:
        print("None")

    print()
    print("Wrote:", OUTDIR)


if __name__ == "__main__":
    main()
