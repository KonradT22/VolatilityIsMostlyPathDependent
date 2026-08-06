import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from calibration.torch_montecarlo import (
    initialize_R,
    identity,
    squared,
)

from scripts.calibration.calibrate_pdv_synthetic_30d import (
    TRUE_BETAS,
    LAM1,
    LAM2,
    N_PATHS,
    SEED_ROOT,
    STRIKES,
    LOG_MONEYNESS,
    OPTION_DTE,
    OPTION_MATURITY,
    OUTDIR as SINGLE_RUN_OUTDIR,
    PARAMETER_BOUNDS,
    INVALID_PENALTY,
    load_spx_history,
    parameters_valid,
    price_surface,
    rmse,
)




OUTDIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_synthetic_30d_multistart"
)

START_POINTS = [
    [0.045, -0.11, 0.60],
    [0.025, -0.15, 0.82],
    [0.055, -0.07, 0.58],
    [0.030, -0.09, 0.75],
    [0.050, -0.14, 0.70],
    [0.035, -0.12, 0.80],
]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    spx = load_spx_history()

    lam1 = torch.tensor(
        LAM1,
        dtype=torch.float32,
    )

    lam2 = torch.tensor(
        LAM2,
        dtype=torch.float32,
    )

    R_init1 = initialize_R(
        lam1,
        past_prices=spx,
        transform=identity,
    )

    R_init2 = initialize_R(
        lam2,
        past_prices=spx,
        transform=squared,
    )

    print("Generating common synthetic target...")

    target = price_surface(
        TRUE_BETAS,
        R_init1,
        R_init2,
    )

    print("True betas:", TRUE_BETAS)
    print("Target generated.")
    print()

    all_runs = []
    all_evaluations = []

    for run_index, start_point in enumerate(
        START_POINTS,
        start=1,
    ):
        start_point = np.asarray(
            start_point,
            dtype=float,
        )

        print("=" * 72)
        print(
            f"RUN {run_index}/{len(START_POINTS)}"
        )
        print("Start:", start_point)
        print("=" * 72)

        evaluation = 0
        best_score = np.inf

        def objective(candidate):
            nonlocal evaluation
            nonlocal best_score

            evaluation += 1

            candidate = np.asarray(
                candidate,
                dtype=float,
            )

            started = time.perf_counter()

            valid, reason = parameters_valid(
                candidate
            )

            if not valid:
                score = INVALID_PENALTY
                result = {
                    "future": np.nan,
                    "min_vol": np.nan,
                    "max_vol": np.nan,
                }

            else:
                try:
                    result = price_surface(
                        candidate,
                        R_init1,
                        R_init2,
                    )

                    score = rmse(
                        result["prices"],
                        target["prices"],
                    )

                    reason = ""

                except Exception as exc:
                    valid = False
                    score = INVALID_PENALTY
                    reason = (
                        f"{type(exc).__name__}:"
                        f"{str(exc)}"
                    )

                    result = {
                        "future": np.nan,
                        "min_vol": np.nan,
                        "max_vol": np.nan,
                    }

            seconds = (
                time.perf_counter()
                - started
            )

            all_evaluations.append({
                "run": run_index,
                "evaluation": evaluation,
                "start_beta0": start_point[0],
                "start_beta1": start_point[1],
                "start_beta2": start_point[2],
                "beta0": candidate[0],
                "beta1": candidate[1],
                "beta2": candidate[2],
                "rmse": score,
                "valid": valid,
                "reason": reason,
                "seconds": seconds,
            })

            if valid and score < best_score:
                best_score = score

            if evaluation % 25 == 0:
                print(
                    f"eval={evaluation:4d} "
                    f"current={score:.10f} "
                    f"best={best_score:.10f}"
                )

            return score

        start_rmse = objective(
            start_point.copy()
        )

        started = time.perf_counter()

        result = minimize(
            objective,
            start_point,
            method="Nelder-Mead",
            options={
                "maxiter": 200,
                "maxfev": 400,
                "xatol": 1e-5,
                "fatol": 1e-8,
                "adaptive": True,
                "disp": False,
            },
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        recovered = np.asarray(
            result.x,
            dtype=float,
        )

        recovered_surface = price_surface(
            recovered,
            R_init1,
            R_init2,
        )

        final_rmse = rmse(
            recovered_surface["prices"],
            target["prices"],
        )

        parameter_error = (
            recovered - TRUE_BETAS
        )

        row = {
            "run": run_index,
            "success": bool(result.success),
            "message": str(result.message),
            "start_beta0": start_point[0],
            "start_beta1": start_point[1],
            "start_beta2": start_point[2],
            "start_rmse": start_rmse,
            "recovered_beta0": recovered[0],
            "recovered_beta1": recovered[1],
            "recovered_beta2": recovered[2],
            "beta0_error": parameter_error[0],
            "beta1_error": parameter_error[1],
            "beta2_error": parameter_error[2],
            "final_price_rmse": final_rmse,
            "iterations": int(result.nit),
            "function_evaluations": int(
                result.nfev
            ),
            "seconds": elapsed,
        }

        all_runs.append(row)

        print()
        print(
            f"Recovered: {recovered}"
        )
        print(
            f"Final RMSE: {final_rmse:.12f}"
        )
        print(
            f"Parameter error: {parameter_error}"
        )
        print(
            f"Evaluations: {result.nfev}"
        )
        print(
            f"Seconds: {elapsed:.2f}"
        )
        print()

    runs = pd.DataFrame(all_runs)

    evaluations = pd.DataFrame(
        all_evaluations
    )

    runs.to_csv(
        OUTDIR / "multistart_summary.csv",
        index=False,
    )

    evaluations.to_csv(
        OUTDIR / "multistart_evaluations.csv",
        index=False,
    )

    recovered_columns = [
        "recovered_beta0",
        "recovered_beta1",
        "recovered_beta2",
    ]

    dispersion = {
        column: {
            "min": float(runs[column].min()),
            "max": float(runs[column].max()),
            "mean": float(runs[column].mean()),
            "std": float(runs[column].std()),
        }
        for column in recovered_columns
    }

    summary = {
        "true_betas": TRUE_BETAS.tolist(),
        "runs": len(runs),
        "successful_runs": int(
            runs["success"].sum()
        ),
        "best_price_rmse": float(
            runs["final_price_rmse"].min()
        ),
        "worst_price_rmse": float(
            runs["final_price_rmse"].max()
        ),
        "mean_price_rmse": float(
            runs["final_price_rmse"].mean()
        ),
        "parameter_dispersion": dispersion,
        "N_paths": N_PATHS,
        "seed_root": SEED_ROOT,
        "dte": OPTION_DTE,
        "strike_count": len(STRIKES),
    }

    with open(
        OUTDIR / "multistart_summary.json",
        "w",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        240,
    )

    print()
    print("=" * 72)
    print("MULTI-START RESULTS")
    print("=" * 72)

    print(
        runs[
            [
                "run",
                "start_beta0",
                "start_beta1",
                "start_beta2",
                "recovered_beta0",
                "recovered_beta1",
                "recovered_beta2",
                "final_price_rmse",
                "function_evaluations",
                "seconds",
            ]
        ]
    )

    print()
    print("Parameter dispersion:")
    print(
        json.dumps(
            dispersion,
            indent=2,
        )
    )

    print()
    print(
        "Wrote:",
        OUTDIR / "multistart_summary.csv",
    )
    print(
        "Wrote:",
        OUTDIR / "multistart_evaluations.csv",
    )
    print(
        "Wrote:",
        OUTDIR / "multistart_summary.json",
    )


if __name__ == "__main__":
    main()
