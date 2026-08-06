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
    TorchMonteCarloExponentialModel,
    initialize_R,
    identity,
    squared,
)

from scripts.calibration.calibrate_pdv_synthetic_30d import (
    LAM1,
    LAM2,
    INVALID_PENALTY,
    load_spx_history,
    rmse,
)


OUTDIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_synthetic_5param"
)

TARGET_DTES = [7, 14, 21, 30, 44, 90]

# [beta0, beta1, beta2, theta1, theta2]
TRUE_PARAMS = np.array(
    [0.04, -0.13, 0.65, 0.25, 0.50],
    dtype=float,
)

PARAMETER_BOUNDS = {
    "beta0": (0.02, 0.06),
    "beta1": (-0.16, -0.06),
    "beta2": (0.55, 0.85),
    "theta1": (0.0, 1.0),
    "theta2": (0.0, 1.0),
}

LOG_MONEYNESS = np.linspace(
    -0.04,
    0.04,
    17,
)

STRIKES = np.exp(LOG_MONEYNESS)

N_PATHS = 4000
TIMESTEP_PER_DAY = 5
SEED_ROOT = 2026080601

SIMULATION_MATURITY = (
    max(TARGET_DTES) / 365.0
)

START_POINTS = [
    [0.045, -0.11, 0.60, 0.30, 0.42],
    [0.030, -0.15, 0.80, 0.15, 0.65],
    [0.055, -0.08, 0.58, 0.45, 0.30],
    [0.035, -0.12, 0.75, 0.20, 0.70],
    [0.050, -0.145, 0.70, 0.40, 0.55],
]


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return (
            value.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(value)



def parameters_valid(params):
    params = np.asarray(params, dtype=float)

    if params.shape != (5,):
        return False, "wrong_parameter_count"

    if not np.all(np.isfinite(params)):
        return False, "nonfinite_parameter"

    names = [
        "beta0",
        "beta1",
        "beta2",
        "theta1",
        "theta2",
    ]

    for name, value in zip(names, params):
        lower, upper = PARAMETER_BOUNDS[name]

        if not (lower <= value <= upper):
            return False, f"{name}_out_of_bounds"

    return True, ""



def build_model(
    params,
    R_init1,
    R_init2,
):
    beta0, beta1, beta2, theta1, theta2 = params
    betas = [beta0, beta1, beta2]
    return TorchMonteCarloExponentialModel(
        lam1=torch.tensor(
            LAM1,
            dtype=torch.float32,
        ),
        lam2=torch.tensor(
            LAM2,
            dtype=torch.float32,
        ),
        betas=torch.tensor(
            betas,
            dtype=torch.float32,
        ),
        R_init1=R_init1,
        R_init2=R_init2,
        theta1=theta1,
        theta2=theta2,
        N=N_PATHS,
        vix_N=1000,
        maturity=SIMULATION_MATURITY,
        timestep_per_day=TIMESTEP_PER_DAY,
        fixed_seed=True,
        seed_root=SEED_ROOT,
        parabolic=0.0,
        parabolic_offset=0.0,
        vol_cap=1.5,
        device="cpu",
    )


def price_multi_maturity(
    params,
    R_init1,
    R_init2,
):
    model = build_model(
        params,
        R_init1,
        R_init2,
    )

    model.simulate(
        save_R=False
    )

    vol = to_numpy(
        model.vol_array
    )

    if not np.all(
        np.isfinite(vol)
    ):
        raise ValueError(
            "nonfinite_volatility"
        )

    if float(
        np.min(vol)
    ) <= 0.0:
        raise ValueError(
            "nonpositive_volatility"
        )

    rows = []
    flattened_prices = []

    for dte in TARGET_DTES:
        maturity = (
            dte / 365.0
        )

        future, _, prices = (
            model.compute_option_price(
                strikes=STRIKES,
                option_maturity=maturity,
                return_future=True,
                var_reduction=True,
            )
        )

        prices = (
            to_numpy(prices)
            .astype(float)
        )

        if not np.all(
            np.isfinite(prices)
        ):
            raise ValueError(
                "nonfinite_option_price"
            )

        flattened_prices.extend(
            prices.tolist()
        )

        for log_m, strike, price in zip(
            LOG_MONEYNESS,
            STRIKES,
            prices,
        ):
            rows.append({
                "dte": dte,
                "future": float(
                    to_numpy(future)
                ),
                "log_moneyness": (
                    log_m
                ),
                "strike": strike,
                "option_price": price,
            })

    return {
        "prices": np.asarray(
            flattened_prices,
            dtype=float,
        ),
        "rows": rows,
        "min_vol": float(
            np.min(vol)
        ),
        "max_vol": float(
            np.max(vol)
        ),
    }


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    spx = (
        load_spx_history()
    )

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

    print(
        "Generating six-maturity "
        "synthetic target..."
    )

    target = (
        price_multi_maturity(
            TRUE_PARAMS,
            R_init1,
            R_init2,
        )
    )

    pd.DataFrame(
        target["rows"]
    ).to_csv(
        OUTDIR
        / "synthetic_target_prices.csv",
        index=False,
    )

    print(
        "True betas:",
        TRUE_PARAMS,
    )

    print(
        "Target prices:",
        len(target["prices"]),
    )

    print(
        "Vol range:",
        target["min_vol"],
        "to",
        target["max_vol"],
    )

    print()

    run_rows = []
    evaluation_rows = []

    for run_number, start in enumerate(
        START_POINTS,
        start=1,
    ):
        start = np.asarray(
            start,
            dtype=float,
        )

        print(
            "=" * 72
        )

        print(
            f"RUN {run_number}/"
            f"{len(START_POINTS)}"
        )

        print(
            "Start:",
            start,
        )

        print(
            "=" * 72
        )

        evaluation = 0
        best = np.inf

        def objective(candidate):
            nonlocal evaluation
            nonlocal best

            evaluation += 1

            candidate = np.asarray(
                candidate,
                dtype=float,
            )

            started = (
                time.perf_counter()
            )

            valid, reason = (
                parameters_valid(
                    candidate
                )
            )

            if not valid:
                score = (
                    INVALID_PENALTY
                )

            else:
                try:
                    result = (
                        price_multi_maturity(
                            candidate,
                            R_init1,
                            R_init2,
                        )
                    )

                    score = rmse(
                        result["prices"],
                        target["prices"],
                    )

                    reason = ""

                except Exception as exc:
                    valid = False
                    score = (
                        INVALID_PENALTY
                    )

                    reason = (
                        f"{type(exc).__name__}:"
                        f"{exc}"
                    )

            seconds = (
                time.perf_counter()
                - started
            )

            evaluation_rows.append({
                "run": run_number,
                "evaluation": evaluation,
                "beta0": candidate[0],
                "beta1": candidate[1],
                "beta2": candidate[2],
                "theta1": candidate[3],
                "theta2": candidate[4],
                "rmse": score,
                "valid": valid,
                "reason": reason,
                "seconds": seconds,
            })

            if (
                valid
                and score < best
            ):
                best = score

            if (
                evaluation % 25
                == 0
            ):
                print(
                    f"eval={evaluation:4d} "
                    f"current="
                    f"{score:.10f} "
                    f"best="
                    f"{best:.10f}"
                )

            return score

        start_rmse = (
            objective(
                start.copy()
            )
        )

        started = (
            time.perf_counter()
        )

        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={
                "maxiter": 250,
                "maxfev": 500,
                "xatol": 1e-6,
                "fatol": 1e-9,
                "adaptive": True,
                "disp": False,
            },
        )

        seconds = (
            time.perf_counter()
            - started
        )

        recovered = (
            np.asarray(
                result.x,
                dtype=float,
            )
        )

        recovered_surface = (
            price_multi_maturity(
                recovered,
                R_init1,
                R_init2,
            )
        )

        final_rmse = rmse(
            recovered_surface[
                "prices"
            ],
            target["prices"],
        )

        parameter_error = (
            recovered
            - TRUE_PARAMS
        )

        run_rows.append({
            "run": run_number,
            "success": bool(
                result.success
            ),
            "start_beta0": start[0],
            "start_beta1": start[1],
            "start_beta2": start[2],
            "start_theta1": start[3],
            "start_theta2": start[4],
            "start_rmse": start_rmse,
            "recovered_beta0": recovered[0],
            "recovered_beta1": recovered[1],
            "recovered_beta2": recovered[2],
            "recovered_theta1": recovered[3],
            "recovered_theta2": recovered[4],
            "beta0_error": parameter_error[0],
            "beta1_error": parameter_error[1],
            "beta2_error": parameter_error[2],
            "theta1_error": parameter_error[3],
            "theta2_error": parameter_error[4],
            "final_price_rmse": (
                final_rmse
            ),
            "iterations": int(
                result.nit
            ),
            "function_evaluations": (
                int(result.nfev)
            ),
            "seconds": seconds,
        })

        print(
            "Recovered:",
            recovered,
        )

        print(
            "Final RMSE:",
            f"{final_rmse:.12f}",
        )

        print(
            "Parameter error:",
            parameter_error,
        )

        print()

    runs = pd.DataFrame(
        run_rows
    )

    evaluations = pd.DataFrame(
        evaluation_rows
    )

    runs.to_csv(
        OUTDIR
        / "multistart_summary.csv",
        index=False,
    )

    evaluations.to_csv(
        OUTDIR
        / "multistart_evaluations.csv",
        index=False,
    )

    summary = {
        "true_betas": (
            TRUE_PARAMS.tolist()
        ),
        "target_dtes": (
            TARGET_DTES
        ),
        "price_vector_length": (
            len(target["prices"])
        ),
        "successful_runs": int(
            runs["success"].sum()
        ),
        "best_rmse": float(
            runs[
                "final_price_rmse"
            ].min()
        ),
        "worst_rmse": float(
            runs[
                "final_price_rmse"
            ].max()
        ),
    }

    with open(
        OUTDIR
        / "multistart_summary.json",
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
    print(
        "=" * 72
    )

    print(
        "MULTI-MATURITY RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        runs[
            [
                "run",
                "recovered_beta0",
                "recovered_beta1",
                "recovered_beta2",
                "recovered_theta1",
                "recovered_theta2",
                "beta0_error",
                "beta1_error",
                "beta2_error",
                "theta1_error",
                "theta2_error",
                "final_price_rmse",
                "function_evaluations",
                "seconds",
            ]
        ]
    )


if __name__ == "__main__":
    main()
