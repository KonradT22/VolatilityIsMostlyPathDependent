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
    THETA1,
    THETA2,
    PARAMETER_BOUNDS,
    INVALID_PENALTY,
    load_spx_history,
    parameters_valid,
)


MARKET_PATH = Path(
    "/users/4/trest017/urop_pdv/data/processed/thetadata/"
    "spxw_2021-06-02_155500_iv_surface_dataset.csv"
)

OUTDIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_market_multi_maturity"
)

TARGET_DTES = [7, 14, 21, 30, 44, 90]

START_BETAS = np.array(
    [0.04, -0.13, 0.65],
    dtype=float,
)

N_PATHS = 4000
TIMESTEP_PER_DAY = 5
SEED_ROOT = 2026080601
SIMULATION_MATURITY = max(TARGET_DTES) / 365.0


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_market():
    df = pd.read_csv(MARKET_PATH)

    df = df[
        (df["liquidity_filter"] == True)
        & (df["dte"].isin(TARGET_DTES))
    ].copy()

    df["side"] = (
        df["otm_right"]
        .astype(str)
        .str.upper()
    )

    df["market_normalized_otm_price"] = (
        df["otm_mid"]
        / (df["discount"] * df["forward"])
    )

    if not np.all(
        np.isfinite(
            df["market_normalized_otm_price"]
        )
    ):
        raise ValueError(
            "Market target contains non-finite prices."
        )

    if (
        df["market_normalized_otm_price"]
        <= 0
    ).any():
        raise ValueError(
            "Market target contains non-positive prices."
        )

    return df.sort_values(
        ["dte", "moneyness"]
    ).reset_index(drop=True)


def build_model(
    betas,
    R_init1,
    R_init2,
):
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
        theta1=THETA1,
        theta2=THETA2,
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


def evaluate_surface(
    betas,
    market,
    R_init1,
    R_init2,
):
    model = build_model(
        betas,
        R_init1,
        R_init2,
    )

    model.simulate(save_R=False)

    vol = to_numpy(
        model.vol_array
    )

    if not np.all(np.isfinite(vol)):
        raise ValueError(
            "nonfinite_volatility"
        )

    if float(np.min(vol)) <= 0.0:
        raise ValueError(
            "nonpositive_volatility"
        )

    output_rows = []

    for dte, slice_df in market.groupby(
        "dte",
        sort=True,
    ):
        maturity = float(dte) / 365.0

        index = int(
            torch.ceil(
                torch.tensor(
                    maturity,
                    dtype=model.timestep.dtype,
                    device=model.device,
                )
                / model.timestep
            )
        )

        model_future = float(
            model.S_array[index]
            .mean()
            .detach()
            .cpu()
        )

        moneyness = (
            slice_df["moneyness"]
            .to_numpy(dtype=float)
        )

        strikes = (
            model_future * moneyness
        )

        _, _, call_prices = (
            model.compute_option_price(
                strikes=strikes,
                option_maturity=maturity,
                return_future=True,
                var_reduction=True,
            )
        )

        call_prices = (
            to_numpy(call_prices)
            .astype(float)
        )

        normalized_calls = (
            call_prices / model_future
        )

        normalized_puts = (
            normalized_calls
            - (1.0 - moneyness)
        )

        sides = (
            slice_df["side"]
            .to_numpy()
        )

        model_otm = np.where(
            sides == "PUT",
            normalized_puts,
            normalized_calls,
        )

        market_otm = (
            slice_df[
                "market_normalized_otm_price"
            ]
            .to_numpy(dtype=float)
        )

        if not np.all(
            np.isfinite(model_otm)
        ):
            raise ValueError(
                "nonfinite_model_option_price"
            )

        for (
            source_row,
            model_price,
            market_price,
        ) in zip(
            slice_df.to_dict(
                orient="records"
            ),
            model_otm,
            market_otm,
        ):
            output_rows.append({
                "dte": int(dte),
                "strike": (
                    source_row["strike"]
                ),
                "market_forward": (
                    source_row["forward"]
                ),
                "moneyness": (
                    source_row["moneyness"]
                ),
                "log_moneyness": (
                    source_row[
                        "log_moneyness"
                    ]
                ),
                "side": (
                    source_row["side"]
                ),
                "market_normalized_otm_price": (
                    market_price
                ),
                "model_normalized_otm_price": (
                    model_price
                ),
                "price_error": (
                    model_price
                    - market_price
                ),
                "model_future": (
                    model_future
                ),
            })

    result = pd.DataFrame(
        output_rows
    )

    errors = (
        result["price_error"]
        .to_numpy(dtype=float)
    )

    global_rmse = float(
        np.sqrt(
            np.mean(errors ** 2)
        )
    )

    per_dte = (
        result.groupby("dte")
        ["price_error"]
        .apply(
            lambda x: float(
                np.sqrt(
                    np.mean(
                        np.asarray(x) ** 2
                    )
                )
            )
        )
    )

    # Equal weight for each maturity so the 90D slice
    # does not dominate simply because it has more quotes.
    equal_maturity_rmse = float(
        per_dte.mean()
    )

    return {
        "rows": result,
        "global_rmse": global_rmse,
        "equal_maturity_rmse": (
            equal_maturity_rmse
        ),
        "per_dte_rmse": (
            per_dte.to_dict()
        ),
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

    market = load_market()

    print(
        "PDV market calibration"
    )
    print(
        "----------------------"
    )
    print(
        "market rows:",
        len(market),
    )
    print(
        "DTEs:",
        sorted(
            market["dte"].unique()
        ),
    )
    print(
        "start betas:",
        START_BETAS,
    )
    print(
        "paths:",
        N_PATHS,
    )
    print()

    market[
        [
            "dte",
            "strike",
            "forward",
            "discount",
            "moneyness",
            "log_moneyness",
            "side",
            "otm_mid",
            "market_normalized_otm_price",
        ]
    ].to_csv(
        OUTDIR
        / "market_target.csv",
        index=False,
    )

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

    evaluation_rows = []
    evaluation_count = 0
    best_score = np.inf

    def objective(candidate):
        nonlocal evaluation_count
        nonlocal best_score

        evaluation_count += 1

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
            global_rmse = np.nan

        else:
            try:
                result = (
                    evaluate_surface(
                        candidate,
                        market,
                        R_init1,
                        R_init2,
                    )
                )

                score = result[
                    "equal_maturity_rmse"
                ]

                global_rmse = result[
                    "global_rmse"
                ]

                reason = ""

            except Exception as exc:
                valid = False
                score = (
                    INVALID_PENALTY
                )
                global_rmse = np.nan
                reason = (
                    f"{type(exc).__name__}:"
                    f"{exc}"
                )

        seconds = (
            time.perf_counter()
            - started
        )

        evaluation_rows.append({
            "evaluation": (
                evaluation_count
            ),
            "beta0": candidate[0],
            "beta1": candidate[1],
            "beta2": candidate[2],
            "objective_rmse": score,
            "global_rmse": (
                global_rmse
            ),
            "valid": valid,
            "reason": reason,
            "seconds": seconds,
        })

        if (
            valid
            and score < best_score
        ):
            best_score = score

            print(
                f"eval "
                f"{evaluation_count:4d} | "
                f"NEW BEST | "
                f"objective="
                f"{score:.8f} | "
                f"betas="
                f"{candidate}"
            )

        elif (
            evaluation_count
            % 10 == 0
        ):
            print(
                f"eval "
                f"{evaluation_count:4d} | "
                f"objective="
                f"{score:.8f} | "
                f"best="
                f"{best_score:.8f}"
            )

        return score

    print(
        "Evaluating starting point..."
    )

    start_result = (
        evaluate_surface(
            START_BETAS,
            market,
            R_init1,
            R_init2,
        )
    )

    print(
        "start equal-maturity RMSE:",
        f"{start_result['equal_maturity_rmse']:.8f}",
    )

    print(
        "start global RMSE:",
        f"{start_result['global_rmse']:.8f}",
    )

    print(
        "start per-DTE RMSE:"
    )

    for dte, score in (
        start_result[
            "per_dte_rmse"
        ].items()
    ):
        print(
            f"  {int(dte):3d}D: "
            f"{score:.8f}"
        )

    print()
    print(
        "Starting Nelder-Mead..."
    )
    print()

    started = (
        time.perf_counter()
    )

    result = minimize(
        objective,
        START_BETAS,
        method="Nelder-Mead",
        options={
            "maxiter": 250,
            "maxfev": 500,
            "xatol": 1e-5,
            "fatol": 1e-8,
            "adaptive": True,
            "disp": True,
        },
    )

    optimization_seconds = (
        time.perf_counter()
        - started
    )

    recovered = np.asarray(
        result.x,
        dtype=float,
    )

    final = evaluate_surface(
        recovered,
        market,
        R_init1,
        R_init2,
    )

    pd.DataFrame(
        evaluation_rows
    ).to_csv(
        OUTDIR
        / "nelder_mead_evaluations.csv",
        index=False,
    )

    final["rows"].to_csv(
        OUTDIR
        / "market_vs_calibrated_pdv.csv",
        index=False,
    )

    per_dte_frame = (
        final["rows"]
        .groupby("dte")
        .agg(
            n_quotes=(
                "price_error",
                "size",
            ),
            rmse=(
                "price_error",
                lambda x: np.sqrt(
                    np.mean(
                        np.asarray(x) ** 2
                    )
                ),
            ),
            mae=(
                "price_error",
                lambda x: np.mean(
                    np.abs(
                        np.asarray(x)
                    )
                ),
            ),
            max_abs_error=(
                "price_error",
                lambda x: np.max(
                    np.abs(
                        np.asarray(x)
                    )
                ),
            ),
        )
        .reset_index()
    )

    per_dte_frame.to_csv(
        OUTDIR
        / "market_calibration_by_dte.csv",
        index=False,
    )

    summary = {
        "success": bool(
            result.success
        ),
        "message": str(
            result.message
        ),
        "start_betas": (
            START_BETAS.tolist()
        ),
        "calibrated_betas": (
            recovered.tolist()
        ),
        "start_equal_maturity_rmse": (
            start_result[
                "equal_maturity_rmse"
            ]
        ),
        "start_global_rmse": (
            start_result[
                "global_rmse"
            ]
        ),
        "final_equal_maturity_rmse": (
            final[
                "equal_maturity_rmse"
            ]
        ),
        "final_global_rmse": (
            final[
                "global_rmse"
            ]
        ),
        "start_per_dte_rmse": (
            start_result[
                "per_dte_rmse"
            ]
        ),
        "final_per_dte_rmse": (
            final[
                "per_dte_rmse"
            ]
        ),
        "iterations": int(
            result.nit
        ),
        "function_evaluations": int(
            result.nfev
        ),
        "optimization_seconds": (
            optimization_seconds
        ),
        "market_rows": int(
            len(market)
        ),
        "N_paths": (
            N_PATHS
        ),
        "seed_root": (
            SEED_ROOT
        ),
        "parameter_bounds": {
            key: list(value)
            for key, value
            in PARAMETER_BOUNDS.items()
        },
    }

    with open(
        OUTDIR
        / "calibration_summary.json",
        "w",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        "=" * 72
    )
    print(
        "MARKET CALIBRATION RESULT"
    )
    print(
        "=" * 72
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        "PER-DTE ERROR:"
    )

    print(
        per_dte_frame.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
