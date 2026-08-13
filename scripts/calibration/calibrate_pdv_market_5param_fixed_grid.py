import argparse
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
)


torch.set_num_threads(1)


TARGET_PATH = Path(
    "/users/4/trest017/urop_pdv/data/processed/thetadata/"
    "spxw_2021-05-05_2021-06-02_155500_fixed_grid_targets.csv"
)

SPOT_PROXY_PATH = Path(
    "/users/4/trest017/urop_pdv/data/processed/thetadata/"
    "spxw_2021-05-05_2021-06-02_155500_spot_proxy.csv"
)

SPX_HISTORY_PATH = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "msi_baseline/spx_history_yfinance.csv"
)

BASE_OUTDIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_market_5param_fixed_grid"
)


PARAMETER_BOUNDS = {
    "beta0": (0.02, 0.06),
    "beta1": (-0.16, -0.06),
    "beta2": (0.55, 0.85),
    "theta1": (0.0, 1.0),
    "theta2": (0.0, 1.0),
}

DEFAULT_START = np.array(
    [
        0.0364937645,
        -0.1285527876,
        0.7375579398,
        0.25,
        0.50,
    ],
    dtype=float,
)

TIMESTEP_PER_DAY = 5
DEFAULT_N_PATHS = 4000
DEFAULT_SEED_ROOT = 2026080701
BATCH_SIZE = 8


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def parse_params(value):
    params = np.array(
        [float(x.strip()) for x in value.split(",")],
        dtype=float,
    )

    if len(params) != 5:
        raise ValueError(
            "--start-params must contain exactly 5 numbers"
        )

    return params


def parameters_valid(params):
    params = np.asarray(params, dtype=float)

    if len(params) != 5:
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


def load_targets(trade_date):
    df = pd.read_csv(TARGET_PATH)

    df = df[
        df["trade_date"].astype(str) == trade_date
    ].copy()

    if len(df) != 77:
        raise RuntimeError(
            f"{trade_date}: expected 77 targets, found {len(df)}"
        )

    if df["target_dte"].nunique() != 7:
        raise RuntimeError(
            f"{trade_date}: expected 7 target tenors"
        )

    if not np.all(
        np.isfinite(df["normalized_otm_price"])
    ):
        raise RuntimeError("nonfinite_market_targets")

    if (df["normalized_otm_price"] <= 0).any():
        raise RuntimeError("nonpositive_market_targets")

    return (
        df.sort_values(
            ["target_dte", "grid_index"]
        )
        .reset_index(drop=True)
    )


def load_state_history(trade_date):
    trade_ts = pd.Timestamp(trade_date)

    history = pd.read_csv(SPX_HISTORY_PATH)

    date_col = history.columns[0]
    history[date_col] = pd.to_datetime(history[date_col])

    series = pd.Series(
        history["Close"].to_numpy(dtype=float),
        index=history[date_col],
    )

    series.index = pd.to_datetime(series.index.date)

    # Strictly earlier closes only.
    series = series.loc[
        series.index < trade_ts
    ].copy()

    proxies = pd.read_csv(SPOT_PROXY_PATH)

    row = proxies[
        proxies["trade_date"].astype(str) == trade_date
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"{trade_date}: expected one spot proxy row"
        )

    proxy = float(row["spx_1555_proxy"].iloc[0])

    if not np.isfinite(proxy) or proxy <= 0:
        raise RuntimeError(
            f"{trade_date}: invalid SPX proxy"
        )

    series.loc[trade_ts] = proxy
    series = series.sort_index()

    return series, proxy


def build_model(
    params,
    R_init1,
    R_init2,
    simulation_maturity,
    n_paths,
    seed_root,
):
    beta0, beta1, beta2, theta1, theta2 = params

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
            [beta0, beta1, beta2],
            dtype=torch.float32,
        ),
        R_init1=R_init1,
        R_init2=R_init2,
        theta1=float(theta1),
        theta2=float(theta2),
        N=n_paths,
        vix_N=1000,
        maturity=simulation_maturity,
        timestep_per_day=TIMESTEP_PER_DAY,
        fixed_seed=True,
        seed_root=seed_root,
        parabolic=0.0,
        parabolic_offset=0.0,
        vol_cap=1.5,
        device="cpu",
    )


def evaluate_surface(
    params,
    market,
    R_init1,
    R_init2,
    n_paths,
    seed_root,
):
    max_actual_dte = int(
        market["actual_dte"].max()
    )

    # Simulate one calendar day beyond the longest requested
    # maturity. The model and pricing code use slightly different
    # integer rounding when mapping maturity to timestep indices;
    # the cushion guarantees the requested pricing index exists.
    simulation_maturity = (
        (max_actual_dte + 1) / 365.0
    )

    model = build_model(
        params=params,
        R_init1=R_init1,
        R_init2=R_init2,
        simulation_maturity=simulation_maturity,
        n_paths=n_paths,
        seed_root=seed_root,
    )

    model.simulate(save_R=False)

    vol = to_numpy(model.vol_array)

    if not np.all(np.isfinite(vol)):
        raise ValueError("nonfinite_volatility")

    if float(np.min(vol)) <= 0:
        raise ValueError("nonpositive_volatility")

    rows = []

    for target_dte, g in market.groupby(
        "target_dte",
        sort=True,
    ):
        actual_dtes = g["actual_dte"].unique()

        if len(actual_dtes) != 1:
            raise RuntimeError(
                f"target {target_dte}: multiple actual DTEs"
            )

        actual_dte = int(actual_dtes[0])
        maturity = actual_dte / 365.0

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

        moneyness = g[
            "moneyness"
        ].to_numpy(dtype=float)

        strikes = (
            model_future * moneyness
        )

        price_chunks = []

        for start in range(
            0,
            len(strikes),
            BATCH_SIZE,
        ):
            stop = min(
                start + BATCH_SIZE,
                len(strikes),
            )

            _, _, prices = (
                model.compute_option_price(
                    strikes=strikes[start:stop],
                    option_maturity=maturity,
                    return_future=True,
                    var_reduction=True,
                )
            )

            price_chunks.append(
                to_numpy(prices).astype(float)
            )

        call_prices = np.concatenate(
            price_chunks
        )

        normalized_calls = (
            call_prices / model_future
        )

        normalized_puts = (
            normalized_calls
            - (1.0 - moneyness)
        )

        sides = (
            g["otm_right"]
            .astype(str)
            .str.upper()
            .to_numpy()
        )

        model_otm = np.where(
            sides == "PUT",
            normalized_puts,
            normalized_calls,
        )

        market_otm = g[
            "normalized_otm_price"
        ].to_numpy(dtype=float)

        if not np.all(np.isfinite(model_otm)):
            raise ValueError(
                "nonfinite_model_option_price"
            )

        for source, model_price, market_price in zip(
            g.to_dict(orient="records"),
            model_otm,
            market_otm,
        ):
            rows.append({
                "trade_date": source["trade_date"],
                "target_dte": int(target_dte),
                "actual_dte": actual_dte,
                "grid_index": int(
                    source["grid_index"]
                ),
                "log_moneyness": float(
                    source["log_moneyness"]
                ),
                "moneyness": float(
                    source["moneyness"]
                ),
                "side": source["otm_right"],
                "market_normalized_otm_price":
                    float(market_price),
                "model_normalized_otm_price":
                    float(model_price),
                "price_error":
                    float(model_price - market_price),
                "model_future": model_future,
            })

    result = pd.DataFrame(rows)

    errors = (
        result["price_error"]
        .to_numpy(dtype=float)
    )

    global_rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    per_tenor = (
        result.groupby("target_dte")
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

    equal_tenor_rmse = float(
        per_tenor.mean()
    )

    return {
        "rows": result,
        "global_rmse": global_rmse,
        "equal_tenor_rmse": equal_tenor_rmse,
        "per_tenor_rmse": per_tenor.to_dict(),
        "min_vol": float(np.min(vol)),
        "max_vol": float(np.max(vol)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trade-date",
        required=True,
    )

    parser.add_argument(
        "--start-params",
        default=",".join(
            str(x) for x in DEFAULT_START
        ),
    )

    parser.add_argument(
        "--n-paths",
        type=int,
        default=DEFAULT_N_PATHS,
    )

    parser.add_argument(
        "--seed-root",
        type=int,
        default=DEFAULT_SEED_ROOT,
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=350,
    )

    parser.add_argument(
        "--maxfev",
        type=int,
        default=700,
    )

    args = parser.parse_args()

    trade_date = args.trade_date
    start_params = parse_params(
        args.start_params
    )

    outdir = (
        BASE_OUTDIR
        / trade_date
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    market = load_targets(
        trade_date
    )

    history, spot_proxy = (
        load_state_history(
            trade_date
        )
    )

    lam1 = torch.tensor(
        LAM1,
        dtype=torch.float32,
    )

    lam2 = torch.tensor(
        LAM2,
        dtype=torch.float32,
    )

    state_max_delta = min(
        1000,
        len(history) - 1,
    )

    if state_max_delta < 1:
        raise RuntimeError(
            "Insufficient SPX history for state initialization"
        )

    R_init1 = initialize_R(
        lam1,
        past_prices=history,
        max_delta=state_max_delta,
        transform=identity,
    )

    R_init2 = initialize_R(
        lam2,
        past_prices=history,
        max_delta=state_max_delta,
        transform=squared,
    )

    print("=" * 78)
    print("5-PARAMETER PDV MARKET CALIBRATION")
    print("=" * 78)

    print("trade date:", trade_date)
    print("targets:", len(market))
    print(
        "actual DTEs:",
        sorted(
            market["actual_dte"]
            .unique()
            .tolist()
        ),
    )
    print("paths:", args.n_paths)
    print("seed root:", args.seed_root)
    print("SPX 15:55 proxy:", spot_proxy)
    print(
        "history observations:",
        len(history),
    )
    print(
        "R_init1:",
        to_numpy(R_init1),
    )
    print(
        "R_init2:",
        to_numpy(R_init2),
    )
    print("start:", start_params)
    print()

    valid, reason = parameters_valid(
        start_params
    )

    if not valid:
        raise RuntimeError(
            f"Invalid starting parameters: {reason}"
        )

    start_result = evaluate_surface(
        params=start_params,
        market=market,
        R_init1=R_init1,
        R_init2=R_init2,
        n_paths=args.n_paths,
        seed_root=args.seed_root,
    )

    print(
        "start global RMSE:",
        f"{start_result['global_rmse']:.10f}",
    )
    print(
        "start equal-tenor RMSE:",
        f"{start_result['equal_tenor_rmse']:.10f}",
    )

    print("start per-tenor RMSE:")

    for dte, score in (
        start_result[
            "per_tenor_rmse"
        ].items()
    ):
        print(
            f"  {int(dte):3d}: "
            f"{score:.10f}"
        )

    print()

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

        started = time.perf_counter()

        valid, reason = (
            parameters_valid(candidate)
        )

        global_rmse = np.nan

        if not valid:
            score = INVALID_PENALTY

        else:
            try:
                result = evaluate_surface(
                    params=candidate,
                    market=market,
                    R_init1=R_init1,
                    R_init2=R_init2,
                    n_paths=args.n_paths,
                    seed_root=args.seed_root,
                )

                score = result[
                    "global_rmse"
                ]

                global_rmse = score
                reason = ""

            except Exception as exc:
                valid = False
                score = INVALID_PENALTY
                reason = (
                    f"{type(exc).__name__}:"
                    f"{exc}"
                )

        elapsed = (
            time.perf_counter()
            - started
        )

        evaluation_rows.append({
            "evaluation":
                evaluation_count,
            "beta0": candidate[0],
            "beta1": candidate[1],
            "beta2": candidate[2],
            "theta1": candidate[3],
            "theta2": candidate[4],
            "objective_rmse": score,
            "global_rmse":
                global_rmse,
            "valid": valid,
            "reason": reason,
            "seconds": elapsed,
        })

        if (
            valid
            and score < best_score
        ):
            best_score = score

            print(
                f"eval={evaluation_count:4d} "
                f"NEW BEST "
                f"rmse={score:.10f} "
                f"params={candidate}",
                flush=True,
            )

        elif evaluation_count % 10 == 0:
            print(
                f"eval={evaluation_count:4d} "
                f"rmse={score:.10f} "
                f"best={best_score:.10f}",
                flush=True,
            )

        return score

    print("\nStarting Nelder-Mead...\n")

    started = time.perf_counter()

    result = minimize(
        objective,
        start_params,
        method="Nelder-Mead",
        options={
            "maxiter": args.maxiter,
            "maxfev": args.maxfev,
            "xatol": 1e-5,
            "fatol": 1e-8,
            "adaptive": True,
            "disp": False,
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
        params=recovered,
        market=market,
        R_init1=R_init1,
        R_init2=R_init2,
        n_paths=args.n_paths,
        seed_root=args.seed_root,
    )

    evaluations = pd.DataFrame(
        evaluation_rows
    )

    evaluations.to_csv(
        outdir
        / "nelder_mead_evaluations.csv",
        index=False,
    )

    final["rows"].to_csv(
        outdir
        / "market_vs_calibrated_pdv.csv",
        index=False,
    )

    per_tenor = pd.DataFrame(
        [
            {
                "target_dte": int(k),
                "rmse": float(v),
            }
            for k, v
            in final[
                "per_tenor_rmse"
            ].items()
        ]
    )

    per_tenor.to_csv(
        outdir
        / "per_tenor_rmse.csv",
        index=False,
    )

    summary = {
        "trade_date": trade_date,
        "success": bool(
            result.success
        ),
        "optimizer_message":
            str(result.message),
        "start_params":
            start_params.tolist(),
        "recovered_params":
            recovered.tolist(),
        "start_global_rmse":
            float(
                start_result[
                    "global_rmse"
                ]
            ),
        "final_global_rmse":
            float(
                final[
                    "global_rmse"
                ]
            ),
        "start_equal_tenor_rmse":
            float(
                start_result[
                    "equal_tenor_rmse"
                ]
            ),
        "final_equal_tenor_rmse":
            float(
                final[
                    "equal_tenor_rmse"
                ]
            ),
        "iterations": int(result.nit),
        "function_evaluations":
            int(result.nfev),
        "optimization_seconds":
            float(
                optimization_seconds
            ),
        "mean_evaluation_seconds":
            float(
                evaluations["seconds"]
                .mean()
            ),
        "N_paths": args.n_paths,
        "seed_root": args.seed_root,
        "spx_1555_proxy":
            float(spot_proxy),
        "history_observations":
            int(len(history)),
        "R_init1":
            to_numpy(
                R_init1
            ).tolist(),
        "R_init2":
            to_numpy(
                R_init2
            ).tolist(),
        "actual_dtes": sorted(
            int(x)
            for x in
            market[
                "actual_dte"
            ].unique()
        ),
        "parameter_bounds": {
            key: list(value)
            for key, value
            in PARAMETER_BOUNDS.items()
        },
    }

    with open(
        outdir
        / "calibration_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print("CALIBRATION RESULT")
    print("=" * 78)
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
