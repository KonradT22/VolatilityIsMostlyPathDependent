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


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

ASOFDATE = pd.Timestamp("2021-06-02")

SPX_HISTORY_PATH = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "msi_baseline/spx_history_yfinance.csv"
)

OUTDIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_synthetic_30d"
)

TRUE_BETAS = np.array(
    [0.04, -0.13, 0.65],
    dtype=float,
)

START_BETAS = np.array(
    [0.045, -0.11, 0.60],
    dtype=float,
)

LAM1 = np.array([55.0, 10.0])
LAM2 = np.array([20.0, 3.0])

THETA1 = 0.25
THETA2 = 0.50

OPTION_DTE = 30
OPTION_MATURITY = OPTION_DTE / 365.0

# We only need to simulate through the option maturity.
SIMULATION_MATURITY = OPTION_MATURITY

TIMESTEP_PER_DAY = 5

# Pilot calibration path count.
# Common random numbers make the objective deterministic.
N_PATHS = 4000

SEED_ROOT = 2026080601

LOG_MONEYNESS = np.linspace(-0.04, 0.04, 17)
STRIKES = np.exp(LOG_MONEYNESS)

# These are pilot/calibration-development bounds.
# They are NOT being claimed as final theoretical PDV bounds.
PARAMETER_BOUNDS = {
    "beta0": (0.02, 0.06),
    "beta1": (-0.16, -0.06),
    "beta2": (0.55, 0.85),
}

INVALID_PENALTY = 1.0e3


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def load_spx_history():
    if not SPX_HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Missing cached SPX history: {SPX_HISTORY_PATH}"
        )

    frame = pd.read_csv(SPX_HISTORY_PATH)

    date_column = frame.columns[0]
    frame[date_column] = pd.to_datetime(frame[date_column])

    series = pd.Series(
        frame["Close"].to_numpy(),
        index=frame[date_column],
    )

    series.index = pd.to_datetime(series.index.date)

    return series.loc[series.index <= ASOFDATE]


def parameters_valid(betas):
    beta0, beta1, beta2 = map(float, betas)

    if not np.all(np.isfinite(betas)):
        return False, "nonfinite_parameter"

    checks = [
        (
            PARAMETER_BOUNDS["beta0"][0]
            <= beta0
            <= PARAMETER_BOUNDS["beta0"][1]
        ),
        (
            PARAMETER_BOUNDS["beta1"][0]
            <= beta1
            <= PARAMETER_BOUNDS["beta1"][1]
        ),
        (
            PARAMETER_BOUNDS["beta2"][0]
            <= beta2
            <= PARAMETER_BOUNDS["beta2"][1]
        ),
    ]

    if not checks[0]:
        return False, "beta0_out_of_bounds"

    if not checks[1]:
        return False, "beta1_out_of_bounds"

    if not checks[2]:
        return False, "beta2_out_of_bounds"

    return True, ""


def build_model(betas, R_init1, R_init2):
    return TorchMonteCarloExponentialModel(
        lam1=torch.tensor(LAM1, dtype=torch.float32),
        lam2=torch.tensor(LAM2, dtype=torch.float32),
        betas=torch.tensor(betas, dtype=torch.float32),
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


def price_surface(betas, R_init1, R_init2):
    model = build_model(
        betas,
        R_init1,
        R_init2,
    )

    model.simulate(save_R=False)

    vol_array = to_numpy(model.vol_array)

    if not np.all(np.isfinite(vol_array)):
        raise ValueError("nonfinite_volatility")

    min_vol = float(np.min(vol_array))
    max_vol = float(np.max(vol_array))

    if min_vol <= 0.0:
        raise ValueError("nonpositive_volatility")

    future, _, prices = model.compute_option_price(
        strikes=STRIKES,
        option_maturity=OPTION_MATURITY,
        return_future=True,
        var_reduction=True,
    )

    prices = to_numpy(prices).astype(float)

    if not np.all(np.isfinite(prices)):
        raise ValueError("nonfinite_option_price")

    return {
        "future": float(to_numpy(future)),
        "prices": prices,
        "min_vol": min_vol,
        "max_vol": max_vol,
    }


def rmse(left, right):
    return float(
        np.sqrt(
            np.mean(
                (np.asarray(left) - np.asarray(right)) ** 2
            )
        )
    )


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    spx = load_spx_history()

    lam1_tensor = torch.tensor(
        LAM1,
        dtype=torch.float32,
    )

    lam2_tensor = torch.tensor(
        LAM2,
        dtype=torch.float32,
    )

    R_init1 = initialize_R(
        lam1_tensor,
        past_prices=spx,
        transform=identity,
    )

    R_init2 = initialize_R(
        lam2_tensor,
        past_prices=spx,
        transform=squared,
    )

    print("PDV synthetic 30D calibration")
    print("--------------------------------")
    print("true betas: ", TRUE_BETAS)
    print("start betas:", START_BETAS)
    print("paths:", N_PATHS)
    print("strike count:", len(STRIKES))
    print("seed root:", SEED_ROOT)
    print()

    # ---------------------------------------------------------------
    # Generate target exactly once.
    # ---------------------------------------------------------------

    print("Generating fixed synthetic target...")

    target_start = time.perf_counter()

    target = price_surface(
        TRUE_BETAS,
        R_init1,
        R_init2,
    )

    target_seconds = (
        time.perf_counter()
        - target_start
    )

    print(
        f"Target generated in {target_seconds:.3f} s"
    )

    target_frame = pd.DataFrame({
        "dte": OPTION_DTE,
        "log_moneyness": LOG_MONEYNESS,
        "strike": STRIKES,
        "target_option_price": target["prices"],
    })

    target_frame.to_csv(
        OUTDIR / "synthetic_target_prices.csv",
        index=False,
    )

    print(
        "target future:",
        f'{target["future"]:.8f}',
    )

    print(
        "target volatility range:",
        f'{target["min_vol"]:.6f}',
        "to",
        f'{target["max_vol"]:.6f}',
    )

    print()

    # ---------------------------------------------------------------
    # Objective function
    # ---------------------------------------------------------------

    history = []
    evaluation_counter = 0
    best_rmse = np.inf

    def objective(candidate):
        nonlocal evaluation_counter
        nonlocal best_rmse

        evaluation_counter += 1

        candidate = np.asarray(
            candidate,
            dtype=float,
        )

        started = time.perf_counter()

        valid, reason = parameters_valid(
            candidate
        )

        if not valid:
            elapsed = time.perf_counter() - started

            history.append({
                "evaluation": evaluation_counter,
                "beta0": candidate[0],
                "beta1": candidate[1],
                "beta2": candidate[2],
                "rmse": INVALID_PENALTY,
                "valid": False,
                "reason": reason,
                "future": np.nan,
                "min_vol": np.nan,
                "max_vol": np.nan,
                "seconds": elapsed,
            })

            return INVALID_PENALTY

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
            valid = True

        except Exception as exc:
            score = INVALID_PENALTY
            valid = False
            reason = (
                f"{type(exc).__name__}:"
                f"{str(exc)}"
            )

            result = {
                "future": np.nan,
                "min_vol": np.nan,
                "max_vol": np.nan,
            }

        elapsed = time.perf_counter() - started

        history.append({
            "evaluation": evaluation_counter,
            "beta0": candidate[0],
            "beta1": candidate[1],
            "beta2": candidate[2],
            "rmse": score,
            "valid": valid,
            "reason": reason,
            "future": result["future"],
            "min_vol": result["min_vol"],
            "max_vol": result["max_vol"],
            "seconds": elapsed,
        })

        if valid and score < best_rmse:
            best_rmse = score

            print(
                f"eval {evaluation_counter:4d} | "
                f"NEW BEST | "
                f"rmse={score:.10f} | "
                f"betas={candidate}"
            )

        elif evaluation_counter % 10 == 0:
            print(
                f"eval {evaluation_counter:4d} | "
                f"rmse={score:.10f} | "
                f"best={best_rmse:.10f}"
            )

        return score

    # ---------------------------------------------------------------
    # Sanity checks before optimization
    # ---------------------------------------------------------------

    truth_rmse = objective(
        TRUE_BETAS.copy()
    )

    start_rmse = objective(
        START_BETAS.copy()
    )

    print()
    print(
        "RMSE at true parameters:",
        f"{truth_rmse:.12f}",
    )

    print(
        "RMSE at starting parameters:",
        f"{start_rmse:.12f}",
    )

    print()

    if truth_rmse > 1e-8:
        raise RuntimeError(
            "Known true parameters did not reproduce "
            "the fixed target. Common-random-number "
            "test failed."
        )

    # ---------------------------------------------------------------
    # Nelder-Mead calibration
    # ---------------------------------------------------------------

    print("Starting Nelder-Mead...")
    print()

    optimization_start = time.perf_counter()

    result = minimize(
        objective,
        START_BETAS,
        method="Nelder-Mead",
        options={
            "maxiter": 200,
            "maxfev": 400,
            "xatol": 1e-5,
            "fatol": 1e-8,
            "adaptive": True,
            "disp": True,
        },
    )

    optimization_seconds = (
        time.perf_counter()
        - optimization_start
    )

    recovered_betas = np.asarray(
        result.x,
        dtype=float,
    )

    recovered_surface = price_surface(
        recovered_betas,
        R_init1,
        R_init2,
    )

    final_price_rmse = rmse(
        recovered_surface["prices"],
        target["prices"],
    )

    parameter_error = (
        recovered_betas
        - TRUE_BETAS
    )

    absolute_parameter_error = np.abs(
        parameter_error
    )

    # ---------------------------------------------------------------
    # Save evaluation history
    # ---------------------------------------------------------------

    history_frame = pd.DataFrame(history)

    history_frame.to_csv(
        OUTDIR / "nelder_mead_evaluation_history.csv",
        index=False,
    )

    comparison_frame = pd.DataFrame({
        "dte": OPTION_DTE,
        "log_moneyness": LOG_MONEYNESS,
        "strike": STRIKES,
        "target_option_price": target["prices"],
        "recovered_option_price": (
            recovered_surface["prices"]
        ),
        "price_error": (
            recovered_surface["prices"]
            - target["prices"]
        ),
    })

    comparison_frame.to_csv(
        OUTDIR / "recovered_vs_target_prices.csv",
        index=False,
    )

    summary = {
        "success": bool(result.success),
        "optimizer_message": str(result.message),
        "true_betas": TRUE_BETAS.tolist(),
        "start_betas": START_BETAS.tolist(),
        "recovered_betas": recovered_betas.tolist(),
        "parameter_error": parameter_error.tolist(),
        "absolute_parameter_error": (
            absolute_parameter_error.tolist()
        ),
        "truth_rmse": float(truth_rmse),
        "start_rmse": float(start_rmse),
        "final_price_rmse": float(final_price_rmse),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "optimization_seconds": float(
            optimization_seconds
        ),
        "mean_evaluation_seconds": float(
            history_frame["seconds"].mean()
        ),
        "N_paths": N_PATHS,
        "timestep_per_day": TIMESTEP_PER_DAY,
        "seed_root": SEED_ROOT,
        "dte": OPTION_DTE,
        "strike_count": len(STRIKES),
        "log_moneyness_min": float(
            LOG_MONEYNESS.min()
        ),
        "log_moneyness_max": float(
            LOG_MONEYNESS.max()
        ),
        "parameter_bounds": {
            key: list(value)
            for key, value
            in PARAMETER_BOUNDS.items()
        },
    }

    with open(
        OUTDIR / "calibration_summary.json",
        "w",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("========================================")
    print("CALIBRATION RESULT")
    print("========================================")

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("True vs recovered:")

    for name, truth, recovered, error in zip(
        ["beta0", "beta1", "beta2"],
        TRUE_BETAS,
        recovered_betas,
        parameter_error,
    ):
        print(
            f"{name:6s} "
            f"true={truth: .8f} "
            f"recovered={recovered: .8f} "
            f"error={error:+.8f}"
        )

    print()
    print(
        "Wrote:",
        OUTDIR / "synthetic_target_prices.csv",
    )

    print(
        "Wrote:",
        OUTDIR / "nelder_mead_evaluation_history.csv",
    )

    print(
        "Wrote:",
        OUTDIR / "recovered_vs_target_prices.csv",
    )

    print(
        "Wrote:",
        OUTDIR / "calibration_summary.json",
    )


if __name__ == "__main__":
    main()
