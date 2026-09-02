"""Calibrate the restricted five-parameter PDV model (beta0, beta1, beta2, theta1,
theta2) against each of the 20 historical fixed-grid option surfaces using
Nelder-Mead at 4,000 Monte Carlo paths per evaluation. Output feeds
validate_pdv_historical_parameters.py for independent numerical validation."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

CALIBRATOR = (
    REPO_ROOT
    / "scripts/calibration/calibrate_pdv_market_5param_fixed_grid.py"
)

BASE_OUTDIR = Path(
    "/users/4/trest017/urop_pdv/benchmarks/"
    "calibration/pdv_market_5param_fixed_grid"
)

SUMMARY_PATH = (
    BASE_OUTDIR
    / "historical_sequence_summary.csv"
)

DATES_BACKWARD = [
    "2021-06-01",
    "2021-05-28",
    "2021-05-27",
    "2021-05-26",
    "2021-05-25",
    "2021-05-24",
    "2021-05-21",
    "2021-05-20",
    "2021-05-19",
    "2021-05-18",
    "2021-05-17",
    "2021-05-14",
    "2021-05-13",
    "2021-05-12",
    "2021-05-11",
    "2021-05-10",
    "2021-05-07",
    "2021-05-06",
    "2021-05-05",
]

N_PATHS = 4000
SEED_ROOT = 2026080701
MAXITER = 600
MAXFEV = 1200


def load_summary(date):
    path = (
        BASE_OUTDIR
        / date
        / "calibration_summary.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing calibration summary: {path}"
        )

    with open(path) as f:
        return json.load(f)


def summary_row(summary):
    p = summary["recovered_params"]

    return {
        "trade_date": summary["trade_date"],
        "success": summary["success"],
        "beta0": p[0],
        "beta1": p[1],
        "beta2": p[2],
        "theta1": p[3],
        "theta2": p[4],
        "start_global_rmse":
            summary["start_global_rmse"],
        "final_global_rmse":
            summary["final_global_rmse"],
        "final_equal_tenor_rmse":
            summary["final_equal_tenor_rmse"],
        "iterations":
            summary["iterations"],
        "function_evaluations":
            summary["function_evaluations"],
        "optimization_seconds":
            summary["optimization_seconds"],
        "spx_1555_proxy":
            summary["spx_1555_proxy"],
    }


def save_sequence(rows):
    frame = pd.DataFrame(rows)

    if len(frame):
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"]
        )

        frame = frame.sort_values(
            "trade_date"
        )

        frame["trade_date"] = (
            frame["trade_date"]
            .dt.strftime("%Y-%m-%d")
        )

    frame.to_csv(
        SUMMARY_PATH,
        index=False,
    )


def main():
    BASE_OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # June 2 is our completed canary and starting point.
    june2 = load_summary(
        "2021-06-02"
    )

    current_params = (
        june2["recovered_params"]
    )

    rows = [
        summary_row(june2)
    ]

    save_sequence(rows)

    print("=" * 78)
    print("HISTORICAL 5-PARAMETER PDV SEQUENCE")
    print("=" * 78)
    print(
        "Starting from completed 2021-06-02 solution:"
    )
    print(current_params)
    print()

    total = len(DATES_BACKWARD)

    for i, date in enumerate(
        DATES_BACKWARD,
        start=1,
    ):
        summary_file = (
            BASE_OUTDIR
            / date
            / "calibration_summary.json"
        )

        print()
        print("=" * 78)
        print(
            f"DAY {i:02d}/{total}: {date}"
        )
        print("=" * 78)
        print(
            "Warm start:",
            current_params,
            flush=True,
        )

        # Resume successful dates. An existing unsuccessful
        # calibration is rerun instead of being propagated as
        # the next day's warm start.
        existing_summary = None

        if summary_file.exists():
            existing_summary = load_summary(
                date
            )

        if (
            existing_summary is not None
            and existing_summary.get("success", False)
        ):
            print(
                "Existing successful summary found; "
                "using saved result.",
                flush=True,
            )

            summary = existing_summary

        else:
            if existing_summary is not None:
                print(
                    "Existing calibration was unsuccessful; "
                    "rerunning this date.",
                    flush=True,
                )
            start_arg = ",".join(
                f"{x:.17g}"
                for x in current_params
            )

            cmd = [
                sys.executable,
                str(CALIBRATOR),
                "--trade-date",
                date,
                "--start-params",
                start_arg,
                "--n-paths",
                str(N_PATHS),
                "--seed-root",
                str(SEED_ROOT),
                "--maxiter",
                str(MAXITER),
                "--maxfev",
                str(MAXFEV),
            ]

            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
            )

            if completed.returncode != 0:
                raise RuntimeError(
                    f"{date}: calibrator failed "
                    f"with exit code "
                    f"{completed.returncode}"
                )

            summary = load_summary(
                date
            )

        rows.append(
            summary_row(summary)
        )

        # Only a successfully converged calibration is allowed
        # to become the next day's warm start. If this date did
        # not converge, preserve the previous successful vector.
        if summary["success"]:
            current_params = (
                summary["recovered_params"]
            )
        else:
            print(
                "Calibration did not converge; "
                "keeping previous successful warm start.",
                flush=True,
            )

        # Save after every date so a partial run
        # is never lost.
        save_sequence(rows)

        print(
            f"{date} complete | "
            f"success={summary['success']} | "
            f"RMSE="
            f"{summary['final_global_rmse']:.10f}",
            flush=True,
        )

        print(
            "Recovered:",
            current_params,
            flush=True,
        )

    print()
    print("=" * 78)
    print("ALL HISTORICAL CALIBRATIONS FINISHED")
    print("=" * 78)
    print("Wrote:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
