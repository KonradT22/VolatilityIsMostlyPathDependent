from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.stats import norm


BASE = Path(
    "/users/4/trest017/urop_pdv/data/processed/thetadata"
)

MANIFEST = (
    BASE
    / "spxw_2021-05-05_2021-06-02_155500_historical_manifest.csv"
)

GRID_BOUNDS = {
    7:  (-0.040, 0.010),
    14: (-0.050, 0.020),
    21: (-0.050, 0.025),
    30: (-0.040, 0.035),
    45: (-0.040, 0.040),
    60: (-0.040, 0.040),
    90: (-0.035, 0.040),
}

POINTS_PER_TENOR = 11


def normalized_otm_price(log_moneyness, sigma, T):
    """
    Black normalized OTM price with F=1 and D=1.

    The resulting price is equivalent to market price / (discount * forward).
    """
    m = math.exp(log_moneyness)

    if sigma <= 0 or T <= 0:
        return float("nan")

    vol_sqrt_t = sigma * math.sqrt(T)

    d1 = (
        -log_moneyness
        + 0.5 * sigma * sigma * T
    ) / vol_sqrt_t

    d2 = d1 - vol_sqrt_t

    if m < 1.0:
        # Normalized put.
        return m * norm.cdf(-d2) - norm.cdf(-d1)

    # Normalized call.
    return norm.cdf(d1) - m * norm.cdf(d2)


def main():
    manifest = pd.read_csv(MANIFEST)

    rows = []
    grid_rows = []

    for target_dte, (low, high) in GRID_BOUNDS.items():
        grid = np.linspace(
            low,
            high,
            POINTS_PER_TENOR,
        )

        for j, logm in enumerate(grid):
            grid_rows.append({
                "target_dte": target_dte,
                "grid_index": j,
                "log_moneyness": logm,
                "moneyness": math.exp(logm),
            })

    grid_spec = pd.DataFrame(grid_rows)

    for trade_date in manifest["trade_date"]:
        path = (
            BASE
            / f"spxw_{trade_date}_155500_iv_surface_dataset.csv"
        )

        df = pd.read_csv(path)

        clean = df[
            df["liquidity_filter"] == True
        ].copy()

        for target_dte, (low, high) in GRID_BOUNDS.items():
            g = clean[
                clean["target_dte"] == target_dte
            ].copy()

            if g.empty:
                raise RuntimeError(
                    f"{trade_date}: no clean data for target "
                    f"{target_dte}"
                )

            actual_dtes = g["dte"].unique()

            if len(actual_dtes) != 1:
                raise RuntimeError(
                    f"{trade_date} target {target_dte}: "
                    f"multiple actual DTEs {actual_dtes}"
                )

            actual_dte = int(actual_dtes[0])
            T = actual_dte / 365.0

            g = (
                g.groupby(
                    "log_moneyness",
                    as_index=False,
                )
                .agg(iv=("iv", "mean"))
                .sort_values("log_moneyness")
            )

            market_low = float(
                g["log_moneyness"].min()
            )
            market_high = float(
                g["log_moneyness"].max()
            )

            if market_low > low or market_high < high:
                raise RuntimeError(
                    f"{trade_date} target {target_dte}: "
                    f"requested [{low:.6f}, {high:.6f}] "
                    f"outside market "
                    f"[{market_low:.6f}, {market_high:.6f}]"
                )

            grid = np.linspace(
                low,
                high,
                POINTS_PER_TENOR,
            )

            interp_iv = np.interp(
                grid,
                g["log_moneyness"].to_numpy(),
                g["iv"].to_numpy(),
            )

            for j, (logm, iv) in enumerate(
                zip(grid, interp_iv)
            ):
                m = math.exp(logm)

                rows.append({
                    "trade_date": trade_date,
                    "target_dte": target_dte,
                    "actual_dte": actual_dte,
                    "T": T,
                    "grid_index": j,
                    "log_moneyness": logm,
                    "moneyness": m,
                    "otm_right": (
                        "PUT" if m < 1.0 else "CALL"
                    ),
                    "interpolated_iv": iv,
                    "normalized_otm_price":
                        normalized_otm_price(
                            logm,
                            iv,
                            T,
                        ),
                })

    out = pd.DataFrame(rows)

    expected_per_day = (
        len(GRID_BOUNDS)
        * POINTS_PER_TENOR
    )

    counts = out.groupby("trade_date").size()

    if not (counts == expected_per_day).all():
        raise RuntimeError(
            "Not every date has the expected "
            f"{expected_per_day} points:\n{counts}"
        )

    if not np.isfinite(
        out["normalized_otm_price"]
    ).all():
        raise RuntimeError(
            "Non-finite normalized prices found"
        )

    grid_path = BASE / "pdv_fixed_option_grid_7x11.csv"

    long_path = (
        BASE
        / "spxw_2021-05-05_2021-06-02_155500_"
          "fixed_grid_targets.csv"
    )

    grid_spec.to_csv(
        grid_path,
        index=False,
    )

    out.to_csv(
        long_path,
        index=False,
    )

    print("=" * 78)
    print("FIXED HISTORICAL GRID")
    print("=" * 78)
    print("dates:", out["trade_date"].nunique())
    print("target tenors:", out["target_dte"].nunique())
    print("points per tenor:", POINTS_PER_TENOR)
    print("points per day:", expected_per_day)
    print("total rows:", len(out))

    print("\nACTUAL DTE RANGE")
    print(
        out.groupby("target_dte")["actual_dte"]
        .agg(["min", "median", "max"])
        .to_string()
    )

    print("\nGRID SPECIFICATION")
    print(
        grid_spec.groupby("target_dte")
        .agg(
            logm_min=("log_moneyness", "min"),
            logm_max=("log_moneyness", "max"),
            points=("grid_index", "count"),
        )
        .to_string()
    )

    print("\nPRICE RANGE BY TENOR")
    print(
        out.groupby("target_dte")[
            "normalized_otm_price"
        ]
        .agg(["min", "median", "max"])
        .to_string()
    )

    print("\nWrote:", grid_path)
    print("Wrote:", long_path)


if __name__ == "__main__":
    main()
