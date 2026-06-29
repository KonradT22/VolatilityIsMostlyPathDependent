from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INFILE = Path("/users/4/trest017/urop_pdv/data/processed/thetadata/spx_wide_paired_otm_smile_input_2021-06-02_1555_exp_2021-07-16.csv")
OUTDIR = Path("/users/4/trest017/urop_pdv/data/processed/thetadata")
OUTDIR.mkdir(parents=True, exist_ok=True)

TRADE_DATE = "2021-06-02"
EXPIRATION = "2021-07-16"
DTE = 44
T = DTE / 365.0

df = pd.read_csv(INFILE)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Recover the forward and discount from the saved parity data.
# These are constant across the slice, but recompute from call-put parity for reproducibility.
x = df["strike"].to_numpy(dtype=float)
y = df["call_minus_put"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
discount = -slope
forward = intercept / discount

print("Loaded:", INFILE)
print("Rows:", len(df))
print("DTE:", DTE)
print("T:", T)
print("discount:", discount)
print("forward:", forward)

def bs_forward_price(option_type: str, F: float, K: float, T: float, sigma: float, D: float) -> float:
    if sigma <= 0 or T <= 0:
        if option_type == "CALL":
            return D * max(F - K, 0.0)
        return D * max(K - F, 0.0)

    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if option_type == "CALL":
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "PUT":
        return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    else:
        raise ValueError(f"Unknown option_type: {option_type}")

def implied_vol(option_type: str, price: float, F: float, K: float, T: float, D: float) -> float:
    intrinsic = bs_forward_price(option_type, F, K, T, 1e-12, D)

    # Basic no-arb sanity check. Allow tiny numerical tolerance.
    if price < intrinsic - 1e-8:
        return np.nan

    def objective(sigma: float) -> float:
        return bs_forward_price(option_type, F, K, T, sigma, D) - price

    low = 1e-8
    high = 5.0

    try:
        f_low = objective(low)
        f_high = objective(high)

        if f_low * f_high > 0:
            return np.nan

        return brentq(objective, low, high, maxiter=200, xtol=1e-12)
    except Exception:
        return np.nan

df["iv"] = [
    implied_vol(
        option_type=row["otm_right"],
        price=float(row["otm_mid"]),
        F=float(forward),
        K=float(row["strike"]),
        T=float(T),
        D=float(discount),
    )
    for _, row in df.iterrows()
]

df["forward"] = forward
df["discount"] = discount
df["dte"] = DTE
df["T"] = T
df["moneyness"] = df["strike"] / forward

# Basic quality filters for a clean first smile.
df["iv_valid"] = df["iv"].notna()
df["liquidity_filter"] = (
    (df["otm_mid"] > 0)
    & (df["otm_relative_spread"] <= 0.08)
    & df["iv_valid"]
)

out = OUTDIR / "spx_otm_iv_smile_2021-06-02_1555_exp_2021-07-16.csv"
df.to_csv(out, index=False)

clean = df[df["liquidity_filter"]].copy()

print("\nIV results:")
print("valid IV count:", int(df["iv_valid"].sum()))
print("clean IV count:", len(clean))
print("\nIV summary all valid:")
print(df.loc[df["iv_valid"], "iv"].describe())

print("\nIV summary clean:")
print(clean["iv"].describe())

print("\nClean smile preview:")
print(clean[[
    "strike",
    "moneyness",
    "log_moneyness",
    "otm_right",
    "otm_mid",
    "otm_relative_spread",
    "iv",
    "parity_residual",
]].sort_values("strike").head(30))

print("\nClean smile tail:")
print(clean[[
    "strike",
    "moneyness",
    "log_moneyness",
    "otm_right",
    "otm_mid",
    "otm_relative_spread",
    "iv",
    "parity_residual",
]].sort_values("strike").tail(30))

plot_path = OUTDIR / "spx_otm_iv_smile_2021-06-02_1555_exp_2021-07-16.png"

plt.figure()
plt.scatter(clean["log_moneyness"], clean["iv"])
plt.xlabel("log(K / F)")
plt.ylabel("Black-Scholes implied volatility")
plt.title("SPX OTM IV Smile: 2021-06-02 15:55, Exp 2021-07-16")
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path, dpi=150)

print("\nWrote:", out)
print("Wrote:", plot_path)
