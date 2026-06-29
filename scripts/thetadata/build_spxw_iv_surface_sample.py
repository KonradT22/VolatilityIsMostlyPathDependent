from pathlib import Path
import math
import datetime as dt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thetadata import ThetaClient

ENV_PATH = Path("/users/4/trest017/urop_pdv/.theta.env")
RAW_DIR = Path("/users/4/trest017/urop_pdv/data/raw/thetadata")
OUT_DIR = Path("/users/4/trest017/urop_pdv/data/processed/thetadata")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

symbol = "SPXW"
trade_date = dt.date(2021, 6, 2)
time_of_day = "15:55:00"
target_dtes = [7, 14, 21, 30, 45, 60, 90]
strike_range = 50

client = ThetaClient(
    dotenv_path=ENV_PATH,
    dataframe_type="pandas",
)

def bs_forward_price(option_type, F, K, T, sigma, D):
    if sigma <= 0 or T <= 0:
        if option_type == "CALL":
            return D * max(F - K, 0.0)
        return D * max(K - F, 0.0)

    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if option_type == "CALL":
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def implied_vol(option_type, price, F, K, T, D):
    intrinsic = bs_forward_price(option_type, F, K, T, 1e-12, D)
    if price < intrinsic - 1e-8:
        return np.nan

    def objective(sigma):
        return bs_forward_price(option_type, F, K, T, sigma, D) - price

    try:
        low = 1e-8
        high = 5.0
        if objective(low) * objective(high) > 0:
            return np.nan
        return brentq(objective, low, high, maxiter=200, xtol=1e-12)
    except Exception:
        return np.nan

print("Listing expirations for", symbol)
exp = client.option_list_expirations(symbol)
exp["expiration"] = pd.to_datetime(exp["expiration"]).dt.date
exp = exp[exp["expiration"] > trade_date].copy()
exp["dte"] = exp["expiration"].apply(lambda x: (x - trade_date).days)

selected = []
seen = set()

for target in target_dtes:
    nearest = exp.iloc[(exp["dte"] - target).abs().argsort()].iloc[0]
    expiration = nearest["expiration"]
    if expiration not in seen:
        selected.append((expiration, int(nearest["dte"]), target))
        seen.add(expiration)

print("\nSelected expirations:")
for expiration, dte, target in selected:
    print(f"target={target:>3} actual_dte={dte:>3} expiration={expiration}")

all_rows = []
summaries = []

for expiration, dte, target in selected:
    print("\nPulling:", expiration, "DTE:", dte)

    try:
        quotes = client.option_at_time_quote(
            symbol=symbol,
            start_date=trade_date,
            end_date=trade_date,
            time_of_day=time_of_day,
            expiration=expiration,
            strike="*",
            right="both",
            strike_range=strike_range,
        )
    except Exception as e:
        print("FAILED pull:", expiration, repr(e))
        continue

    print("Raw rows:", len(quotes))

    if quotes.empty:
        print("No quotes returned:", expiration)
        continue

    raw_path = RAW_DIR / f"spxw_surface_raw_{trade_date}_{time_of_day.replace(':','')}_exp_{expiration}.csv"
    quotes.to_csv(raw_path, index=False)

    df = quotes.copy()
    df = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]
    df["relative_spread"] = df["spread"] / df["mid"]

    wide = df.pivot_table(
        index="strike",
        columns="right",
        values=["bid", "ask", "mid", "spread", "relative_spread"],
        aggfunc="first",
    )

    wide.columns = [f"{field}_{right.lower()}" for field, right in wide.columns]
    wide = wide.reset_index()
    wide = wide.dropna(subset=["mid_call", "mid_put"]).copy()

    if len(wide) < 10:
        print("Too few paired strikes:", expiration, len(wide))
        continue

    wide["call_minus_put"] = wide["mid_call"] - wide["mid_put"]

    x = wide["strike"].to_numpy(dtype=float)
    y = wide["call_minus_put"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    discount = -slope
    forward = intercept / discount
    T = dte / 365.0

    wide["symbol"] = symbol
    wide["trade_date"] = str(trade_date)
    wide["quote_time"] = time_of_day
    wide["expiration"] = str(expiration)
    wide["dte"] = dte
    wide["target_dte"] = target
    wide["discount"] = discount
    wide["forward"] = forward
    wide["T"] = T

    wide["parity_fitted_c_minus_p"] = intercept + slope * wide["strike"]
    wide["parity_residual"] = wide["call_minus_put"] - wide["parity_fitted_c_minus_p"]

    wide["moneyness"] = wide["strike"] / forward
    wide["log_moneyness"] = np.log(wide["moneyness"])
    wide["otm_right"] = np.where(wide["strike"] < forward, "PUT", "CALL")
    wide["otm_mid"] = np.where(wide["strike"] < forward, wide["mid_put"], wide["mid_call"])
    wide["otm_spread"] = np.where(wide["strike"] < forward, wide["spread_put"], wide["spread_call"])
    wide["otm_relative_spread"] = wide["otm_spread"] / wide["otm_mid"]

    wide["iv"] = [
        implied_vol(
            option_type=row["otm_right"],
            price=float(row["otm_mid"]),
            F=float(forward),
            K=float(row["strike"]),
            T=float(T),
            D=float(discount),
        )
        for _, row in wide.iterrows()
    ]

    wide["iv_valid"] = wide["iv"].notna()
    wide["liquidity_filter"] = (
        (wide["otm_mid"] > 0)
        & (wide["otm_relative_spread"] <= 0.10)
        & wide["iv_valid"]
    )

    clean_count = int(wide["liquidity_filter"].sum())

    summaries.append({
        "symbol": symbol,
        "expiration": str(expiration),
        "dte": dte,
        "target_dte": target,
        "raw_rows": len(quotes),
        "paired_strikes": len(wide),
        "clean_iv_count": clean_count,
        "forward": forward,
        "discount": discount,
        "parity_resid_std": wide["parity_residual"].std(),
        "iv_min": wide.loc[wide["iv_valid"], "iv"].min(),
        "iv_max": wide.loc[wide["iv_valid"], "iv"].max(),
        "iv_median": wide.loc[wide["iv_valid"], "iv"].median(),
    })

    all_rows.append(wide)

if not all_rows:
    raise RuntimeError("No usable SPXW expiration slices were created.")

surface = pd.concat(all_rows, ignore_index=True)
summary = pd.DataFrame(summaries)

surface_path = OUT_DIR / f"spxw_iv_surface_sample_{trade_date}_{time_of_day.replace(':','')}.csv"
summary_path = OUT_DIR / f"spxw_iv_surface_summary_{trade_date}_{time_of_day.replace(':','')}.csv"
plot_path = OUT_DIR / f"spxw_iv_surface_sample_{trade_date}_{time_of_day.replace(':','')}.png"

surface.to_csv(surface_path, index=False)
summary.to_csv(summary_path, index=False)

print("\nSummary:")
print(summary)

clean = surface[surface["liquidity_filter"]].copy()

plt.figure()
for dte, group in clean.groupby("dte"):
    group = group.sort_values("log_moneyness")
    plt.scatter(group["log_moneyness"], group["iv"], label=f"{int(dte)} DTE", s=16)

plt.xlabel("log(K / F)")
plt.ylabel("Black-Scholes implied volatility")
plt.title("SPXW OTM IV Surface Sample: 2021-06-02 15:55")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=150)

print("\nWrote:", surface_path)
print("Wrote:", summary_path)
print("Wrote:", plot_path)
