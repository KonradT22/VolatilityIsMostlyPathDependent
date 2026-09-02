"""Build a standardized implied-volatility surface dataset from ThetaData SPX/SPXW
option quotes: pulls paired call/put quotes per expiration, infers forward and
discount via put-call parity, selects OTM options, and inverts Black-Scholes IV.
Schema: docs/iv_surface_dataset_schema.md."""

from pathlib import Path
import math
import datetime as dt
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thetadata import ThetaClient


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
    if option_type == "PUT":
        return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    raise ValueError(f"Unknown option type: {option_type}")


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


def parse_date(value):
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def parse_target_dtes(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def select_expirations(
    client,
    symbol,
    trade_date,
    quote_time,
    target_dtes,
    strike_range,
    max_candidates=12,
):
    exp = client.option_list_expirations(symbol)
    exp["expiration"] = pd.to_datetime(exp["expiration"]).dt.date
    exp = exp[exp["expiration"] > trade_date].copy()
    exp["dte"] = exp["expiration"].apply(lambda x: (x - trade_date).days)

    selected = []
    seen = set()

    for target in target_dtes:
        candidates = exp.copy()
        candidates["distance"] = (candidates["dte"] - target).abs()
        candidates = candidates.sort_values(
            ["distance", "dte"]
        ).head(max_candidates)

        chosen = None

        for row in candidates.itertuples(index=False):
            expiration = row.expiration

            if expiration in seen:
                continue

            try:
                quotes = client.option_at_time_quote(
                    symbol=symbol,
                    start_date=trade_date,
                    end_date=trade_date,
                    time_of_day=quote_time,
                    expiration=expiration,
                    strike="*",
                    right="both",
                    strike_range=strike_range,
                )

                if quotes is None or quotes.empty:
                    continue

                chosen = (
                    expiration,
                    int(row.dte),
                    target,
                )

                print(
                    f"target={target:>3} "
                    f"selected actual_dte={int(row.dte):>3} "
                    f"expiration={expiration} "
                    f"probe_rows={len(quotes)}"
                )
                break

            except Exception:
                continue

        if chosen is None:
            print(
                f"WARNING: no usable expiration found "
                f"for target_dte={target}"
            )
            continue

        selected.append(chosen)
        seen.add(chosen[0])

    return selected

def build_one_expiration_slice(
    client,
    symbol,
    trade_date,
    quote_time,
    expiration,
    dte,
    target_dte,
    strike_range,
    raw_dir,
):
    print(f"\nPulling {symbol} expiration={expiration} dte={dte} target_dte={target_dte}")

    quotes = client.option_at_time_quote(
        symbol=symbol,
        start_date=trade_date,
        end_date=trade_date,
        time_of_day=quote_time,
        expiration=expiration,
        strike="*",
        right="both",
        strike_range=strike_range,
    )

    print("Raw rows:", len(quotes))

    if quotes.empty:
        raise RuntimeError("No quotes returned")

    raw_path = raw_dir / f"{symbol.lower()}_surface_raw_{trade_date}_{quote_time.replace(':','')}_exp_{expiration}.csv"
    quotes.to_csv(raw_path, index=False)

    df = quotes.copy()
    df = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()

    if df.empty:
        raise RuntimeError("No positive bid/ask quotes after filtering")

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
        raise RuntimeError(f"Too few paired strikes: {len(wide)}")

    wide["call_minus_put"] = wide["mid_call"] - wide["mid_put"]

    x = wide["strike"].to_numpy(dtype=float)
    y = wide["call_minus_put"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    discount = -slope
    forward = intercept / discount
    T = dte / 365.0

    wide["symbol"] = symbol
    wide["trade_date"] = str(trade_date)
    wide["quote_time"] = quote_time
    wide["expiration"] = str(expiration)
    wide["dte"] = dte
    wide["target_dte"] = target_dte
    wide["T"] = T
    wide["forward"] = forward
    wide["discount"] = discount

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

    return wide, raw_path


def make_summary(surface):
    rows = []

    for (symbol, expiration, dte), group in surface.groupby(["symbol", "expiration", "dte"]):
        valid = group[group["iv_valid"]]
        clean = group[group["liquidity_filter"]]

        rows.append({
            "symbol": symbol,
            "expiration": expiration,
            "dte": int(dte),
            "target_dte": int(group["target_dte"].iloc[0]),
            "paired_strikes": len(group),
            "clean_iv_count": len(clean),
            "forward": float(group["forward"].iloc[0]),
            "discount": float(group["discount"].iloc[0]),
            "parity_resid_std": float(group["parity_residual"].std()),
            "iv_min": float(valid["iv"].min()) if len(valid) else np.nan,
            "iv_max": float(valid["iv"].max()) if len(valid) else np.nan,
            "iv_median": float(valid["iv"].median()) if len(valid) else np.nan,
            "raw_quote_file": str(group["raw_quote_file"].iloc[0]),
        })

    return pd.DataFrame(rows).sort_values("dte")


def make_plot(surface, plot_path, title):
    clean = surface[surface["liquidity_filter"]].copy()

    plt.figure()

    for dte, group in clean.groupby("dte"):
        group = group.sort_values("log_moneyness")
        plt.scatter(group["log_moneyness"], group["iv"], label=f"{int(dte)} DTE", s=16)

    plt.xlabel("log(K / F)")
    plt.ylabel("Black-Scholes implied volatility")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)


def main():
    parser = argparse.ArgumentParser(
        description="Build a standardized ThetaData SPX/SPXW implied volatility surface dataset."
    )

    parser.add_argument("--symbol", default="SPXW")
    parser.add_argument("--trade-date", default="2021-06-02")
    parser.add_argument("--quote-time", default="15:55:00")
    parser.add_argument("--target-dtes", default="7,14,21,30,45,60,90")
    parser.add_argument("--strike-range", type=int, default=50)
    parser.add_argument("--dotenv-path", default="/users/4/trest017/urop_pdv/.theta.env")
    parser.add_argument("--raw-dir", default="/users/4/trest017/urop_pdv/data/raw/thetadata")
    parser.add_argument("--out-dir", default="/users/4/trest017/urop_pdv/data/processed/thetadata")

    args = parser.parse_args()

    symbol = args.symbol
    trade_date = parse_date(args.trade_date)
    quote_time = args.quote_time
    target_dtes = parse_target_dtes(args.target_dtes)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = ThetaClient(
        dotenv_path=Path(args.dotenv_path),
        dataframe_type="pandas",
    )

    selected = select_expirations(
        client=client,
        symbol=symbol,
        trade_date=trade_date,
        quote_time=quote_time,
        target_dtes=target_dtes,
        strike_range=args.strike_range,
    )

    print("Selected expirations:")
    for expiration, dte, target in selected:
        print(f"target={target:>3} actual_dte={dte:>3} expiration={expiration}")

    all_rows = []
    failures = []

    for expiration, dte, target_dte in selected:
        try:
            slice_df, raw_path = build_one_expiration_slice(
                client=client,
                symbol=symbol,
                trade_date=trade_date,
                quote_time=quote_time,
                expiration=expiration,
                dte=dte,
                target_dte=target_dte,
                strike_range=args.strike_range,
                raw_dir=raw_dir,
            )
            slice_df["raw_quote_file"] = str(raw_path)
            all_rows.append(slice_df)
        except Exception as e:
            print(f"FAILED {symbol} expiration={expiration} dte={dte}: {repr(e)}")
            failures.append({
                "symbol": symbol,
                "expiration": str(expiration),
                "dte": dte,
                "target_dte": target_dte,
                "error": repr(e),
            })

    if not all_rows:
        raise RuntimeError("No usable expiration slices were created.")

    surface = pd.concat(all_rows, ignore_index=True)

    cols = [
        "symbol",
        "trade_date",
        "quote_time",
        "expiration",
        "dte",
        "target_dte",
        "T",
        "forward",
        "discount",
        "strike",
        "moneyness",
        "log_moneyness",
        "otm_right",
        "otm_mid",
        "otm_spread",
        "otm_relative_spread",
        "iv",
        "iv_valid",
        "liquidity_filter",
        "parity_residual",
        "mid_call",
        "mid_put",
        "spread_call",
        "spread_put",
        "relative_spread_call",
        "relative_spread_put",
        "raw_quote_file",
    ]

    existing_cols = [c for c in cols if c in surface.columns]
    surface = surface[existing_cols].sort_values(["dte", "strike"]).copy()

    summary = make_summary(surface)
    failure_df = pd.DataFrame(failures)

    tag = f"{symbol.lower()}_{trade_date}_{quote_time.replace(':','')}"
    surface_path = out_dir / f"{tag}_iv_surface_dataset.csv"
    summary_path = out_dir / f"{tag}_iv_surface_summary.csv"
    failure_path = out_dir / f"{tag}_iv_surface_failures.csv"
    plot_path = out_dir / f"{tag}_iv_surface_plot.png"

    surface.to_csv(surface_path, index=False)
    summary.to_csv(summary_path, index=False)
    failure_df.to_csv(failure_path, index=False)

    make_plot(
        surface=surface,
        plot_path=plot_path,
        title=f"{symbol} OTM IV Surface Dataset: {trade_date} {quote_time}",
    )

    print("\nSummary:")
    print(summary)

    if len(failure_df):
        print("\nFailures:")
        print(failure_df)

    print("\nWrote:", surface_path)
    print("Wrote:", summary_path)
    print("Wrote:", failure_path)
    print("Wrote:", plot_path)


if __name__ == "__main__":
    main()
