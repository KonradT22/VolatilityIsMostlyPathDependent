from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("/users/4/trest017/urop_pdv/data/processed/thetadata/spx_quote_wide_processed_2021-06-02_1555_exp_2021-07-16.csv")
OUTDIR = Path("/users/4/trest017/urop_pdv/data/processed/thetadata")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("Loaded:", RAW)
print("Shape:", df.shape)

# Keep clean quotes only.
df = df.copy()
df = df[(df["bid"] > 0) & (df["ask"] > 0)]
df["mid"] = (df["bid"] + df["ask"]) / 2.0
df["spread"] = df["ask"] - df["bid"]
df["relative_spread"] = df["spread"] / df["mid"]

# Pivot calls and puts by strike.
wide = df.pivot_table(
    index="strike",
    columns="right",
    values=["bid", "ask", "mid", "spread", "relative_spread"],
    aggfunc="first",
)

# Flatten columns.
wide.columns = [f"{field}_{right.lower()}" for field, right in wide.columns]
wide = wide.reset_index()

needed = ["mid_call", "mid_put"]
wide = wide.dropna(subset=needed).copy()

# Put-call parity:
# C - P = D(F - K) = D*F - D*K
wide["call_minus_put"] = wide["mid_call"] - wide["mid_put"]

x = wide["strike"].to_numpy(dtype=float)
y = wide["call_minus_put"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)

discount = -slope
forward = intercept / discount if discount != 0 else np.nan

print("\nPut-call parity regression:")
print("slope:", slope)
print("intercept:", intercept)
print("discount_factor_estimate:", discount)
print("forward_estimate:", forward)

wide["parity_fitted_c_minus_p"] = intercept + slope * wide["strike"]
wide["parity_residual"] = wide["call_minus_put"] - wide["parity_fitted_c_minus_p"]

# Choose OTM quote by forward.
wide["log_moneyness"] = np.log(wide["strike"] / forward)
wide["otm_right"] = np.where(wide["strike"] < forward, "PUT", "CALL")
wide["otm_mid"] = np.where(wide["strike"] < forward, wide["mid_put"], wide["mid_call"])
wide["otm_spread"] = np.where(wide["strike"] < forward, wide["spread_put"], wide["spread_call"])
wide["otm_relative_spread"] = wide["otm_spread"] / wide["otm_mid"]

print("\nPaired strike count:", len(wide))
print("Strike min/max:", wide["strike"].min(), wide["strike"].max())
print("Forward estimate:", forward)

print("\nParity residual summary:")
print(wide["parity_residual"].describe())

print("\nOTM smile input preview:")
preview_cols = [
    "strike",
    "log_moneyness",
    "otm_right",
    "otm_mid",
    "otm_spread",
    "otm_relative_spread",
    "mid_call",
    "mid_put",
    "call_minus_put",
    "parity_residual",
]
print(wide[preview_cols].sort_values("strike").head(20))
print(wide[preview_cols].sort_values("strike").tail(20))

out = OUTDIR / "spx_wide_paired_otm_smile_input_2021-06-02_1555_exp_2021-07-16.csv"
wide.to_csv(out, index=False)
print("\nWrote:", out)
