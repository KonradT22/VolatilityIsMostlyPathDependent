from pathlib import Path
import pandas as pd

path = Path("/users/4/trest017/urop_pdv/data/raw/thetadata/spx_quote_sample_raw_2021-06-02_1555.csv")

df = pd.read_csv(path)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("Path:", path)
print("Shape:", df.shape)

print("\nColumns:")
for c in df.columns:
    print(" ", c)

print("\nDtypes:")
print(df.dtypes)

print("\nHead full:")
print(df.head(30))

print("\nUnique rights:")
print(df["right"].value_counts(dropna=False))

print("\nTimestamp values:")
print(df["timestamp"].value_counts(dropna=False).head(20))

print("\nStrike range:")
print(df["strike"].min(), df["strike"].max())

print("\nBid/ask summary:")
print(df[["bid", "ask"]].describe())

df["mid"] = (df["bid"] + df["ask"]) / 2.0
df["spread"] = df["ask"] - df["bid"]
df["relative_spread"] = df["spread"] / df["mid"]

print("\nProcessed head:")
print(df[["symbol", "expiration", "strike", "right", "timestamp", "bid", "ask", "mid", "spread", "relative_spread"]].head(30))

out = Path("/users/4/trest017/urop_pdv/data/processed/thetadata/spx_quote_sample_processed_2021-06-02_1555.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print("\nWrote:", out)
