from pathlib import Path
import datetime as dt
import pandas as pd

from thetadata import ThetaClient

ENV_PATH = Path("/users/4/trest017/urop_pdv/.theta.env")
OUT_RAW = Path("/users/4/trest017/urop_pdv/data/raw/thetadata")
OUT_PROCESSED = Path("/users/4/trest017/urop_pdv/data/processed/thetadata")
OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT_PROCESSED.mkdir(parents=True, exist_ok=True)

client = ThetaClient(
    dotenv_path=ENV_PATH,
    dataframe_type="pandas",
)

symbol = "SPX"
trade_date = dt.date(2021, 6, 2)
expiration = dt.date(2021, 7, 16)
time_of_day = "15:55:00"
strike_range = 50

print("Pulling wider SPX quote slice...")
print("Symbol:", symbol)
print("Trade date:", trade_date)
print("Expiration:", expiration)
print("Time:", time_of_day)
print("Strike range:", strike_range)

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

print("Returned type:", type(quotes))
print("Shape:", quotes.shape)
print("Columns:", list(quotes.columns))

raw_path = OUT_RAW / "spx_quote_wide_raw_2021-06-02_1555_exp_2021-07-16.csv"
quotes.to_csv(raw_path, index=False)

df = quotes.copy()
df["mid"] = (df["bid"] + df["ask"]) / 2.0
df["spread"] = df["ask"] - df["bid"]
df["relative_spread"] = df["spread"] / df["mid"]

processed_path = OUT_PROCESSED / "spx_quote_wide_processed_2021-06-02_1555_exp_2021-07-16.csv"
df.to_csv(processed_path, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\nHead:")
print(df.head(20))

print("\nRights:")
print(df["right"].value_counts(dropna=False))

print("\nStrike range:")
print(df["strike"].min(), df["strike"].max())

print("\nBid/ask/mid/spread summary:")
print(df[["bid", "ask", "mid", "spread", "relative_spread"]].describe())

print("\nWrote raw:", raw_path)
print("Wrote processed:", processed_path)
