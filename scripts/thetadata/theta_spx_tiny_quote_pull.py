from pathlib import Path
import datetime as dt
import pandas as pd

from thetadata import ThetaClient

ENV_PATH = Path("/users/4/trest017/urop_pdv/.theta.env")
OUTDIR = Path("/users/4/trest017/urop_pdv/data/raw/thetadata")
OUTDIR.mkdir(parents=True, exist_ok=True)

client = ThetaClient(
    dotenv_path=ENV_PATH,
    dataframe_type="pandas",
)

symbol = "SPX"
trade_date = dt.date(2021, 6, 2)
target_min_dte = 30
time_of_day = "15:55:00"

print("Client authenticated as:", getattr(client, "email", None))
print("Symbol:", symbol)
print("Trade date:", trade_date)
print("Target time:", time_of_day)

# 1. Pick a monthly SPX expiration after the trade date.
exp = client.option_list_expirations(symbol)
exp["expiration"] = pd.to_datetime(exp["expiration"]).dt.date

candidate_exp = exp[exp["expiration"] >= trade_date + dt.timedelta(days=target_min_dte)].copy()
candidate_exp["dte"] = candidate_exp["expiration"].apply(lambda x: (x - trade_date).days)
candidate_exp = candidate_exp.sort_values("expiration")

print("\nCandidate expirations:")
print(candidate_exp.head(10))

expiration = candidate_exp.iloc[0]["expiration"]
print("\nSelected expiration:", expiration)
print("Selected DTE:", (expiration - trade_date).days)

# Save metadata.
candidate_exp.head(20).to_csv(
    OUTDIR / "spx_candidate_expirations_2021-06-02.csv",
    index=False,
)

# 2. Pull a tiny quote slice.
print("\nPulling option_at_time_quote...")

quotes = client.option_at_time_quote(
    symbol=symbol,
    start_date=trade_date,
    end_date=trade_date,
    time_of_day=time_of_day,
    expiration=expiration,
    strike="*",
    right="both",
    strike_range=10,
)

print("\nReturned type:", type(quotes))
print("Shape:", getattr(quotes, "shape", None))

if hasattr(quotes, "columns"):
    print("Columns:")
    for col in quotes.columns:
        print(" ", col)

    print("\nHead:")
    print(quotes.head(20))

    raw_path = OUTDIR / "spx_quote_sample_raw_2021-06-02_1555.csv"
    quotes.to_csv(raw_path, index=False)
    print("\nWrote:", raw_path)

    schema_path = OUTDIR / "spx_quote_sample_schema_2021-06-02_1555.txt"
    with open(schema_path, "w") as f:
        f.write(f"symbol={symbol}\n")
        f.write(f"trade_date={trade_date}\n")
        f.write(f"time_of_day={time_of_day}\n")
        f.write(f"expiration={expiration}\n")
        f.write(f"rows={len(quotes)}\n")
        f.write("columns=\n")
        for col in quotes.columns:
            f.write(f"  {col}\n")
    print("Wrote:", schema_path)
else:
    print(quotes)
