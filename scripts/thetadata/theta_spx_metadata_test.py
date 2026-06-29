from pathlib import Path
import pandas as pd

from thetadata import ThetaClient

ENV_PATH = Path("/users/4/trest017/urop_pdv/.theta.env")
OUTDIR = Path("/users/4/trest017/urop_pdv/data/raw/thetadata")
OUTDIR.mkdir(parents=True, exist_ok=True)

client = ThetaClient(
    dotenv_path=ENV_PATH,
    dataframe_type="pandas",
)

# 1. Search available option symbols for SPX-like roots.
symbols = client.option_list_symbols()
symbols.to_csv(OUTDIR / "option_symbols_sample.csv", index=False)

print("option_list_symbols shape:", symbols.shape)
print("option_list_symbols columns:", list(symbols.columns))

if "symbol" in symbols.columns:
    spx_like = symbols[symbols["symbol"].astype(str).str.contains("SPX", case=False, na=False)]
    print("\nSPX-like option symbols:")
    print(spx_like.head(50))
    spx_like.to_csv(OUTDIR / "spx_like_option_symbols.csv", index=False)
else:
    print("\nNo 'symbol' column found. Head:")
    print(symbols.head())

# 2. Try SPX expirations directly.
symbol = "SPX"
print(f"\nListing expirations for {symbol}...")

exp = client.option_list_expirations(symbol)
exp.to_csv(OUTDIR / "spx_expirations_sample.csv", index=False)

print("expirations shape:", exp.shape)
print("expirations columns:", list(exp.columns))
print(exp.head(30))

print("\nWrote:")
print(" ", OUTDIR / "option_symbols_sample.csv")
print(" ", OUTDIR / "spx_like_option_symbols.csv")
print(" ", OUTDIR / "spx_expirations_sample.csv")
