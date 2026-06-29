from pathlib import Path
from thetadata import ThetaClient

ENV_PATH = Path("/users/4/trest017/urop_pdv/.theta.env")

client = ThetaClient(
    dotenv_path=ENV_PATH,
    dataframe_type="pandas",
)

print("Authenticated.")
print("Email:", getattr(client, "email", None))
print("Stock subscription:", getattr(client, "stock_subscription", None))
print("Options subscription:", getattr(client, "options_subscription", None))
print("Index subscription:", getattr(client, "index_subscription", None))
print("MDDS host:", getattr(client, "mdds_host", None))
print("MDDS port:", getattr(client, "mdds_port", None))

symbols = client.option_list_symbols()
print("\nOption symbols type:", type(symbols))
print(symbols.head() if hasattr(symbols, "head") else symbols)
