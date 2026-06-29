# ThetaData SPX Quote Sample Schema

Date pulled: 2026-06-29
Vendor: ThetaData
Environment: MSI Agate
Python environment: theta-pull

## Sample request

Root: SPX
Trade date: 2021-06-02
Requested time: 15:55:00
Returned quote times: latest quotes immediately before 15:55:00
Expiration: 2021-07-16
DTE: 44
Right: both
Strike range: 10

## Raw output

Raw file:
/users/4/trest017/urop_pdv/data/raw/thetadata/spx_quote_sample_raw_2021-06-02_1555.csv

Processed file:
/users/4/trest017/urop_pdv/data/processed/thetadata/spx_quote_sample_processed_2021-06-02_1555.csv

Rows: 24
Columns: 13

## Columns

symbol
expiration
strike
right
timestamp
bid_size
bid_exchange
bid
bid_condition
ask_size
ask_exchange
ask
ask_condition

## Observed notes

Rights:
PUT 12
CALL 12

Timestamp example:
2021-06-02 15:54:57.291000-04:00

Strike range:
4160.0 to 4250.0

Derived fields:
mid = (bid + ask) / 2
spread = ask - bid
relative_spread = spread / mid

## Research implications

This confirms that ThetaData SPX option quote ingestion works. The quote endpoint returns bid/ask data but not implied volatility or Greeks. Raw files should remain immutable under data/raw/thetadata, while processed files should be written separately under data/processed/thetadata.
