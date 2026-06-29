# IV Surface Dataset Schema

This document describes the standardized implied-volatility surface dataset produced by:

scripts/thetadata/build_iv_surface_dataset.py

## Default sample

- Symbol: SPXW
- Trade date: 2021-06-02
- Quote time: 15:55:00
- Target DTEs: 7, 14, 21, 30, 45, 60, 90
- Strike range: 50

## Outputs

The script writes:

- iv_surface_dataset.csv
- iv_surface_summary.csv
- iv_surface_failures.csv
- iv_surface_plot.png

## Key columns

symbol: option root, such as SPXW
trade_date: quote trade date
quote_time: requested quote time
expiration: option expiration date
dte: calendar days to expiration
target_dte: requested maturity bucket
T: year fraction, computed as dte / 365
forward: forward estimate inferred from put-call parity
discount: discount factor inferred from put-call parity
strike: option strike
moneyness: strike divided by forward
log_moneyness: log(K / F)
otm_right: selected OTM option side, PUT below forward and CALL above forward
otm_mid: selected OTM mid price
otm_spread: selected OTM bid-ask spread
otm_relative_spread: selected OTM spread divided by OTM mid
iv: Black-Scholes implied volatility from the selected OTM mid
iv_valid: whether IV inversion succeeded
liquidity_filter: quality flag for usable IV rows
parity_residual: put-call parity regression residual
mid_call: call mid price
mid_put: put mid price
spread_call: call bid-ask spread
spread_put: put bid-ask spread
relative_spread_call: call relative spread
relative_spread_put: put relative spread
raw_quote_file: raw ThetaData CSV used for the row

## Method

For each expiration, the script pulls paired call and put quotes across strikes. It computes mid prices and estimates the forward and discount factor from put-call parity:

C - P = D(F - K)

A regression of C - P against strike gives:

slope = -D
intercept = D * F

Therefore:

discount = -slope
forward = intercept / discount

The script then selects OTM options:

If K < F, use the put mid.
If K >= F, use the call mid.

Finally, the script computes Black-Scholes implied volatility from the selected OTM mid price.

## Failure handling

If an expiration cannot be pulled or does not have usable paired quotes, the script records the failure in the failures CSV instead of crashing.

For the initial SPXW sample, the 2021-08-02 expiration at 61 DTE returned no quote data for 2021-06-02 15:55.
