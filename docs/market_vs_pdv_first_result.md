# First PDV Baseline vs Market IV Result

This note summarizes the first comparison between the reproduced 4F-PDV Python baseline and a real SPXW market implied-volatility smile from ThetaData.

## Market data sample

The market smile was built from ThetaData SPXW option quotes using the standardized IV surface dataset builder.

Sample:

- Symbol: SPXW
- Trade date: 2021-06-02
- Quote time: 15:55:00
- Matched expiration: 2021-07-02
- Matched DTE: 30

The market IV surface was constructed by:

1. Pulling paired call and put quotes.
2. Computing bid-ask mid prices.
3. Estimating the forward and discount factor from put-call parity.
4. Selecting OTM options.
5. Computing Black-Scholes implied volatility from OTM mid prices.

## PDV baseline

The PDV baseline comes from the reproduced Python notebook benchmark.

The baseline SPX option benchmark has:

- Estimated maturity: 30.42 DTE
- Strike grid: 0.90 to 1.05 relative moneyness
- IV output from the reproduced PDV pricing notebook

## Matched comparison

The reproduced PDV baseline was matched to the nearest available market maturity:

- PDV estimated DTE: 30.42
- Market DTE: 30
- Market expiration: 2021-07-02

## Error metrics

On the overlapping log-moneyness grid:

- RMSE: 0.0152
- MAE: 0.0134
- Bias, PDV minus market: -0.0134

Interpreted in volatility points, the baseline is roughly 1.3 to 1.5 vol points below the market smile on average.

## Shape comparison

The existing PDV baseline captures the broad negative equity skew shape, but it is systematically below the market smile across most of the overlapping grid.

Specific observations:

- Near ATM, the PDV baseline is close but slightly low.
- The left wing is reasonably close.
- The right wing is too low: the market smile flattens around 10 to 11 percent IV, while the PDV baseline continues falling.
- The PDV left-right wing gap is therefore larger than the market wing gap.

## Initial research interpretation

The reproduced 4F-PDV baseline generates a plausible 30 DTE SPX-style skew, but it is not yet calibrated to the real SPXW market surface. It roughly captures the ATM level and local skew direction, but it underestimates the upside/right-wing implied volatility and produces an overly steep left-right wing asymmetry.

This result motivates the next phase: adapting the PDV reference lane so it can produce implied-volatility smiles on the same maturities and moneyness grid as the standardized ThetaData market surface.
