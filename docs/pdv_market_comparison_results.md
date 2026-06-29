# PDV versus SPXW Market IV Surface: First Empirical Results

## Context

This note summarizes the first model-versus-market comparison for the reproduced 4-factor Path-Dependent Volatility (PDV) baseline.

Market data source:

- ThetaData SPXW option quotes
- Trade date: 2021-06-02
- Quote time: 15:55:00
- Maturities: 7, 14, 21, 30, 44, and 90 DTE
- Smile construction: put-call parity forward estimate, OTM option selection, Black-Scholes implied volatility inversion

PDV baseline:

- Original reproduced 4F-PDV parameterization
- Baseline betas: [0.04, -0.13, 0.65]
- theta1 = 0.25
- theta2 = 0.5
- N = 10000 Monte Carlo paths
- CPU execution on MSI Agate

## Baseline multi-maturity result

The reproduced baseline captures the broad term-structure direction and approximate ATM volatility level, but it systematically underestimates upside/right-wing implied volatility.

Aggregate baseline errors across 7, 14, 21, 30, 44, and 90 DTE:

| Metric | Baseline value |
| --- | ---: |
| Mean abs ATM IV error | 0.0090 |
| Mean abs left 5% IV error | 0.0122 |
| Mean abs right 5% IV error | 0.0519 |
| Mean abs wing-gap error | 0.0403 |
| Mean abs skew-slope error | 0.2481 |
| Mean objective, no slope | 0.1652 |

The largest baseline issue is the right wing. The baseline right-wing errors are especially large at short and medium maturities:

| DTE | Baseline right-wing IV error |
| ---: | ---: |
| 7 | -0.0673 |
| 14 | -0.0669 |
| 21 | -0.0848 |
| 30 | -0.0427 |
| 44 | -0.0288 |
| 90 | -0.0206 |

Interpretation: the baseline PDV smile is too steep and too low on the upside-call side, especially for short-dated maturities.

## Stability check

A 30D right-wing stability check showed that the right-wing underestimation is not explained away by simple Monte Carlo noise.

Key 30D right-wing IV values:

| Case | Right 5% IV |
| --- | ---: |
| N=10000, seed 123456 | 0.0628 |
| N=10000, seed 777777 | 0.0628 |
| N=20000, seed 123456 | 0.0640 |
| Narrow grid [-0.04, 0.04] | 0.0745 |

The result is stable under a larger path count, although far-wing points show some additional numerical sensitivity. This suggests the right-wing miss is primarily a model/parameter issue, with some additional IV-inversion instability in the most extreme wing points.

## 30D parameter sensitivity

A one-parameter 30D sweep showed that the smile is sensitive to beta1 and beta2.

Making beta1 less negative improved the right-wing fit:

| Case | Objective score |
| --- | ---: |
| beta1 = -0.07, beta2 = 0.65 | 0.0726 |
| baseline beta1 = -0.13, beta2 = 0.65 | 0.1363 |

Increasing beta2 lifted the smile level and improved the right wing, but could overshoot ATM and left-wing IV.

## 2D beta1-beta2 sweep

A compact 2D sweep over beta1 and beta2 found the best 30D candidate:

| Parameter | Value |
| --- | ---: |
| beta0 | 0.04 |
| beta1 | -0.09 |
| beta2 | 0.75 |

30D market comparison for the best candidate:

| Metric | Market | PDV candidate | Error |
| --- | ---: | ---: | ---: |
| ATM IV | 0.1308 | 0.1447 | +0.0139 |
| Left 5% IV | 0.1909 | 0.1871 | -0.0038 |
| Right 5% IV | 0.1056 | 0.1018 | -0.0038 |
| Wing gap | 0.0853 | 0.0854 | +0.0000 |

This candidate nearly matches the 30D left-right wing structure, but overshoots the ATM level and flattens the ATM skew slope.

## Candidate multi-maturity evaluation

The 30D-tuned candidate was then tested across all six maturities.

Candidate parameter sets:

| Case | Betas |
| --- | --- |
| Baseline | [0.04, -0.13, 0.65] |
| Best 30D candidate | [0.04, -0.09, 0.75] |
| Slope-balanced candidate | [0.04, -0.10, 0.75] |

Aggregate multi-maturity ranking:

| Case | Mean objective, no slope | Mean abs ATM error | Mean abs right-wing error | Mean abs wing-gap error |
| --- | ---: | ---: | ---: | ---: |
| Best 30D candidate | 0.0782 | 0.0169 | 0.0190 | 0.0167 |
| Slope-balanced candidate | 0.0892 | 0.0159 | 0.0237 | 0.0240 |
| Baseline | 0.1652 | 0.0090 | 0.0519 | 0.0403 |

The best candidate improves the multi-maturity objective from 0.1652 to 0.0782, which is roughly a 53% improvement.

## Interpretation

The reproduced PDV baseline systematically underestimates upside/right-wing implied volatility. A local beta1-beta2 sensitivity sweep shows that this miss is parameter-sensitive: reducing the magnitude of beta1 and increasing beta2 substantially improves the right-wing and wing-gap fit across maturities.

However, the improvement comes with tradeoffs:

- ATM volatility is overshot, especially at short maturities.
- The ATM skew slope becomes too flat.
- The 7D right wing remains difficult to match.
- The best 30D candidate does not uniformly dominate baseline across every metric.

## Research conclusion

The first empirical result is not just that the reproduced PDV baseline is uncalibrated. More specifically, the baseline has a stable and systematic upside-wing deficiency. Parameter changes can partially correct this deficiency, but they introduce a level/skew tradeoff. This motivates a more formal calibration objective and eventually a surrogate model trained on calibrated or stress-tested PDV parameter regimes.
