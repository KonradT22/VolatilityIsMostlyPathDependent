# Guardrails

The inverse ANN surrogate is a fast approximation of an expensive numerical calibration. It is only useful if there is a reliable way to know when it should *not* be trusted. This document explains the guardrail architecture, why each piece exists, and what it actually found.

## Why guardrails, not just a good R²

A neural surrogate trained on bounded local perturbations around 16 historical anchor dates (see [`RESEARCH_OVERVIEW.md`](RESEARCH_OVERVIEW.md)) has no principled way to extrapolate to a market regime unlike anything it saw during training. Its output on such an input is not an error — the network will still produce five numbers — but those numbers carry no calibration guarantee. The guardrail layer's job is to catch this before a downstream consumer (a fallback calibration routine, in this repo) treats the neural output as trustworthy.

## 1. Numerical instability discovered during historical validation

Before any synthetic data or ANN existed, 4 of the 20 historical calibration dates failed independent numerical re-validation: recalibrating at higher path count (20,000 paths) under three additional Monte Carlo seeds revealed nonpositive or nonfinite instantaneous volatility that the lower-fidelity (4,000-path) calibration RMSE had not surfaced. This is why only 16 of 20 dates became anchors for synthetic data generation — the excluded dates were not discarded for convenience; they failed an explicit validity check (`scripts/calibration/validate_pdv_historical_parameters.py`, `scripts/calibration/build_pdv_robust_historical_prior.py`).

This is the first guardrail lesson of the project: even the numerical reference model can be silently unstable, and stability must be checked independently rather than assumed from a single calibration run.

## 2. Parameter-bound projection

Each PDV parameter has a documented valid range (e.g., `beta0 ∈ [0.02, 0.06]`, `theta1, theta2 ∈ [0, 1]`). Raw ANN output is a real-valued regression and is not constrained to these ranges by construction. Before use, predictions are clipped to the documented bounds (`scripts/surrogate/evaluate_pdv_inverse_mlp_test.py`). On the 959-scenario test set, this projection was needed for a small number of predictions; violation counts and magnitudes are logged per parameter.

## 3. PDV numerical-validity check

Bound projection alone does not guarantee the resulting parameter set produces a numerically valid PDV path (finite, positive volatility throughout the simulation). After projection, the candidate is checked against the same validity conditions used during historical validation and synthetic dataset generation. 4 of 959 same-seed test predictions remained invalid after projection and were excluded from the repricing evaluation rather than silently repriced with a broken model.

## 4. Mahalanobis / Ledoit-Wolf OOD detector

The primary OOD detector operates in the full **81-dimensional input space** — the same 77 option prices plus 4 state variables the ANN consumes (`scripts/surrogate/analyze_pdv_mahalanobis_guardrail.py`):

1. Fit a mean/std standardizer and a **Ledoit-Wolf shrinkage covariance estimate** on standardized in-domain training rows. Ledoit-Wolf shrinkage is used here purely to stabilize the covariance estimate that feeds the Mahalanobis distance — it does not train or improve the ANN itself; it belongs entirely to this separate detector.
2. Score any row by its **squared Mahalanobis distance** to that fitted distribution.
3. Fix a detection threshold using only a held-out **in-domain calibration split** (20% of training anchors, stratified by anchor date) — never using the chronological or OOD data the detector is later evaluated against.
4. Evaluate the detector's separation between in-domain calibration rows and **chronological** validation/test rows (later historical dates: 2021-05-27, 2021-05-28 for validation, 2021-06-01, 2021-06-02 for test) that were held out entirely from training.

Results:

| Quantity | Value |
| --- | ---: |
| ID p99 threshold (squared Mahalanobis distance) | ≈ 50.85 |
| ID calibration false-positive rate at p99 | ≈ 1.01% |
| Chronological validation + test flag rate at p99 | 100% |
| AUROC, ID calibration vs. chronological OOD | ≈ 0.999999 |

In plain terms: a detector tuned only on in-domain data, using a threshold that flags about 1% of genuinely in-domain rows, flags **every single tested observation from four later, unseen trading dates.** This is the diagnostic evidence that chronological extrapolation is a real failure mode (see [Limitations](../README.md#what-failed--limitations)) and that a simple, cheap statistical check can reliably catch it before the ANN's parameter estimate is trusted.

A second, state-only 4-feature variant of the same detector (using only `R1_fast`, `R1_slow`, `R2_fast`, `R2_slow`) is computed alongside the full 81-D detector, to separate how much of the OOD signal comes from the option-surface prices versus the path-dependent state alone.

## Leave-one-anchor-out diagnostic

A complementary check (`scripts/surrogate/analyze_pdv_state_guardrail_leave_one_anchor_out.py`) asks: if each of the 12 chronological-training-era anchor states had instead been held out and treated as if it were "unseen," how far outside the remaining distribution would it have scored? This gives a training-era reference distribution of "in-family" distances. Later chronological states are then compared against the maximum and median of that in-family distribution, rather than only against a fixed percentile threshold — a robustness check on the OOD claim above.

## Fallback philosophy

Any one of the following routes a request away from the neural estimate and toward the original numerical (Nelder-Mead + Monte Carlo) calibration:

- The 81-D Mahalanobis OOD detector flags the input.
- The projected parameters fail the PDV numerical-validity check.

The system is designed so that the fast path is only ever used when both statistical and numerical checks pass — the neural surrogate never overrides a check it fails.

## Current limitations of the guardrail layer

- The full guarded pipeline (OOD check → ANN → projection → validity check → fallback) has not been benchmarked end-to-end for latency. Only the isolated ANN forward pass has a measured latency (0.109 ms median); the fallback and validity-check paths may themselves involve reference-model Monte Carlo work.
- The OOD detector's chronological test is a diagnostic on four specific later dates, not a guarantee of detection performance across all possible future market regimes.
- The fallback path (numerical calibration) is the original, expensive Nelder-Mead + Monte Carlo procedure — invoking it recovers correctness at the cost of the latency advantage the surrogate exists to provide.
