# Research Overview

This document walks through the experiment sequence behind the inverse PDV surrogate, in the order it was actually developed, and points to the script(s) responsible for each stage.

## 0. Upstream starting point

`calibration/torch_montecarlo.py` and `calibration/torch_utils.py` implement the 4-factor Markovian representation of the Guyon–Lekeufack Path-Dependent Volatility model: instantaneous volatility as a function of exponentially-weighted past returns (`R1`) and squared past returns (`R2`), each split into a fast and slow component. `empirical_study/` and both root notebooks are the original authors' historical-fitting and pricing-demonstration code. None of this was modified for the UROP work beyond what upstream already contained.

## 1. Baseline reproduction and market comparison

Before any calibration to real data, the reproduced baseline (default parameters `betas=[0.04, -0.13, 0.65]`, `theta1=0.25`, `theta2=0.5`) was compared against a real SPXW implied-volatility smile pulled from ThetaData (`scripts/thetadata/build_iv_surface_dataset.py`, `scripts/compare/plot_pdv_baseline_vs_market_smile.py`). This surfaced a systematic issue: the baseline underestimated upside/right-wing implied volatility, especially at short maturities. A local `beta1`/`beta2` sensitivity sweep (`scripts/pdv/sweep_pdv_beta1_beta2_grid_30d.py`) found that reducing the magnitude of `beta1` and increasing `beta2` improved the right-wing fit substantially (mean multi-maturity objective improved from 0.1652 to 0.0782), at the cost of overshooting ATM level and flattening ATM skew slope. Full narrative and numbers: [`market_vs_pdv_first_result.md`](market_vs_pdv_first_result.md), [`pdv_market_comparison_results.md`](pdv_market_comparison_results.md).

This exploratory phase (`scripts/pdv/`, `scripts/compare/`) motivated moving from hand-tuned parameter sweeps to a proper multi-parameter calibration objective — which is where the five-parameter Nelder-Mead calibration below starts.

## 2. Fixed option grid and historical data

`scripts/thetadata/build_historical_fixed_grid.py` and `scripts/thetadata/build_iv_surface_dataset.py` construct a fixed 77-coordinate option grid (7 maturities × 11 log-moneyness points each) from ThetaData SPXW/SPX quotes, for 20 trading dates spanning May 5 – June 2, 2021, at ~3:55pm ET snapshots. Schema documented in [`thetadata_spx_quote_schema.md`](thetadata_spx_quote_schema.md) and [`iv_surface_dataset_schema.md`](iv_surface_dataset_schema.md).

## 3. Historical five-parameter calibration and validation

`scripts/calibration/run_pdv_market_5param_history.py` calibrates `beta0, beta1, beta2, theta1, theta2` against each of the 20 historical grids using `scipy.optimize.minimize(method="Nelder-Mead")` at 4,000 Monte Carlo paths (the `pdv-nelder-mead` branch this work lives on is named for this step).

`scripts/calibration/validate_pdv_historical_parameters.py` then independently re-evaluates each calibrated parameter set at 20,000 paths under three additional, previously unused Monte Carlo seeds. 16 of 20 dates were stable (finite, positive volatility) across all three seeds; 4 failed and were excluded (see [`GUARDRAILS.md`](GUARDRAILS.md) for why this matters).

`scripts/calibration/build_pdv_robust_historical_prior.py` builds the empirical parameter prior (mean, std, Ledoit-Wolf-shrunk covariance) from the 16 robust dates, along with each date's path-dependent state (`R1_fast`, `R1_slow`, `R2_fast`, `R2_slow`) preserved alongside the parameters.

## 4. Synthetic anchor-jitter dataset

`scripts/calibration/build_pdv_anchor_jitter_prior.py` draws 10,000 candidates by uniformly sampling one of the 16 robust anchors, then perturbing its five parameters with independent Gaussian jitter in bounded logit space (so perturbed values stay within documented parameter bounds by construction). Each candidate keeps the state vector of the anchor it was drawn from.

`scripts/calibration/generate_pdv_inverse_dataset.py` prices every candidate's full 77-coordinate surface at 10,000 Monte Carlo paths, run as 100 SLURM array jobs of 100 candidates each (`generate_pdv_inverse_dataset.slurm`). Candidates producing nonfinite/nonpositive volatility or option prices are rejected rather than silently kept; 9,667 of 10,000 candidates survived. `scripts/surrogate/merge_pdv_surface_dataset.py` combines the per-chunk outputs into the final dataset.

## 5. Train/validation/test splits

Two split strategies are prepared from the same 9,667-row dataset:

- **Interpolation split** (`scripts/calibration/prepare_pdv_inverse_interpolation_splits.py`): stratified by anchor date, used to train and evaluate the ANN's ability to interpolate within the space the 16 anchors define. This is the split behind the headline 959-scenario test results.
- **Chronological split** (`scripts/calibration/prepare_pdv_inverse_splits.py`): 12 earliest anchor dates for training, 2 middle dates (2021-05-27, 2021-05-28) for validation, 2 latest dates (2021-06-01, 2021-06-02) for test — used specifically to test generalization to *later, unseen* market states and to fit/evaluate the OOD guardrail.

## 6. Inverse ANN training and evaluation

`scripts/surrogate/train_pdv_inverse_mlp.py` trains the `81 → 256 → 256 → 128 → 5` GELU network with AdamW, early stopping on standardized validation MSE. `scripts/surrogate/evaluate_pdv_inverse_mlp_test.py` runs the frozen model against the untouched 959-row interpolation test set, applies parameter-bound projection, and records per-parameter R², bound-violation counts, and CPU inference timing.

## 7. Repricing and fresh-seed robustness

`scripts/surrogate/reprice_pdv_inverse_test.py` reprices every held-out (bound-projected) prediction through the full PDV Monte Carlo model on the original per-scenario seed, and compares against the true 77-price surface. `scripts/surrogate/reprice_pdv_inverse_fresh_seeds.py` repeats this for a balanced subset (4 scenarios × 16 anchors) under three entirely new Monte Carlo seeds, to check that repricing accuracy isn't an artifact of common-random-number seed reuse.

## 8. OOD / guardrail analysis

`scripts/surrogate/analyze_pdv_mahalanobis_guardrail.py` fits the 81-D (and a state-only 4-D) Mahalanobis/Ledoit-Wolf OOD detector on the chronological training split and evaluates it against the chronological validation/test states. `scripts/surrogate/analyze_pdv_state_guardrail_leave_one_anchor_out.py` runs a complementary leave-one-anchor-out diagnostic purely in state space. Full results and interpretation: [`GUARDRAILS.md`](GUARDRAILS.md).

## 9. Final figures

`scripts/analysis/build_pdv_inverse_final_figures.py` produces the parameter-recovery scatter plots, repricing-by-maturity plot, fresh-seed robustness bar chart, and the 81-D Mahalanobis separation histogram referenced in the root README's [Selected Figures](../README.md#selected-figures) section.

## Earlier surrogate-ensemble exploration (superseded)

Before settling on the inverse-ANN + Mahalanobis-OOD architecture above, an earlier phase explored a *forward* surface surrogate (predicting the IV surface from PDV parameters) trained as a 5-model ensemble, with fallback guardrails driven by ensemble disagreement (`scripts/surrogate/train_pdv_surface_surrogate.py`, `evaluate_surrogate_ensemble_guardrails.py`, `summarize_two_lane_operating_points.py`). This exploration is retained in the repository for historical continuity but is not part of the final architecture described in the root README — the inverse-ANN direction with Mahalanobis/Ledoit-Wolf OOD detection is what the results in [`RESULTS.md`](RESULTS.md) describe.
