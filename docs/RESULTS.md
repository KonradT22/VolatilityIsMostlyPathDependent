# Results

Detailed metrics for the inverse PDV surrogate. Summary figures live in the root [`README.md`](../README.md#key-results); this document gives the full breakdown and states exactly which script produced each number.

**Note on provenance:** the metrics below are the author's reported results from running the pipeline described in [`RESEARCH_OVERVIEW.md`](RESEARCH_OVERVIEW.md) on MSI. The underlying result CSV/JSON files are generated artifacts under a gitignored `benchmarks/` directory and are not included in this repository — see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for how to regenerate them from scratch.

## Historical calibration and validation

- 20 historical SPXW/SPX surfaces, May 5 – June 2, 2021, ~3:55pm ET quote snapshots.
- Each date calibrated with a restricted five-parameter Nelder-Mead search (`beta0`, `beta1`, `beta2`, `theta1`, `theta2`) against the fixed 77-coordinate option grid, using 4,000 Monte Carlo paths per evaluation (`scripts/calibration/run_pdv_market_5param_history.py`).
- Each calibration independently re-validated at 20,000 paths under three additional Monte Carlo seeds (`scripts/calibration/validate_pdv_historical_parameters.py`).
- **16 of 20 dates** had all three validation seeds numerically stable (positive, finite volatility throughout) and became robust historical anchors. **4 of 20 dates** failed validation (nonpositive/nonfinite volatility on at least one seed) and were excluded.

## Synthetic inverse-learning dataset

- 10,000 candidate parameter/state perturbations sampled by picking one of the 16 robust anchors uniformly at random, then applying independent Gaussian jitter in bounded logit space (`scripts/calibration/build_pdv_anchor_jitter_prior.py`).
- Each candidate priced across the full 77-coordinate grid at 10,000 Monte Carlo paths (`scripts/calibration/generate_pdv_inverse_dataset.py`, run as 100 SLURM array chunks of 100 candidates each).
- **9,667 of 10,000** candidates produced a fully valid, finite, positive-volatility 77-price surface; 333 were rejected.

## Inverse ANN test performance

Frozen 959-scenario in-domain interpolation test set (`scripts/surrogate/evaluate_pdv_inverse_mlp_test.py`), architecture `81 → 256 → 256 → 128 → 5`, GELU, standardized inputs/targets:

| Parameter | Test R² |
| --- | ---: |
| beta0 | 0.9934 |
| beta1 | 0.9883 |
| beta2 | 0.9943 |
| theta1 | 0.9689 |
| theta2 | 0.9933 |

## Repricing evaluation (same-seed, held-out)

Held-out ANN parameter predictions (after bound projection) repriced through the full PDV Monte Carlo model and compared to the true 77-coordinate surface (`scripts/surrogate/reprice_pdv_inverse_test.py`):

| Metric | Value |
| --- | ---: |
| Scenarios repriced | 955 / 959 |
| Mean surface RMSE | 1.41 bp of forward |
| Median surface RMSE | 1.05 bp |
| 95th percentile RMSE | 3.86 bp |
| Maximum RMSE | 9.31 bp |

4 of 959 test predictions remained numerically invalid after bound projection and were excluded from repricing.

## Fresh-seed robustness

64 balanced scenarios (4 per historical anchor × 16 anchors) repriced under 3 previously unused Monte Carlo seeds — 192 paired true-vs-predicted comparisons (`scripts/surrogate/reprice_pdv_inverse_fresh_seeds.py`):

| Metric | Value |
| --- | ---: |
| Paired comparisons | 192 / 192 completed |
| Mean surface RMSE | 1.32 bp of forward |

Performance was consistent across all three seeds — no seed showed materially different error.

## Latency

CPU-only neural-network forward pass, batched inference over the 959-row test set (`scripts/surrogate/reprice_pdv_inverse_test.py`, timing block):

| Metric | Value |
| --- | ---: |
| Median forward-pass latency | 0.109 ms |
| p99 forward-pass latency | ≈ 0.120 ms |

This is the ANN forward pass only — not the end-to-end guarded pipeline (see [Limitations](../README.md#what-failed--limitations)).

## OOD / guardrail diagnostics

Full detail in [`GUARDRAILS.md`](GUARDRAILS.md). Headline numbers (`scripts/surrogate/analyze_pdv_mahalanobis_guardrail.py`), 81-D full-feature detector:

| Metric | Value |
| --- | ---: |
| ID p99 threshold (squared Mahalanobis distance) | ≈ 50.85 |
| ID calibration false-positive rate at p99 | ≈ 1.01% |
| Chronological validation + test flag rate | 100% |
| AUROC (ID calibration vs. chronological OOD) | ≈ 0.999999 |

Chronological validation/test anchors: 2021-05-27, 2021-05-28 (validation), 2021-06-01, 2021-06-02 (test) — all four were flagged OOD at the p99 threshold in every tested observation.
