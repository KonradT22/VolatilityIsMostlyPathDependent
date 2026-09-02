# Reproducibility

## Environment

- Python 3.10 (matches the MSI `python3/3.10.9_anaconda2023.03_libmamba` module used for all SLURM jobs in this repo).
- Core dependencies: `requirements.txt` (matplotlib, numpy, pandas, scikit-learn, scipy, seaborn, statsmodels, torch, yfinance, jupyterlab). These are unpinned, matching the original upstream `requirements.txt` — exact versions used for the results in this repo were not recorded, so bit-for-bit reproducibility of Monte Carlo output at the floating-point level is not guaranteed across environments.
- ThetaData ingestion (`scripts/thetadata/`) additionally requires the `thetadata` Python client — see `requirements-thetadata.txt` — plus a running ThetaData Terminal and a paid ThetaData subscription (SPX/SPXW options data). This is not required for anything outside `scripts/thetadata/`.

## What is / isn't reproducible from this repository alone

| Stage | Reproducible here? |
| --- | --- |
| Upstream PDV Monte Carlo engine, both notebooks | Yes — `pip install -r requirements.txt` and run |
| ThetaData ingestion (`scripts/thetadata/`) | Only with a licensed ThetaData subscription; raw quotes are not redistributed |
| Historical calibration, synthetic dataset generation, ANN training/eval, guardrail analysis | Yes, once ThetaData-derived option-surface CSVs exist locally, following the pipeline order in [`RESEARCH_OVERVIEW.md`](RESEARCH_OVERVIEW.md) |
| Exact numeric results in [`RESULTS.md`](RESULTS.md) | Not guaranteed bit-for-bit (unpinned dependency versions, MSI-specific hardware); the pipeline and seeds are reproducible, the last-decimal-place output is not promised |

There is no single "run everything" command. The project was developed incrementally across many independent MSI SLURM jobs, each writing intermediate CSV/JSON outputs consumed by the next stage.

## MSI / SLURM workflow

All `scripts/**/*.py` files that were run on MSI have a matching `.slurm` submission script in the same directory (e.g., `scripts/calibration/generate_pdv_inverse_dataset.py` ↔ `scripts/calibration/generate_pdv_inverse_dataset.slurm`). These scripts:

- Load MSI's `python3/3.10.9_anaconda2023.03_libmamba` module and a project-specific virtualenv.
- Use SLURM job arrays for embarrassingly parallel stages — e.g., synthetic dataset generation runs as 100 array tasks of 100 candidates each (`--array=0-99%10`); fresh-seed repricing runs as 8 array tasks (`--array=0-7%8`).
- Write outputs under a fixed base directory (`/users/4/trest017/urop_pdv/...` in the committed scripts).

**On the hardcoded MSI paths:** the scripts as committed hardcode one author's MSI home-directory paths (e.g., `BASE = Path("/users/4/trest017/urop_pdv/...")`) rather than taking them as CLI arguments or environment variables. This was a deliberate low-risk choice when polishing this repository — these are historical research scripts that produced the results in `RESULTS.md` under those exact paths, and rewriting them to be path-agnostic risks silently changing behavior without a way to re-verify the original numbers. To reproduce on a different system, update the `BASE` / `OUTDIR` / path constants at the top of each script to your own environment before running.

## Seeds

Seeds are fixed and recorded per stage, not shared globally:

- Historical calibration: `seed_root=2026080701` (default) at 4,000 paths.
- Independent historical validation: seeds `2026080720, 2026080721, 2026080722` at 20,000 paths.
- Synthetic dataset generation: `seed_base=202608190000 + sample_id` at 10,000 paths.
- Inverse ANN training: `seed=2026081801`.
- Fresh-seed repricing: seeds `2026081840, 2026081841, 2026081842` (previously unused in training or same-seed repricing), with scenario selection seeded at `2026081843`.
- OOD detector fit/calibration split: `seed=2026081805`.

## What is not redistributed

- Raw or processed ThetaData SPX/SPXW quotes (`.gitignore`d under `data/`).
- Generated synthetic datasets, trained model checkpoints, and result CSV/JSON files (`.gitignore`d under `benchmarks/`).
- ThetaData credentials (`.theta.env`, `.theta_creds.txt`, `creds.txt`, `*.env` — all gitignored; credentials are loaded from a dotenv file outside the repository, never hardcoded in source).

## Licensing

No LICENSE file currently exists in this repository, and none was present upstream. Any future licensing decision should account for the provenance of the upstream implementation.
