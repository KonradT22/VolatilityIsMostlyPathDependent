#!/bin/bash
#SBATCH --job-name=pdv-mkt5-jun2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/users/4/trest017/urop_pdv/slurm_logs/pdv-mkt5-jun2-%j.out
#SBATCH --error=/users/4/trest017/urop_pdv/slurm_logs/pdv-mkt5-jun2-%j.err

set -euo pipefail

cd /users/4/trest017/urop_pdv/repo/VolatilityIsMostlyPathDependent

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/users/4/trest017/urop_pdv/envs/pdv-urop/bin/python \
  scripts/calibration/calibrate_pdv_market_5param_fixed_grid.py \
  --trade-date 2021-06-02 \
  --n-paths 4000 \
  --seed-root 2026080701 \
  --maxiter 350 \
  --maxfev 700
