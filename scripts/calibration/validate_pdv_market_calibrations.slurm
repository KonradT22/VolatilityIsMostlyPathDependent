#!/bin/bash
#SBATCH --job-name=pdv-market-val
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/users/4/trest017/urop_pdv/slurm_logs/pdv-market-val-%j.out
#SBATCH --error=/users/4/trest017/urop_pdv/slurm_logs/pdv-market-val-%j.err

set -euo pipefail

cd /users/4/trest017/urop_pdv/repo/VolatilityIsMostlyPathDependent

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/users/4/trest017/urop_pdv/envs/pdv-urop/bin/python \
  scripts/calibration/validate_pdv_market_calibrations.py
