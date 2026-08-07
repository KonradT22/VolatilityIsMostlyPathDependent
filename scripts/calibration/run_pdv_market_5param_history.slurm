#!/bin/bash
#SBATCH --job-name=pdv-mkt5-hist
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/users/4/trest017/urop_pdv/slurm_logs/pdv-mkt5-hist-%j.out
#SBATCH --error=/users/4/trest017/urop_pdv/slurm_logs/pdv-mkt5-hist-%j.err

set -euo pipefail

cd /users/4/trest017/urop_pdv/repo/VolatilityIsMostlyPathDependent

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/users/4/trest017/urop_pdv/envs/pdv-urop/bin/python \
  scripts/calibration/run_pdv_market_5param_history.py
