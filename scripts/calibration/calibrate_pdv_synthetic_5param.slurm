#!/bin/bash
#SBATCH --job-name=pdv-synth5
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/users/4/trest017/urop_pdv/slurm_logs/pdv-synth5-%j.out
#SBATCH --error=/users/4/trest017/urop_pdv/slurm_logs/pdv-synth5-%j.err

set -euo pipefail

cd /users/4/trest017/urop_pdv/repo/VolatilityIsMostlyPathDependent

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/users/4/trest017/urop_pdv/envs/pdv-urop/bin/python \
  scripts/calibration/calibrate_pdv_synthetic_5param.py
