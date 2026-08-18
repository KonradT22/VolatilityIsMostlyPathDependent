#!/bin/bash
#SBATCH --job-name=pdv-reprice
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --array=0-19%10
#SBATCH --output=/users/4/trest017/urop_pdv/slurm_logs/pdv-reprice-%A_%a.out
#SBATCH --error=/users/4/trest017/urop_pdv/slurm_logs/pdv-reprice-%A_%a.err

set -euo pipefail

cd /users/4/trest017/urop_pdv/repo/VolatilityIsMostlyPathDependent

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CHUNK_SIZE=50
START_ROW=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))

/users/4/trest017/urop_pdv/envs/pdv-urop/bin/python \
  scripts/surrogate/reprice_pdv_inverse_test.py \
  --start-row "${START_ROW}" \
  --count "${CHUNK_SIZE}" \
  --chunk-id "${SLURM_ARRAY_TASK_ID}" \
  --n-paths 10000
