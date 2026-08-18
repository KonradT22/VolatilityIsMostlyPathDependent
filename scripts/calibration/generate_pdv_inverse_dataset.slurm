#!/bin/bash
#SBATCH --job-name=pdv-inverse
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-99%10
#SBATCH --output=/users/4/trest017/urop_pdv/slurm_logs/pdv-inverse-%A_%a.out
#SBATCH --error=/users/4/trest017/urop_pdv/slurm_logs/pdv-inverse-%A_%a.err

set -euo pipefail

cd /users/4/trest017/urop_pdv/repo/VolatilityIsMostlyPathDependent

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CHUNK_SIZE=100
START_ROW=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))

/users/4/trest017/urop_pdv/envs/pdv-urop/bin/python \
  scripts/calibration/generate_pdv_inverse_dataset.py \
  --start-row "${START_ROW}" \
  --count "${CHUNK_SIZE}" \
  --n-paths 10000 \
  --seed-base 202608190000 \
  --chunk-id "${SLURM_ARRAY_TASK_ID}" \
  --output-dir \
  /users/4/trest017/urop_pdv/benchmarks/inverse_ann/pdv_anchor_jitter_10k
