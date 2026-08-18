#!/bin/bash
#SBATCH --job-name=pdv-inv-int
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/users/4/trest017/urop_pdv/slurm_logs/pdv-inv-int-%j.out
#SBATCH --error=/users/4/trest017/urop_pdv/slurm_logs/pdv-inv-int-%j.err

set -euo pipefail

cd /users/4/trest017/urop_pdv/repo/VolatilityIsMostlyPathDependent

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

/users/4/trest017/urop_pdv/envs/pdv-urop/bin/python \
  scripts/surrogate/train_pdv_inverse_mlp.py \
  --split-dir \
  /users/4/trest017/urop_pdv/benchmarks/inverse_ann/pdv_anchor_jitter_10k/splits_interpolation \
  --output-dir \
  /users/4/trest017/urop_pdv/benchmarks/inverse_ann/pdv_anchor_jitter_10k/models/mlp_interpolation \
  --seed 2026081801 \
  --epochs 1000 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --patience 100
