#!/bin/bash
#SBATCH --job-name=pdv_ensemble
#SBATCH --account=dodso013
#SBATCH --partition=agsmall
#SBATCH --array=0-4
#SBATCH --output=/users/4/trest017/urop_pdv/slurm_logs/pdv_ensemble_%A_%a.out
#SBATCH --error=/users/4/trest017/urop_pdv/slurm_logs/pdv_ensemble_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module purge
module load python3/3.10.9_anaconda2023.03_libmamba

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg

cd /users/4/trest017/urop_pdv/repo/VolatilityIsMostlyPathDependent

SEED=$((20260728 + SLURM_ARRAY_TASK_ID))
OUT_DIR="/users/4/trest017/urop_pdv/benchmarks/surrogate/pilot_surface_ann_ensemble/model_${SLURM_ARRAY_TASK_ID}"

/users/4/trest017/urop_pdv/envs/pdv-urop/bin/python \
  scripts/surrogate/train_pdv_surface_surrogate.py \
  --seed "$SEED" \
  --out-dir "$OUT_DIR"
