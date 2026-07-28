#!/bin/bash
#SBATCH --job-name=pdv_data
#SBATCH --account=dodso013
#SBATCH --partition=agsmall
#SBATCH --array=0-7
#SBATCH --output=/users/4/trest017/urop_pdv/slurm_logs/pdv_data_%A_%a.out
#SBATCH --error=/users/4/trest017/urop_pdv/slurm_logs/pdv_data_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

module purge
module load python3/3.10.9_anaconda2023.03_libmamba

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLBACKEND=Agg

cd /users/4/trest017/urop_pdv/repo/VolatilityIsMostlyPathDependent

/users/4/trest017/urop_pdv/envs/pdv-urop/bin/python \
  scripts/surrogate/generate_pdv_surface_dataset.py \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --samples-per-shard 64 \
  --n-paths 4000
