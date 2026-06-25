#!/bin/bash
#SBATCH --job-name=pdv_py_base
#SBATCH --account=dodso013
#SBATCH --partition=agsmall
#SBATCH --output=/users/4/trest017/urop_pdv/slurm_logs/python_baseline_%j.out
#SBATCH --error=/users/4/trest017/urop_pdv/slurm_logs/python_baseline_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module purge
module load python3/3.10.9_anaconda2023.03_libmamba

export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /users/4/trest017/urop_pdv/repo/VolatilityIsMostlyPathDependent

/users/4/trest017/urop_pdv/envs/pdv-urop/bin/python scripts/run_python_baseline.py
