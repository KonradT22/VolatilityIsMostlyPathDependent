#!/bin/bash
#SBATCH --job-name=pdv_nb_cpu
#SBATCH --account=dodso013
#SBATCH --partition=agsmall
#SBATCH --output=/users/4/trest017/urop_pdv/slurm_logs/pdv_notebook_%j.out
#SBATCH --error=/users/4/trest017/urop_pdv/slurm_logs/pdv_notebook_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module purge
module load python3/3.10.9_anaconda2023.03_libmamba

export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ENV_PY=/users/4/trest017/urop_pdv/envs/pdv-urop/bin/python
REPO=/users/4/trest017/urop_pdv/repo/VolatilityIsMostlyPathDependent
OUTDIR=/users/4/trest017/urop_pdv/benchmarks/msi_baseline

cd "$REPO"
mkdir -p "$OUTDIR"

echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "Python executable:"
$ENV_PY -c "import sys; print(sys.executable); print(sys.version)"
echo ""

echo "Package check:"
$ENV_PY -c "import yfinance, torch, pandas, numpy; print('yfinance OK'); print('torch', torch.__version__); print('pandas', pandas.__version__); print('numpy', numpy.__version__)"
echo ""

echo "Available kernels:"
$ENV_PY -m jupyter kernelspec list
echo ""

echo "Running notebook with pdv-urop kernel..."
$ENV_PY -m nbconvert \
  --to notebook \
  --execute option_pricing_4fmpdv.ipynb \
  --output "$OUTDIR/option_pricing_4fmpdv_executed.ipynb" \
  --ExecutePreprocessor.kernel_name=pdv-urop \
  --ExecutePreprocessor.timeout=1800

echo "Finished at: $(date)"
