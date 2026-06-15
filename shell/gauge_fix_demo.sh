#!/bin/bash -l
#SBATCH --job-name=gauge_fix_demo
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=./logs/gauge_demo_%j.out
#SBATCH --error=./logs/gauge_demo_%j.err

# Scheme V.1 demo: gauge-fixed layer-by-layer interpretation, on hs_bignet
# (the cleanest fwd-KL baseline). Post-training analysis only; no flow re-train.
#
# Output:
#   - data/32Ising_T2.269_hs_bignet/gauge_transforms.pt
#       (per-layer per-site quantile knots used to map y_s marginal → N(0,1))
#   - stdout: zscore-MSE vs gauge-fixed MSE for every adjacent scale pair,
#     plus the V3-pairing-readme operational guide.
#
# Tunables via env:
#   FOLDER  (default data/32Ising_T2.269_hs_bignet — the cleanest converged fwd-KL run)
#   N_SAMPLES (default 4000)
#   N_KNOTS   (default 128)

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDER="${FOLDER:-data/32Ising_T2.269_hs_bignet}"
N_SAMPLES="${N_SAMPLES:-4000}"
N_KNOTS="${N_KNOTS:-128}"

echo "=========================================="
echo "Gauge-fix demo  |  folder=$FOLDER  N=$N_SAMPLES K=$N_KNOTS"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="

python -u analyzers/rg_fixed_point/gauge_fix.py \
    --folder "$FOLDER" \
    --n-samples "$N_SAMPLES" \
    --n-knots "$N_KNOTS"

echo "Done."
