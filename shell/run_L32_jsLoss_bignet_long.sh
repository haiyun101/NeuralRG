#!/bin/bash -l
#SBATCH --job-name=L32_jb_long
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=5:00:00
#SBATCH --output=./logs/L32_jb_long_%j.out
#SBATCH --error=./logs/L32_jb_long_%j.err

# Fresh long L=32 jsLoss BIGNET run -- 8000 epochs from scratch.
# Replaces the broken phase-2 extension: resume + Adam reset destroyed the
# phase-1 progress (loss regressed by ~11 nat and never recovered after 3500
# epochs of extension training). A fresh single-run preserves continuous
# Adam state across the entire trajectory.
#
# Expected convergence (extrapolating original 3000-ep slope of -2.6 nat/1000ep
# with diminishing returns):
#   3000ep -> -212  (matches original run)
#   5000ep -> ~-217
#   8000ep -> ~-220 to -222  (KL_qp + KL_pq ~ 25, ~12 each direction)

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
N=200000
HS_PT="data/mcmc_data/hs_L32_T${T}_N${N}.pt"
FOLDER="./data/32Ising_T2.269_jsLoss_bignet_long_lam0.5"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"
echo "=========================================="
echo "L=32 jsLoss BIGNET long run (8000 ep, fresh, no resume)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python main.py \
    -L 32 -T "$T" \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 8000 \
    -batch 128 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -savePeriod 100 \
    -symmetry \
    -skipHMC \
    -jsLoss -jsLambda 0.5 \
    -dataPath "$HS_PT" \
    -noDeq

echo "Done."
