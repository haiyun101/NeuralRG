#!/bin/bash -l
#SBATCH --job-name=L16_jsLoss
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=3:00:00
#SBATCH --output=./logs/L16_jsLoss_%j.out
#SBATCH --error=./logs/L16_jsLoss_%j.err

# L=16 JS-like (symmetrized KL) training, lambda=0.5, DEFAULT arch.
# loss = 0.5 * KL(q||p) + 0.5 * KL(p||q)
#
# Fills the mid-L point between L=8 (already done) and L=32 (default + bignet
# done / in progress). Default arch (10/64) matches the existing L=16 sym
# and L=16 hs_dataDriven baselines for direct apples-to-apples comparison.
#
# Reference (L=16 T_c, default arch):
#   sym             (pure reverse-KL):  KL(q||p) = 2.43,  off-obj KL(p||q) = 10.82
#   hs_dataDriven   (pure forward-KL):  KL(p||q) = 4.97,  off-obj KL(q||p) = 3.79
#   THIS RUN        (JS, lam=0.5):      predicted both KL ~ 2-4 nat (filling the curve)
#
# Budget: 10000 epochs (~2-2.5h on l40 with JS 2x cost vs L=16 reverse-KL).

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
N=200000
HS_PT="data/mcmc_data/hs_L16_T${T}_N${N}.pt"
FOLDER="./data/16Ising_T2.269_jsLoss_lam0.5"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"
echo "=========================================="
echo "L=16 JS-like training  |  lambda=0.5  |  default arch"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python main.py \
    -L 16 -T "$T" \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 10000 \
    -batch 128 \
    -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 \
    -savePeriod 200 \
    -symmetry \
    -skipHMC \
    -jsLoss -jsLambda 0.5 \
    -dataPath "$HS_PT" \
    -noDeq

echo "Done."
