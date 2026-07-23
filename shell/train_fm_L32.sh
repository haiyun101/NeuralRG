#!/bin/bash -l
#SBATCH --job-name=fm_L32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/fm_L32_%j.out
#SBATCH --error=./logs/fm_L32_%j.err

# Flow-matching L=32 T_c prototype.
# Trains VelocityUNet with CFM (OT-path) loss on HS-transformed Wolff MCMC data.
# Compare to Champion (Best-200 ≈ 1891 at L=32 with hcg_shared nr=1): can flow
# matching hit similar KL, and can it produce more accurate χ / U₄?
#
# Model: U-Net with nhidden=64 base channels → ~1M params. Adam lr=1e-3.
# Data: hs_L32_T2.269_N200000.pt (200k Wolff samples, HS-transformed).
# Sampling: Euler ODE with 50 steps per periodic eval (200 samples per eval).

module load miniforge
source activate neuralrg

mkdir -p logs

L=32
T=2.269185314213022
FOLDER="./data/L${L}_T2.269_flowmatching_h64"

echo "=========================================="
echo "Flow Matching prototype L=$L, T=$T"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="
date

python -u train/fm_learn.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -epochs 20000 -batch 128 \
    -nhidden 64 -tembDim 128 \
    -lr 1e-3 -gradClip 1.0 \
    -savePeriod 1000 -samplePeriod 500 \
    -sampleSteps 50 -sampleN 500 \
    -cuda 0 -seed 0

echo "Done."
date
