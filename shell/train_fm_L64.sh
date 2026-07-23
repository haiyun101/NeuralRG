#!/bin/bash -l
#SBATCH --job-name=fm_L64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/fm_L64_%j.out
#SBATCH --error=./logs/fm_L64_%j.err

# Flow matching L=64 T_c.
#
# The real test: L=32 was already saturated by the RNVP+MERA champion
# (KL ≈ 0). L=64 is where we saw the champion's forward-KL and physics
# deficits (KL ≈ 37, χ ≈ 66% of GT). If FM at L=64 maintains "near-zero
# KL AND near-perfect physics" from L=32, that's the definitive evidence
# FM structurally beats RNVP+MERA on tail modeling.
#
# Config:
#   Same U-Net width as L=32 prototype (nhidden=64) — 4× more sites but
#   the architecture is shape-agnostic. Runs 64→32→16 downsampling.
#   Batch 128 same as L=32 for consistency (memory OK on A100 40GB).
#   No sigma standardization: FM sees raw HS data (std ≈ 3.5).

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T=2.269185314213022
FOLDER="./data/L${L}_T2.269_flowmatching_h64"

echo "=========================================="
echo "Flow Matching L=$L T=$T (the real test)"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "  nhidden=64, batch=128"
echo "=========================================="
date

python -u train/fm_learn.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -epochs 5000 -batch 128 \
    -nhidden 64 -tembDim 128 \
    -lr 1e-3 -gradClip 1.0 \
    -savePeriod 200 -samplePeriod 200 \
    -sampleSteps 100 -sampleN 500 \
    -cuda 0 -seed 0

echo "Done."
date
