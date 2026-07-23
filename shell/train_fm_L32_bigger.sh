#!/bin/bash -l
#SBATCH --job-name=fm_L32_big
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/fm_L32_big_%j.out
#SBATCH --error=./logs/fm_L32_big_%j.err

# Bigger flow matching L=32 T_c: nhidden=128 (2x wider, ~4M params vs 1M),
# batch=256 (2x, halves per-epoch step count and improves gradient stability
# for the plateaued MSE loss), gradient clip 1.0.
#
# Motivation: nhidden=64 prototype (job 41818094) reached χ=33.00 (98% of GT)
# at ep 1000 but its CFM loss plateaued at ~3.09 from ep 500. The plateau
# suggests the small arch has saturated. Bigger nhidden should push loss
# lower, and we'll see if physics stays at GT-level or improves further.

module load miniforge
source activate neuralrg

mkdir -p logs

L=32
T=2.269185314213022
FOLDER="./data/L${L}_T2.269_flowmatching_h128"

echo "=========================================="
echo "Flow Matching L=$L, T=$T — bigger model"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "  nhidden=128, batch=256, lr=1e-3"
echo "=========================================="
date

python -u train/fm_learn.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -epochs 20000 -batch 256 \
    -nhidden 128 -tembDim 128 \
    -lr 1e-3 -gradClip 1.0 \
    -savePeriod 500 -samplePeriod 500 \
    -sampleSteps 100 -sampleN 500 \
    -cuda 0 -seed 0

echo "Done."
date
