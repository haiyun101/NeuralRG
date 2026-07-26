#!/bin/bash -l
#SBATCH --job-name=truemeraFM_L32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/truemeraFM_L32_%j.out
#SBATCH --error=./logs/truemeraFM_L32_%j.err

# TRUE MERA-structured velocity field for CFM (rectified flow).
# Reuses MERA's im2col dispatch/collect at 2 * log2(L) = 10 layers.
# Each 2x2 patch is processed by a time-modulated MLP producing a velocity
# delta; deltas accumulate sequentially. Not invertible.
#
# hidden=256, n_hidden_layers=2 => ~1.5M params (much smaller than MERAUNet
# 17M but with strong physical inductive bias).

module load miniforge
source activate neuralrg
mkdir -p logs

python -u train/fm_learn.py \
    -L 32 -T 2.269185314213022 \
    -folder ./data/L32_T2.269_truemeraFM_h256 \
    -epochs 5000 -batch 128 \
    -arch truemera \
    -nhidden 256 -tembDim 128 \
    -meraNrepeat 1 -meraHiddenLayers 2 \
    -lr 1e-3 -gradClip 1.0 \
    -savePeriod 200 -samplePeriod 200 \
    -sampleSteps 100 -sampleN 500 \
    -cuda 0 -seed 0

date
