#!/bin/bash -l
#SBATCH --job-name=truemeraFM_L64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/truemeraFM_L64_%j.out
#SBATCH --error=./logs/truemeraFM_L64_%j.err

# TRUE MERA-structured velocity field for CFM (rectified flow), L=64.
# depth = 2 * log2(64) = 12 blocks. Each 2x2 patch processed by time-modulated
# MLP; deltas accumulate sequentially through scales.
#
# hidden=256, n_hidden_layers=2 => ~1.8M params. Compare against MERAUNet L=64
# @ 22M params — physical inductive bias should compensate for tighter budget.

module load miniforge
source activate neuralrg
mkdir -p logs

python -u train/fm_learn.py \
    -L 64 -T 2.269185314213022 \
    -folder ./data/L64_T2.269_truemeraFM_h256 \
    -epochs 5000 -batch 128 \
    -arch truemera \
    -nhidden 256 -tembDim 128 \
    -meraNrepeat 1 -meraHiddenLayers 2 \
    -lr 1e-3 -gradClip 1.0 \
    -savePeriod 200 -samplePeriod 200 \
    -sampleSteps 100 -sampleN 500 \
    -cuda 0 -seed 0

date
