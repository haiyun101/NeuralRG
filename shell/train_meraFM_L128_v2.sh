#!/bin/bash -l
#SBATCH --job-name=meraFM_L128_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/meraFM_L128_v2_%j.out
#SBATCH --error=./logs/meraFM_L128_v2_%j.err

# meraFM L=128 RESTART after divergence (original 1798868: loss 3→32→104).
# Fixes:
#   lr: 5e-4 → 1e-4 (5× smaller, prevent gradient overshoot at L=128 scale)
#   gradClip: 1.0 → 0.5 (tighter direction bound)
#   savePeriod: 100 → 50 (get checkpoints earlier for safety + physics)
#   samplePeriod: 100 → 50
# Fresh start (previous had no saved ckpts).

module load miniforge
source activate neuralrg
mkdir -p logs

python -u train/fm_learn.py \
    -L 128 -T 2.269185314213022 \
    -folder ./data/L128_T2.269_meraFM_h128_v2 \
    -epochs 2000 -batch 16 \
    -arch meraunet -nhidden 128 -tembDim 128 -maxChannelMult 4 \
    -lr 1e-4 -gradClip 0.5 \
    -savePeriod 50 -samplePeriod 50 \
    -sampleSteps 50 -sampleN 64 \
    -cuda 0 -seed 0

date
