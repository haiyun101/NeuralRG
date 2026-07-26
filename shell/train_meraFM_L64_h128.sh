#!/bin/bash -l
#SBATCH --job-name=meraFM_L64_h128
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=./logs/meraFM_L64_h128_%j.out
#SBATCH --error=./logs/meraFM_L64_h128_%j.err

# Physics-aware Flow Matching L=64 T_c, LARGER capacity (nhidden=128).
# Motivation: the nhidden=64 run (job 1773878) had persistent numerical
# spikes (ep 0: 1e6, ep 60: 1.3e4), suggesting the 6-scale U-Net cannot
# express the field with only 64 base channels.
#
# nhidden=128, maxChannelMult=4  =>  channel schedule [128, 256, 512, 512, 512, 512, 512]
# Estimated params: ~80M (vs h64: 22M). Fits A100 40GB w/ batch=128 comfortably.
# batch=96 as conservative starting point in case forward pass OOMs.

module load miniforge
source activate neuralrg
mkdir -p logs

python -u train/fm_learn.py \
    -L 64 -T 2.269185314213022 \
    -folder ./data/L64_T2.269_meraFM_h128 \
    -epochs 5000 -batch 96 \
    -arch meraunet -nhidden 128 -tembDim 128 -maxChannelMult 4 \
    -lr 5e-4 -gradClip 1.0 \
    -savePeriod 200 -samplePeriod 200 \
    -sampleSteps 100 -sampleN 500 \
    -cuda 0 -seed 0

date
