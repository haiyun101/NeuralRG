#!/bin/bash -l
#SBATCH --job-name=pureFM_L128
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/pureFM_L128_%j.out
#SBATCH --error=./logs/pureFM_L128_%j.err

# Pure Flow Matching at L=128 — VelocityUNet baseline (2-stage naive
# U-Net, 3.43M params). Comparison against MERAUNet-FM (~86M params at
# L=128 h128) to isolate whether MERA-style scale hierarchy adds value
# vs just having more parameters.
#
# Config matches meraFM_L128_v2 stability fixes: lr=1e-4, gradClip=0.5,
# savePeriod=50 for early physics readouts.
#
# Small param count → can afford batch=32 (vs 16 for MERAUNet).

module load miniforge
source activate neuralrg
cd /cluster/home/hhuang05/NeuralRG
mkdir -p logs

python -u train/fm_learn.py \
    -L 128 -T 2.269185314213022 \
    -folder ./data/L128_T2.269_pureFM_h64 \
    -epochs 2000 -batch 32 \
    -arch unet -nhidden 64 -tembDim 128 \
    -lr 1e-4 -gradClip 0.5 \
    -savePeriod 50 -samplePeriod 50 \
    -sampleSteps 50 -sampleN 128 \
    -cuda 0 -seed 0

date
