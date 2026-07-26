#!/bin/bash -l
#SBATCH --job-name=vuFM_L128_h192
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/vuFM_L128_h192_%j.out
#SBATCH --error=./logs/vuFM_L128_h192_%j.err

# VelocityUNet-FM h=192 at L=128 (~30M params).
# Fair-capacity control vs MERAUNet-FM L=128 h=128 (~86M).
# If VelocityUNet still loses despite matching/exceeding capacity of
# MERAUNet L=32 (17M), suggests MERA scale hierarchy adds value beyond
# just parameters.
#
# Config: matches meraFM_L128_v2 (lr=1e-4, gradClip=0.5, batch=16 for
# larger arch memory footprint).

module load miniforge
source activate neuralrg
cd /cluster/home/hhuang05/NeuralRG
mkdir -p logs

python -u train/fm_learn.py \
    -L 128 -T 2.269185314213022 \
    -folder ./data/L128_T2.269_velocityUNet_h192 \
    -epochs 2000 -batch 16 \
    -arch unet -nhidden 192 -tembDim 128 \
    -lr 1e-4 -gradClip 0.5 \
    -savePeriod 50 -samplePeriod 50 \
    -sampleSteps 50 -sampleN 128 \
    -cuda 0 -seed 0

date
