#!/bin/bash -l
#SBATCH --job-name=vuFM_L32_h192
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/vuFM_L32_h192_%j.out
#SBATCH --error=./logs/vuFM_L32_h192_%j.err

# VelocityUNet-FM h=192 at L=32 (~30M params).
# Fair-capacity control vs MERAUNet-FM L=32 h=64 (~17M).
# VelocityUNet with 1.7× MORE capacity than MERAUNet — if it still
# loses on physics, hierarchy is doing real work beyond parameter count.

module load miniforge
source activate neuralrg
cd /cluster/home/hhuang05/NeuralRG
mkdir -p logs

python -u train/fm_learn.py \
    -L 32 -T 2.269185314213022 \
    -folder ./data/L32_T2.269_velocityUNet_h192 \
    -epochs 5000 -batch 64 \
    -arch unet -nhidden 192 -tembDim 128 \
    -lr 5e-4 -gradClip 1.0 \
    -savePeriod 100 -samplePeriod 100 \
    -sampleSteps 50 -sampleN 500 \
    -cuda 0 -seed 0

date
