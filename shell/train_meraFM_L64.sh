#!/bin/bash -l
#SBATCH --job-name=meraFM_L64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/meraFM_L64_%j.out
#SBATCH --error=./logs/meraFM_L64_%j.err

# Physics-aware Flow Matching L=64 T_c.
# MERAUNet: log2(64)=6 downsampling stages matching MERA scale hierarchy.

module load miniforge
source activate neuralrg
mkdir -p logs

python -u train/fm_learn.py \
    -L 64 -T 2.269185314213022 \
    -folder ./data/L64_T2.269_meraFM_h64 \
    -epochs 5000 -batch 128 \
    -arch meraunet -nhidden 64 -tembDim 128 -maxChannelMult 4 \
    -lr 1e-3 -gradClip 1.0 \
    -savePeriod 200 -samplePeriod 200 \
    -sampleSteps 100 -sampleN 500 \
    -cuda 0 -seed 0

date
