#!/bin/bash -l
#SBATCH --job-name=meraFM_L64_h128_c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/meraFM_L64_h128_c_%j.out
#SBATCH --error=./logs/meraFM_L64_h128_c_%j.err

# Continue meraFM_L64_h128 (previously timed out at ep 80).
# Same config, uses -load to resume from latest checkpoint.
# Goal: reach ep 200+ to get first physics readouts.

module load miniforge
source activate neuralrg
mkdir -p logs

python -u train/fm_learn.py \
    -L 64 -T 2.269185314213022 \
    -folder ./data/L64_T2.269_meraFM_h128 \
    -load \
    -epochs 5000 -batch 96 \
    -arch meraunet -nhidden 128 -tembDim 128 -maxChannelMult 4 \
    -lr 5e-4 -gradClip 1.0 \
    -savePeriod 50 -samplePeriod 50 \
    -sampleSteps 50 -sampleN 200 \
    -cuda 0 -seed 0

date
