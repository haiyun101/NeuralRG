#!/bin/bash -l
#SBATCH --job-name=fm_L64_b256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/fm_L64_b256_%j.out
#SBATCH --error=./logs/fm_L64_b256_%j.err

# Continuation of L=64 FM training with batch=256 (2x from 128) — target ~2x
# throughput. Job 1717126 hit walltime at ep 220 with batch=128 (192 s/epoch).
# batch=256 should push to ~450 epochs in 12h.
# Also resume from ep 200 checkpoint with -load so we don't waste the 11 h
# already invested.

module load miniforge
source activate neuralrg
mkdir -p logs

L=64
T=2.269185314213022
FOLDER="./data/L64_T2.269_flowmatching_h64"

echo "=========================================="
echo "FM L=$L T=$T  continuation from ep 200  batch=256"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="
date

# NOTE: fm_learn.py -load picks up the latest checkpoint automatically.
# We override batch here on CLI (200-epoch save + physics-eval every 200).
python -u train/fm_learn.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -epochs 5000 -batch 256 \
    -nhidden 64 -tembDim 128 \
    -lr 1e-3 -gradClip 1.0 \
    -savePeriod 200 -samplePeriod 200 \
    -sampleSteps 100 -sampleN 500 \
    -cuda 0 -seed 0 \
    -load

date
