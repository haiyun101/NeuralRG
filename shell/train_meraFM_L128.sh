#!/bin/bash -l
#SBATCH --job-name=meraFM_L128
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/meraFM_L128_%j.out
#SBATCH --error=./logs/meraFM_L128_%j.err
#SBATCH --dependency=afterok:1796152

# Physics-aware Flow Matching L=128 T_c.
# MERAUNet: log2(128)=7 downsampling stages, ~104M params @ nhidden=128.
# CFM (rectified-flow) MSE loss on L=128 HS data (from 1796152 gen).
#
# ─── Memory sanity (A100 40GB) ─────────────────────────────────
#   params:         0.42 GB (fp32)
#   grads:          0.42 GB (fp32)
#   optimizer:      0.84 GB (Adam m/v)
#   activations:    ~2.5 GB @ batch=16 (fp32, 8 scales × ResBlock stages)
#   sampling (RK4-50 at physReg eval): batched to sampleN=64 for safety
#   total peak:     ~5-8 GB → comfortable headroom
#
# ─── Wall time budget ──────────────────────────────────────────
#   L=64 h128 batch=96 = 10.5 min/epoch (measured from job 1777304)
#   L=128 h128 batch=16 = ~8.4 min/epoch (compute ratio × batch ratio)
#   48h walltime → ~340 epochs
#   savePeriod=100 → 3 checkpoints + physics readouts
#
# ─── Depends on ────────────────────────────────────────────────
#   1796152 (L=128 N=100K HS data gen) — must complete first

module load miniforge
source activate neuralrg
mkdir -p logs

python -u train/fm_learn.py \
    -L 128 -T 2.269185314213022 \
    -folder ./data/L128_T2.269_meraFM_h128 \
    -epochs 2000 -batch 16 \
    -arch meraunet -nhidden 128 -tembDim 128 -maxChannelMult 4 \
    -lr 5e-4 -gradClip 1.0 \
    -savePeriod 100 -samplePeriod 100 \
    -sampleSteps 50 -sampleN 64 \
    -cuda 0 -seed 0

date
