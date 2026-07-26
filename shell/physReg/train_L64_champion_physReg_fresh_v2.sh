#!/bin/bash -l
#SBATCH --job-name=L64_physReg_fresh_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/L64_physReg_fresh_v2_%j.out
#SBATCH --error=./logs/L64_physReg_fresh_v2_%j.err

# L=64 FRESH+physReg RESTART after 1801031 diverged at ep 399.
# Original diverged much earlier than warm version (399 vs 1299).
# Fixes:
#   -gradClip 1.0: prevent runaway gradient (was ABSENT)
#   -physRegBatch 64 → 16: less physReg gradient variance
#   -physRegWeightChi/U4 0.1 → 0.05: weaker constraint, less pressure to overshoot
#   -lr 3e-4 → 1e-4: gentler updates during physReg activation

module load miniforge
source activate neuralrg
mkdir -p logs

python -u main.py \
    -L 64 -T 2.269185314213022 \
    -folder ./data/L64_T2.269_physReg_fresh_chi0.05_u40.05_v2 \
    -dataDriven -skipHMC \
    -dataPath ./data/mcmc_data/hs_L64_T2.269185314213022_N500000.pt \
    -epochs 15000 -batch 16 -lr 1e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1 \
    -volumePreservingWeight 1e-3 \
    -physRegWeightChi 0.05 -physRegWeightU4 0.05 \
    -physRegBatch 16 \
    -gradClip 1.0 \
    -savePeriod 200 -cuda 0

date
