#!/bin/bash -l
#SBATCH --job-name=L32_physReg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L32_physReg_%j.out
#SBATCH --error=./logs/L32_physReg_%j.err

# Multi-observable physical regularizer sweep at L=32.
# Warm-starts each cell from champion (fixdil+VP-1e-3 nr=1 @ ep 9500) via
# same-L transfer (-loadFromSmallerL). Trains 5000 more epochs with
# χ + U4 physical regularizer at different weight settings.
#
# Sequential runs in one job to save queue slots. Total ~20h for 3 cells.
#
# Research hypothesis:
#   better physics fit → closer to RG fixed point → more self-similarity
#   → better cross-L transferability + better interpretability

module load miniforge
source activate neuralrg
mkdir -p logs

SRC_CKPT=data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64/savings/SymmMERA_l16_M3H128_R1_IsingSaving_epoch9500.saving
SRC_STRIDES="16,8,4,2,1"

# Common flags
COMMON_ARGS="-L 32 -T 2.269185314213022 \
    -dataDriven -skipHMC \
    -dataPath ./data/mcmc_data/hs_L32_T2.269185314213022_N200000.pt \
    -epochs 5000 -batch 64 -lr 3e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1 \
    -volumePreservingWeight 1e-3 \
    -loadFromSmallerL $SRC_CKPT \
    -loadFromSmallerLStrides $SRC_STRIDES \
    -physRegBatch 128 \
    -savePeriod 200 -cuda 0"

# ============================================================
# Cell 1: light regularizer (λ_χ = 0.01, λ_U4 = 0.01)
# ============================================================
echo "==================================================================="
echo "=== Cell 1: physReg light (λ_χ=0.01, λ_U4=0.01)"
echo "==================================================================="
python -u main.py $COMMON_ARGS \
    -folder ./data/32Ising_T2.269_physReg_chi0.01_u40.01 \
    -physRegWeightChi 0.01 -physRegWeightU4 0.01

echo
echo "==================================================================="
echo "=== Cell 2: physReg moderate (λ_χ=0.1, λ_U4=0.1)"
echo "==================================================================="
python -u main.py $COMMON_ARGS \
    -folder ./data/32Ising_T2.269_physReg_chi0.1_u40.1 \
    -physRegWeightChi 0.1 -physRegWeightU4 0.1

echo
echo "==================================================================="
echo "=== Cell 3: physReg strong (λ_χ=1.0, λ_U4=1.0)"
echo "==================================================================="
python -u main.py $COMMON_ARGS \
    -folder ./data/32Ising_T2.269_physReg_chi1.0_u41.0 \
    -physRegWeightChi 1.0 -physRegWeightU4 1.0

echo
echo "Done."
date
