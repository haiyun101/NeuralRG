#!/bin/bash -l
#SBATCH --job-name=L64_physReg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/L64_physReg_%j.out
#SBATCH --error=./logs/L64_physReg_%j.err

# L=64 champion warm-start + physical-observable regularizer (χ + U₄).
# Fills the physReg ladder: L=32 (sweep) → L=64 (this) → L=128 (already submitted).
#
# Weights: moderate (λ_χ = 0.1, λ_U4 = 0.1) — matches L=128 physReg config.
# If L=32 sweep says stronger/weaker is better, later runs can adjust.
#
# Warm-start from L=64 champion via same-L transfer (loadFromSmallerL).
# Also using bf16 for ~2× speedup (verified stable at L=32).

module load miniforge
source activate neuralrg
mkdir -p logs

python -u main.py \
    -L 64 -T 2.269185314213022 \
    -folder ./data/L64_T2.269_champion_physReg_chi0.1_u40.1 \
    -dataDriven -skipHMC \
    -dataPath ./data/mcmc_data/hs_L64_T2.269185314213022_N500000.pt \
    -epochs 5000 -batch 16 -lr 3e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1 \
    -volumePreservingWeight 1e-3 \
    -loadFromSmallerL data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16/savings/SymmMERA_l16_M3H128_R1_IsingSaving_epoch9500.saving \
    -loadFromSmallerLStrides "32,16,8,4,2,1" \
    -physRegWeightChi 0.1 -physRegWeightU4 0.1 \
    -physRegBatch 64 \
    -bf16 \
    -savePeriod 200 -cuda 0

date
