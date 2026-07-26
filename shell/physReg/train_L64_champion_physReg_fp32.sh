#!/bin/bash -l
#SBATCH --job-name=L64_physReg_fp32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L64_physReg_fp32_%j.out
#SBATCH --error=./logs/L64_physReg_fp32_%j.err

# L=64 warm-start + physReg — fp32 CONTROL for the bf16 job 1799146.
# Same config as 1799146 but WITHOUT -bf16.
# Purpose: isolate bf16 cost from physReg cost in the final loss.
#   - bf16 warm+physReg (1799146): loss 7670 = champion + 11 nat
#   - fp32 warm+physReg (THIS):    loss ?     = attributes 11 nat between
#                                                physReg penalty vs bf16 precision loss
# Warm-start from L=64 champion @ ep 9500, 5000 epochs.

module load miniforge
source activate neuralrg
mkdir -p logs

python -u main.py \
    -L 64 -T 2.269185314213022 \
    -folder ./data/L64_T2.269_champion_physReg_chi0.1_u40.1_fp32 \
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
    -savePeriod 200 -cuda 0

date
