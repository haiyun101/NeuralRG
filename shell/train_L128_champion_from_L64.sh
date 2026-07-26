#!/bin/bash -l
#SBATCH --job-name=L128_from_L64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L128_from_L64_%j.out
#SBATCH --error=./logs/L128_from_L64_%j.err

# L=128 champion warm-start from L=64 champion (RG universality scaling test).
# Source strides [32, 16, 8, 4, 2, 1] (L=64 HCG defaults).
# L=128 will have strides [64, 32, 16, 8, 4, 2, 1]; only stride-64 CNN
# stays fresh (L=64 has no counterpart there).
#
# L=128 lattice is 4× area vs L=64, so batch scaled down to 8 (from L=64's 16).
# nrepeat=1 to keep memory low. hidden 128 matches L=32/L=64 champion arch.
#
# DEPENDENCY: waits on L=128 HS data generation (job before) via --afterok.

module load miniforge
source activate neuralrg
mkdir -p logs

python -u main.py \
    -L 128 -T 2.269185314213022 \
    -folder ./data/L128_T2.269_champion_from_L64 \
    -dataDriven -skipHMC \
    -dataPath ./data/mcmc_data/hs_L128_T2.269185314213022_N100000.pt \
    -epochs 15000 -batch 8 -lr 3e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1 \
    -volumePreservingWeight 1e-3 \
    -loadFromSmallerL data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16/savings/SymmMERA_l16_M3H128_R1_IsingSaving_epoch9500.saving \
    -loadFromSmallerLStrides "32,16,8,4,2,1" \
    -savePeriod 200 -cuda 0

date
