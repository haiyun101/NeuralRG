#!/bin/bash -l
#SBATCH --job-name=L64_physReg_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L64_physReg_v2_%j.out
#SBATCH --error=./logs/L64_physReg_v2_%j.err

# L=64 warm+physReg RESTART after 1801086 diverged at ep 1299.
# Trajectory was healthy through ep 1000 (loss ~7710) then blew up.
# Fixes:
#   -gradClip 1.0: prevent the specific bad gradient step (was ABSENT)
#   -physRegBatch 64 → 16: smaller sample, less variance in physReg gradient
# Same λ_χ=0.1, λ_U4=0.1 as 1801086 (test if fixes are sufficient at that λ)
# Warm-start from L=64 champion again (fresh folder).

module load miniforge
source activate neuralrg
mkdir -p logs

python -u main.py \
    -L 64 -T 2.269185314213022 \
    -folder ./data/L64_T2.269_champion_physReg_chi0.1_u40.1_fp32_v2 \
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
    -physRegBatch 16 \
    -gradClip 1.0 \
    -savePeriod 200 -cuda 0

date
