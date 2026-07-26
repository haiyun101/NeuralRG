#!/bin/bash -l
#SBATCH --job-name=L64_from_L32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L64_from_L32_%j.out
#SBATCH --error=./logs/L64_from_L32_%j.err

# RG universality warm-start experiment:
# Train an L=64 fixdil+VP-1e-3 nr=1 champion-config run initialised from
# the L=32 champion checkpoint (stride-aligned CNN transfer + scale-index
# MERA transfer). The extra coarsest MERA blocks + coarsest HCG CNN stay
# at fresh init.
#
# A/B against a fresh-init run of the same config to see whether transfer:
#   (1) reaches the same Best-200 in fewer epochs
#   (2) reaches a BETTER Best-200 (different basin)
#   (3) gets stuck / drifts (warm start doesn't help)
#
# DO NOT SUBMIT YET — waits on the champion CNN cross-L report verdict.
# If the report says "same-stride CNNs are near-identical across L", the
# transferred warm-start should give ≥25% epoch speedup.
#
# Source: L=32 champion @ ep 9500 (nearest saved to Best-200 ep 9401)
# Src HCG strides: [16, 8, 4, 2, 1] (L=32 default)

module load miniforge
source activate neuralrg
mkdir -p logs

python -u main.py \
    -L 64 -T 2.269185314213022 \
    -folder ./data/L64_T2.269_champion_from_L32 \
    -dataDriven -skipHMC \
    -epochs 20000 -batch 16 -lr 3e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1 \
    -volumePreservingWeight 1e-3 \
    -loadFromSmallerL data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64/savings/SymmMERA_l16_M3H128_R1_IsingSaving_epoch9500.saving \
    -loadFromSmallerLStrides "16,8,4,2,1" \
    -savePeriod 200 -cuda 0

date
