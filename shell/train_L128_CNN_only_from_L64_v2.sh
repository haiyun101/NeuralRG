#!/bin/bash -l
#SBATCH --job-name=L128_CNNonly_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/L128_CNNonly_v2_%j.out
#SBATCH --error=./logs/L128_CNNonly_v2_%j.err

# ABLATION 2 v2: CNN-only transfer with stability fixes (Fix 3).
# Original (1806653) diverged to F=1.1M — fresh MERA + trained CNN
# creates instability.
# Fixes:
#   -gradClip 1.0: clip per-step gradient norm (was absent)
#   -lr 1e-4:      was 3e-4, gentler updates during warmup
# If v2 succeeds → ablation valid → we can compare CNN-only vs MERA-only
# vs both to attribute physics importance.

module load miniforge
source activate neuralrg
cd /cluster/home/hhuang05/NeuralRG
mkdir -p logs

python -u main.py \
    -L 128 -T 2.269185314213022 \
    -folder ./data/L128_T2.269_CNNonly_from_L64_v2 \
    -dataDriven -skipHMC \
    -dataPath ./data/mcmc_data/hs_L128_T2.269185314213022_N100000.pt \
    -epochs 15000 -batch 8 -lr 1e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry -alpha 0 \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1 \
    -volumePreservingWeight 1e-3 \
    -loadFromSmallerL data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16/savings/SymmMERA_l16_M3H128_R1_IsingSaving_epoch9500.saving \
    -loadFromSmallerLStrides "32,16,8,4,2,1" \
    -loadFromSmallerLComponents cnn \
    -gradClip 1.0 \
    -savePeriod 200 -cuda 0

date
