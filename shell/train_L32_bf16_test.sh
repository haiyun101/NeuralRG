#!/bin/bash -l
#SBATCH --job-name=L32_bf16_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=./logs/L32_bf16_test_%j.out
#SBATCH --error=./logs/L32_bf16_test_%j.err

# bf16 mixed-precision test: continue L=32 champion for 500 epochs with
# bf16 autocast enabled. Compare:
#   - wall time (expect ~2× speedup vs fp32)
#   - loss trajectory stability
#   - Best-200 at end (should be comparable to plain champion at same epoch)

module load miniforge
source activate neuralrg
mkdir -p logs

# Warm-start from champion via same-L self-transfer (no -load, so training
# starts at epoch 0 but weights are champion's — comparable to a fresh
# 500-epoch continuation with bf16).
python -u main.py \
    -L 32 -T 2.269185314213022 \
    -folder ./data/L32_T2.269_bf16_test \
    -dataDriven -skipHMC \
    -dataPath ./data/mcmc_data/hs_L32_T2.269185314213022_N200000.pt \
    -epochs 500 -batch 64 -lr 3e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1 \
    -volumePreservingWeight 1e-3 \
    -loadFromSmallerL data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64/savings/SymmMERA_l16_M3H128_R1_IsingSaving_epoch9500.saving \
    -loadFromSmallerLStrides "16,8,4,2,1" \
    -bf16 \
    -savePeriod 100 -cuda 0

date
