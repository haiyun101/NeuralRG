#!/bin/bash -l
#SBATCH --job-name=L32_physReg_fresh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L32_physReg_fresh_%j.out
#SBATCH --error=./logs/L32_physReg_fresh_%j.err

# L=32 champion architecture + physReg (χ + U₄) — FROM SCRATCH.
# Parallels the L=128 physReg design (fresh init).
# Together with the warm-start L=32 physReg sweep (1797768), lets us
# separate "physReg improves converged champion" vs "physReg helps
# from-scratch training".
#
# Weights: λ_χ = 0.1, λ_U4 = 0.1 (matches L=64/L=128 physReg configs).
# bf16 enabled for ~2× speedup.

module load miniforge
source activate neuralrg
mkdir -p logs

python -u main.py \
    -L 32 -T 2.269185314213022 \
    -folder ./data/L32_T2.269_physReg_fresh_chi0.1_u40.1 \
    -dataDriven -skipHMC \
    -dataPath ./data/mcmc_data/hs_L32_T2.269185314213022_N200000.pt \
    -epochs 15000 -batch 64 -lr 3e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1 \
    -volumePreservingWeight 1e-3 \
    -physRegWeightChi 0.1 -physRegWeightU4 0.1 \
    -physRegBatch 128 \
    -savePeriod 200 -cuda 0

date
