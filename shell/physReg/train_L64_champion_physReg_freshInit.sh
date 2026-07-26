#!/bin/bash -l
#SBATCH --job-name=L64_physReg_fresh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/L64_physReg_fresh_%j.out
#SBATCH --error=./logs/L64_physReg_fresh_%j.err

# L=64 champion architecture + physReg (χ + U₄) — FROM SCRATCH.
# Parallels the L=128 physReg design + this-file's L=32 sibling.
# Compare to plain L=64 champion (7659) for physReg-alone effect.
#
# Weights: λ_χ = 0.1, λ_U4 = 0.1 (matches L=32/L=128 physReg cells).
# bf16 enabled for ~2× speedup.

module load miniforge
source activate neuralrg
mkdir -p logs

python -u main.py \
    -L 64 -T 2.269185314213022 \
    -folder ./data/L64_T2.269_physReg_fresh_chi0.1_u40.1 \
    -dataDriven -skipHMC \
    -dataPath ./data/mcmc_data/hs_L64_T2.269185314213022_N500000.pt \
    -epochs 15000 -batch 16 -lr 3e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1 \
    -volumePreservingWeight 1e-3 \
    -physRegWeightChi 0.1 -physRegWeightU4 0.1 \
    -physRegBatch 64 \
    -savePeriod 200 -cuda 0

date
