#!/bin/bash -l
#SBATCH --job-name=L128_physReg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L128_physReg_%j.out
#SBATCH --error=./logs/L128_physReg_%j.err

# L=128 champion architecture + physical-observable regularizer (χ + U₄).
# From scratch (no warm-start). Uses moderate λ from L=32 sweep results
# (initial guess; if L=32 sweep says something better fits, update this).
#
# physReg targets auto-computed from L=128 HS data at startup.
# Compare to L128_freshInit (same arch, no physReg) for physReg effect.

module load miniforge
source activate neuralrg
mkdir -p logs

python -u main.py \
    -L 128 -T 2.269185314213022 \
    -folder ./data/L128_T2.269_champion_physReg_chi0.1_u40.1 \
    -dataDriven -skipHMC \
    -dataPath ./data/mcmc_data/hs_L128_T2.269185314213022_N100000.pt \
    -epochs 15000 -batch 8 -lr 3e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1 \
    -volumePreservingWeight 1e-3 \
    -physRegWeightChi 0.1 -physRegWeightU4 0.1 \
    -physRegBatch 8 \
    -savePeriod 200 -cuda 0

date
