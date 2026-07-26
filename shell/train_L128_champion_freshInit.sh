#!/bin/bash -l
#SBATCH --job-name=L128_freshInit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L128_freshInit_%j.out
#SBATCH --error=./logs/L128_freshInit_%j.err

# L=128 champion architecture — FROM SCRATCH (no warm-start).
# Serves as the baseline for L=128 experiments:
#   - vs L128_from_L64 (forward-KL warm-start from L=64 champion, job 1796153)
#   - vs L128_revKL_from_L64 (reverse-KL warm-start, job 1796176)
#   - vs L128_physReg (forward-KL + χ+U4 reg, another job)
#
# Same arch as L=32/L=64 champion (fixdil+VP-1e-3 nr=1). Waits on L=128
# HS data generation (job 1796152).
#
# batch=8 (matches other L=128 jobs due to memory).

module load miniforge
source activate neuralrg
mkdir -p logs

python -u main.py \
    -L 128 -T 2.269185314213022 \
    -folder ./data/L128_T2.269_champion_freshInit \
    -dataDriven -skipHMC \
    -dataPath ./data/mcmc_data/hs_L128_T2.269185314213022_N100000.pt \
    -epochs 15000 -batch 8 -lr 3e-4 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -symmetry \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1 \
    -volumePreservingWeight 1e-3 \
    -savePeriod 200 -cuda 0

date
