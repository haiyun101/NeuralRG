#!/bin/bash
#SBATCH --job-name=L32_sym_bignet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=./logs/L32_sym_bignet_%j.out
#SBATCH --error=./logs/L32_sym_bignet_%j.err

# Reverse-KL (energy-based) training at L=32, T=2.269 with the bignet
# architecture (nlayers=16, nhidden=128). Companion to the forward-KL
# hs_bignet result -- tests whether the L=32 default-arch reverse-KL gap
# (sym_longer training KL(q||p) ~ 11.96 nat) is also a capacity issue.
#
# Comparison points:
#   data/32Ising_T2.269_sym         (1000 epochs, nlayers=10 nhidden=64)
#   data/32Ising_T2.269_sym_longer  (1600 epochs, nlayers=10 nhidden=64)
#   this run -> data/32Ising_T2.269_sym_bignet  (2000 epochs, bignet)

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDER="./data/32Ising_T2.269_sym_bignet"
mkdir -p "$FOLDER"

echo "=========================================="
echo "L=32 reverse-KL bignet  |  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python main.py \
    -L 32 -T 2.269 \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 2000 \
    -batch 128 \
    -nlayers 16 \
    -nmlp 3 \
    -nhidden 128 \
    -nrepeat 1 \
    -savePeriod 50 \
    -symmetry \
    -skipHMC

echo "Done."
