#!/bin/bash -l
#SBATCH --job-name=L64_pg_bignet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L64_pg_%j.out
#SBATCH --error=./logs/L64_pg_%j.err

# L=64 T_c reverse-KL via STL path gradient, bignet (nlayers=16, nhidden=128).
# No data needed (samples drawn from the flow during training).
# MERA depth at L=64 is log2(64)*2 = 12 (vs 10 at L=32).
# batch=64 (down from 128 at L=32) for safe activation memory on A100-40G:
# 4x more pixels per sample => ~4x activation memory at fixed batch.
# Time budget: aim for ~10000-15000 epochs in 24h; bump arch only if KL
# refuses to drop past the first few thousand epochs.

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T=2.269185314213022
FOLDER="./data/${L}Ising_T2.269_pathgrad_bignet"
mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L STL path-gradient bignet (b=64, A100)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python -u main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 20000 \
    -batch 8  \
    -nlayers 16 \
    -nmlp 3 \
    -nhidden 128 \
    -nrepeat 1 \
    -savePeriod 100 \
    -symmetry \
    -skipHMC \
    -pathGrad

echo "Done."
