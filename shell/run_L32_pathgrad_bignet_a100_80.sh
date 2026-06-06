#!/bin/bash
#SBATCH --job-name=L32_pg_b128
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=./logs/L32_pg_b128_%j.out
#SBATCH --error=./logs/L32_pg_b128_%j.err

# Full-batch (b=128) reverse-KL path-gradient pilot at L=32 T_c.
# Runs in parallel with shell/run_L32_pathgrad_bignet.sh (b=64 on L40).
# Targets 80GB A100 (s1cmp006/007) so the double-forward STL fits.
#
# Purpose: disentangle "STL is broken" from "b=64 is too small to converge."
#   - If this b=128 STL also stalls at L >> baseline -> implementation/STL issue.
#   - If this b=128 STL converges to baseline F -> b=64 was the problem.
#
# Companion baseline: data/32Ising_T2.269_sym_bignet (b=128, std reparam, L40).
# Output: ./data/32Ising_T2.269_pathgrad_bignet_b128
#
# Caveat: preempt + --requeue can drop Adam moments on requeue
# (see project_resume_optimizer_state memory). 80GB A100s on s1cmp006/007
# are currently idle, so preemption risk is low for the first hour.

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDER="./data/32Ising_T2.269_pathgrad_bignet_b128"
mkdir -p "$FOLDER"

echo "=========================================="
echo "L=32 reverse-KL PATH-GRADIENT bignet (b=128, A100-80G)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python -u main.py \
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
    -skipHMC \
    -pathGrad

echo "Done."
