#!/bin/bash
#SBATCH --job-name=L32_pg_bignet
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=./logs/L32_pg_bignet_%j.out
#SBATCH --error=./logs/L32_pg_bignet_%j.err

# Pilot: reverse-KL path-gradient (STL / Vaitl 2024) at L=32 T_c, bignet arch.
# Architecture and run length match data/32Ising_T2.269_sym_bignet so the
# two trajectories are directly comparable.
#
# Hypothesis: the late-training LOSS instability (final-epoch overstates KL
# by 10-90 nat, see project_l32_late_training_instability memory) is driven
# by the explicit-θ score-function term of the standard reparam gradient.
# Dropping it via STL ("sticking the landing") should narrow the noise
# floor without changing the asymptotic minimum.
#
# Output: ./data/32Ising_T2.269_pathgrad_bignet
# Compare against: ./data/32Ising_T2.269_sym_bignet  (same arch, std reparam)

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDER="./data/32Ising_T2.269_pathgrad_bignet"
mkdir -p "$FOLDER"

echo "=========================================="
echo "L=32 reverse-KL PATH-GRADIENT bignet"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python main.py \
    -L 32 -T 2.269 \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 2000 \
    -batch 64 \
    -nlayers 16 \
    -nmlp 3 \
    -nhidden 128 \
    -nrepeat 1 \
    -savePeriod 50 \
    -symmetry \
    -skipHMC \
    -pathGrad

echo "Done."
