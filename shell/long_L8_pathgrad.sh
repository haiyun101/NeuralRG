#!/bin/bash
#SBATCH --job-name=pg_L8_long
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=./logs/pg_L8_long_%j.out
#SBATCH --error=./logs/pg_L8_long_%j.err

# Extended L=8 path-gradient comparison: 5000 epochs each, matched flags.
# Replaces the 500-epoch sanity (which was too short to test STL's
# late-noise-floor benefit -- "sticking the landing" only kicks in once
# q is close to p, well past the warmup phase).
#
# Reference: the well-converged sym at L=8 needed ~4200 epochs to reach
# KL<0.5 and ~5100 to reach KL<0.4. 5000 epochs puts us inside that
# regime where the STL/baseline noise comparison is meaningful.

module load miniforge
source activate neuralrg

mkdir -p logs

T=2.269
L=8
EPOCHS=5000

echo "=========================================="
echo "L=8 Path-Gradient LONG comparison (5000 epochs each)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "=========================================="

# Run 1: baseline (standard reparametrization)
FOLDER="./data/8Ising_T${T}_long5k_baseline"
mkdir -p $FOLDER
echo ""
echo "--- Run 1/2: baseline (standard reparam) ---"
python -u main.py \
    -L $L -T $T \
    -folder $FOLDER \
    -cuda 0 \
    -epochs $EPOCHS \
    -batch 128 \
    -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 \
    -savePeriod 100 \
    -symmetry -skipHMC
echo "Done: baseline"

# Run 2: path-gradient (STL)
FOLDER="./data/8Ising_T${T}_long5k_pathgrad"
mkdir -p $FOLDER
echo ""
echo "--- Run 2/2: pathGrad (STL) ---"
python -u main.py \
    -L $L -T $T \
    -folder $FOLDER \
    -cuda 0 \
    -epochs $EPOCHS \
    -batch 128 \
    -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 \
    -savePeriod 100 \
    -symmetry -skipHMC \
    -pathGrad
echo "Done: pathGrad"

echo ""
echo "=========================================="
echo "Comparison ready. Records:"
echo "  baseline: ./data/8Ising_T${T}_long5k_baseline/records/"
echo "  pathGrad: ./data/8Ising_T${T}_long5k_pathgrad/records/"
echo "=========================================="
