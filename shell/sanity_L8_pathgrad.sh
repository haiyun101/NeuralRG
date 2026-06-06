#!/bin/bash
#SBATCH --job-name=pg_L8_sane
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:45:00
#SBATCH --output=./logs/pg_L8_sane_%j.out
#SBATCH --error=./logs/pg_L8_sane_%j.err

# Path-gradient pilot sanity check at L=8.
# Runs reverse-KL twice (matched flags, only -pathGrad differs).
# Expected outcome:
#   - both runs converge to similar best-LOSS (both estimators unbiased)
#   - pathGrad run has lower LOSS noise in the late phase ("sticking the landing")
# If the pathGrad run NaNs or diverges, the implementation is wrong.

module load miniforge
source activate neuralrg

mkdir -p logs

T=2.269
L=8
EPOCHS=500

echo "=========================================="
echo "L=8 Path-Gradient Sanity Check"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "=========================================="

# Baseline: standard reparametrization gradient
FOLDER="./data/8Ising_T${T}_sane_baseline"
mkdir -p $FOLDER
echo ""
echo "--- Run 1/2: baseline (standard reparam) ---"
python main.py \
    -L $L -T $T \
    -folder $FOLDER \
    -cuda 0 \
    -epochs $EPOCHS \
    -batch 128 \
    -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 \
    -savePeriod 50 \
    -symmetry -skipHMC
echo "Done: baseline"

# Path-gradient (STL)
FOLDER="./data/8Ising_T${T}_sane_pathgrad"
mkdir -p $FOLDER
echo ""
echo "--- Run 2/2: pathGrad (STL) ---"
python main.py \
    -L $L -T $T \
    -folder $FOLDER \
    -cuda 0 \
    -epochs $EPOCHS \
    -batch 128 \
    -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 \
    -savePeriod 50 \
    -symmetry -skipHMC \
    -pathGrad
echo "Done: pathGrad"

echo ""
echo "=========================================="
echo "All runs complete. Compare:"
echo "  baseline: ./data/8Ising_T${T}_sane_baseline/records/"
echo "  pathGrad: ./data/8Ising_T${T}_sane_pathgrad/records/"
echo "=========================================="
