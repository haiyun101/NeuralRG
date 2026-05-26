#!/bin/bash
#SBATCH --job-name=nrg_L16_rkl
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=./logs/L16_rkl_%j.out
#SBATCH --error=./logs/L16_rkl_%j.err

module load miniforge
source activate neuralrg

mkdir -p logs

T=2.269
L=16
EPOCHS=10000

echo "=========================================="
echo "L=16 Reverse KL Training"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "=========================================="

# ----------------------------------------------------------
# Run 1: sym  (Z2 symmetry, comparable to sym_dataDriven)
# ----------------------------------------------------------
FOLDER="./data/16Ising_T${T}_sym"
mkdir -p $FOLDER
echo ""
echo "--- Run 1: sym (reverse KL + Z2 symmetry) ---"
python main.py \
    -L $L \
    -T $T \
    -folder $FOLDER \
    -cuda 0 \
    -epochs $EPOCHS \
    -batch 128 \
    -nlayers 10 \
    -nmlp 3 \
    -nhidden 64 \
    -nrepeat 1 \
    -savePeriod 200 \
    -symmetry \
    -skipHMC
echo "Done: sym"

# ----------------------------------------------------------
# Run 2: nsym  (no symmetry wrapper, plain reverse KL)
# ----------------------------------------------------------
FOLDER="./data/16Ising_T${T}_nsym"
mkdir -p $FOLDER
echo ""
echo "--- Run 2: nsym (reverse KL, no symmetry) ---"
python main.py \
    -L $L \
    -T $T \
    -folder $FOLDER \
    -cuda 0 \
    -epochs $EPOCHS \
    -batch 128 \
    -nlayers 10 \
    -nmlp 3 \
    -nhidden 64 \
    -nrepeat 1 \
    -savePeriod 200 \
    -skipHMC
echo "Done: nsym"

echo ""
echo "=========================================="
echo "All runs complete."
echo "=========================================="
