#!/bin/bash
#SBATCH --job-name=nrg_L16_dd
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=./logs/L16_dd_%j.out
#SBATCH --error=./logs/L16_dd_%j.err

module load miniforge
source activate neuralrg

mkdir -p logs
mkdir -p data/16Ising_T2.269_sym_dataDriven

T=2.269
L=16
N_MCMC=200000
FOLDER="./data/16Ising_T2.269_sym_dataDriven"

echo "=========================================="
echo "L=16 Data-Driven Expressiveness Test"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "=========================================="

# ----------------------------------------------------------
# Phase 1: Generate MCMC data (Wolff, Numba JIT)
# Exact L=16 values already exist in etc/exactz.md.
# ----------------------------------------------------------
MCMC_PATH="./data/mcmc_data/mcmc_wolff_L${L}_T${T}_N${N_MCMC}.pt"

if [ -f "$MCMC_PATH" ]; then
    echo ""
    echo "Phase 1: MCMC data already exists at $MCMC_PATH, skipping."
else
    echo ""
    echo "Phase 1: Generating $N_MCMC MCMC samples (L=$L, T=$T)..."
    python generate_mcmc_data.py -L $L -T $T -N $N_MCMC
    echo "Done."
fi

# ----------------------------------------------------------
# Phase 2: Data-driven training
# ----------------------------------------------------------
echo ""
echo "Phase 2: Starting data-driven training (forward KL)..."

python main.py \
    -L $L \
    -T $T \
    -folder $FOLDER \
    -dataDriven \
    -cuda 0 \
    -epochs 10000 \
    -batch 128 \
    -nlayers 10 \
    -nmlp 3 \
    -nhidden 64 \
    -nrepeat 1 \
    -savePeriod 200 \
    -symmetry \
    -skipHMC

echo ""
echo "=========================================="
echo "All phases complete."
echo "=========================================="
