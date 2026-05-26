#!/bin/bash
#SBATCH --job-name=nrg_L8_dd
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=./logs/L8_dd_%j.out
#SBATCH --error=./logs/L8_dd_%j.err

module load miniforge
source activate neuralrg

mkdir -p logs
mkdir -p data/8Ising_T2.269_sym_dataDriven

T=2.269
L=8
N_MCMC=200000
FOLDER="./data/8Ising_T2.269_sym_dataDriven"

echo "=========================================="
echo "L=8 Data-Driven Expressiveness Test"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "=========================================="

# ----------------------------------------------------------
# Phase 1: Compute exact L=8 partition function (transfer
# matrix, ~1s on CPU — done here so it runs on compute node)
# ----------------------------------------------------------
echo ""
echo "Phase 1: Computing exact L=8 free energies..."
python compute_exact_L8.py
echo "Done. L=8 values appended to etc/exactz.md."

# ----------------------------------------------------------
# Phase 2: Generate MCMC data (Wolff, Numba JIT)
# ----------------------------------------------------------
MCMC_PATH="./data/mcmc_data/mcmc_wolff_L${L}_T${T}_N${N_MCMC}.pt"

if [ -f "$MCMC_PATH" ]; then
    echo ""
    echo "Phase 2: MCMC data already exists at $MCMC_PATH, skipping."
else
    echo ""
    echo "Phase 2: Generating $N_MCMC MCMC samples (L=$L, T=$T)..."
    python generate_mcmc_data.py -L $L -T $T -N $N_MCMC
    echo "Done."
fi

# ----------------------------------------------------------
# Phase 3: Data-driven training
# ----------------------------------------------------------
echo ""
echo "Phase 3: Starting data-driven training (forward KL)..."

python main.py \
    -L $L \
    -T $T \
    -folder $FOLDER \
    -dataDriven \
    -cuda 0 \
    -epochs 5000 \
    -batch 128 \
    -nlayers 10 \
    -nmlp 3 \
    -nhidden 64 \
    -nrepeat 1 \
    -savePeriod 100 \
    -symmetry \
    -skipHMC

echo ""
echo "=========================================="
echo "All phases complete."
echo "=========================================="
