#!/bin/bash
#SBATCH --job-name=nrg_sweep_data
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/sweep_data_%x_%j.out
#SBATCH --error=./logs/sweep_data_%x_%j.err

# Generate MCMC + HS continuous samples for one (L, T) pair.
# Usage: sbatch --job-name=data_L{L}_T{T} shell/sweep_temp_data.sh <L> <T>
# Skips phases whose output file already exists, so it is safe to re-submit.

module load miniforge
source activate neuralrg

mkdir -p logs data/mcmc_data

L="$1"
T="$2"
N=200000

if [ -z "$L" ] || [ -z "$T" ]; then
    echo "ERROR: usage  sbatch shell/sweep_temp_data.sh <L> <T>"
    exit 1
fi

MCMC_PT="data/mcmc_data/mcmc_wolff_L${L}_T${T}_N${N}.pt"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"

echo "=========================================="
echo "Data gen  |  L=${L}  T=${T}  N=${N}"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="

echo ""
echo "--- Phase 1: Wolff MCMC ---"
if [ -f "$MCMC_PT" ]; then
    echo "MCMC samples already exist: $MCMC_PT -- skip."
else
    python generate_mcmc_data.py -L "$L" -T "$T" -N "$N"
fi

echo ""
echo "--- Phase 2: HS continuous conversion ---"
if [ -f "$HS_PT" ]; then
    echo "HS samples already exist: $HS_PT -- skip."
else
    python generate_hs_samples.py -L "$L" -T "$T" \
        --in_path "$MCMC_PT" \
        --out_dir data/mcmc_data
fi

echo ""
echo "Done (L=${L}, T=${T})."
