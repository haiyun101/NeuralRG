#!/bin/bash -l
#SBATCH --job-name=mcmc_L64_multiT
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/mcmc_L64_multiT_%j.out
#SBATCH --error=./logs/mcmc_L64_multiT_%j.err

# Generate L=64 HS ground-truth data at 4 non-T_c temperatures via Numba
# Wolff. Enables multi-T tier1 comparisons + fixdil+VP champion sweep
# across the transition.

module load miniforge
source activate neuralrg

mkdir -p logs data/mcmc_data

for T in 2.15 2.22 2.32 2.4; do
    echo "==================================="
    echo "L=64 T=$T  N=200000"
    echo "==================================="
    python -u generate_mcmc_data.py -L 64 -T $T -N 200000
    echo
done

echo "Done."
