#!/bin/bash
#SBATCH --job-name=mcmc_L32
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/mcmc_L32_%j.out
#SBATCH --error=./logs/mcmc_L32_%j.err

# Generate 200k Wolff-cluster MCMC samples for L=32 at T=2.269,
# to match the 200k sample count used at L=8 and L=16.
# Wolff is a Numba-JIT CPU algorithm (no GPU path).
# Output: ./data/mcmc_data/mcmc_wolff_L32_T2.269_N200000.pt

module load miniforge
source activate neuralrg

mkdir -p logs

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
python generate_mcmc_data.py -L 32 -T 2.269 -N 200000
echo "MCMC generation done."
