#!/bin/bash -l
#SBATCH --job-name=rg_robust
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=./logs/rg_robust_%j.out
#SBATCH --error=./logs/rg_robust_%j.err

# Robustness checks for the RG fixed-point probe -- 3 variants:
#   V1: global vs per-position z-score
#   V2: chain-input (production composition) vs same-input probe
#   V3: identity-residual per scale-block (is each f_s near-identity?)
# CPU job, ~ a minute per flow.

module load miniforge
source activate neuralrg

mkdir -p logs

python analyzers/rg_fixed_point/rg_fixed_point_robustness.py --N 10000

echo "Done."
