#!/bin/bash -l
#SBATCH --job-name=gauge_mmse
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=./logs/gauge_mmse_%j.out
#SBATCH --error=./logs/gauge_mmse_%j.err

# Matched-pair gauge MSE on 4 main flows (hs_bignet, sym_bignet, T=2.15, T=2.40).
# Output: analyzers/csv/rg_v5_gauge_matched_mse.csv

module load miniforge
source activate neuralrg

mkdir -p logs

python -u analyzers/rg_fixed_point/gauge_v5_matched_mse.py --N 2000

echo "Done."
