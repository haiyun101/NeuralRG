#!/bin/bash -l
#SBATCH --job-name=gauge_v0v3
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=./logs/gauge_v0v3_%j.out
#SBATCH --error=./logs/gauge_v0v3_%j.err

# V0/V1/V2/V2b/V3 in gauge-fixed (per-site quantile transform) coords.
# Probe input: z ~ N(0,I) at shape (N, 1, 2, 2). Each block's output is
# gauge-fixed before adjacency MSE / identity residual.
#
# Output: analyzers/csv/rg_v0_v3_gauge.csv

module load miniforge
source activate neuralrg

mkdir -p logs

python -u analyzers/rg_fixed_point/gauge_probes_v0_v3.py --N 10000

echo "Done."
