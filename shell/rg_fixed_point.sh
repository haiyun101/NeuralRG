#!/bin/bash -l
#SBATCH --job-name=rg_probe
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=./logs/rg_probe_%j.out
#SBATCH --error=./logs/rg_probe_%j.err

# RG fixed-point probe: extract each scale-block from the trained
# MERA flow, apply it to a standardised Gaussian probe batch, and
# measure how the (z-scored) outputs change between adjacent scales.
# If the T_c flow is at an RG fixed point, deep scale-blocks should
# act on the probe similarly -> adjacent-MSE plateaus to a low value.

module load miniforge
source activate neuralrg

mkdir -p logs

python analyzers/rg_fixed_point.py --N 10000

echo "Done."
