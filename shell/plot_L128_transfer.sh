#!/bin/bash -l
#SBATCH --job-name=plot_L128_xfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=./logs/plot_L128_xfer_%j.out
#SBATCH --error=./logs/plot_L128_xfer_%j.err

# Generate configuration, G(r), M-distribution, and observable-bar plots
# for L=64→L=128 transfer report.

module load miniforge
source activate neuralrg
mkdir -p logs figures/L128_transfer

python -u analyzers/rg_fixed_point/plot_L128_transfer.py \
    --N 1000 --batch 32 --device cuda \
    --out figures/L128_transfer

date
