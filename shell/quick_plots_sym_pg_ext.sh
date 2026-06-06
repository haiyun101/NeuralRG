#!/bin/bash -l
#SBATCH --job-name=qplot_ext
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=./logs/qplot_ext_%j.out
#SBATCH --error=./logs/qplot_ext_%j.err

# Fast plot-only generation of flow_samples.png + flow_correlations.png
# for the two new methods. Uses --no-json so the running full diag jobs
# (39408285, 39408286) can finalise the JSON later without conflict.
# Small N=1500 keeps it under 5 min while still giving reasonable
# config grids and per-config-magnetisation / G(r) panels.

module load miniforge
source activate neuralrg

mkdir -p logs

python analyzers/flow_sample_diagnostic.py \
    ./data/32Ising_T2.269_sym_bignet_ext/ \
    ./data/32Ising_T2.269_pathgrad_bignet_long_ext/ \
    -n 1500 -b 256 --no-json

echo "Done."
