#!/bin/bash -l
#SBATCH --job-name=qplot_pg
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=./logs/qplot_pg_%j.out
#SBATCH --error=./logs/qplot_pg_%j.err

module load miniforge
source activate neuralrg

mkdir -p logs

python analyzers/flow_sample_diagnostic.py \
    ./data/32Ising_T2.269_pathgrad_bignet_long_ext/ \
    -n 1500 -b 256 --no-json

echo "Done."
