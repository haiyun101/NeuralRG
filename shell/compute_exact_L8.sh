#!/bin/bash
#SBATCH --job-name=exact_L8
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:10:00
#SBATCH --output=./logs/exact_L8_%j.out
#SBATCH --error=./logs/exact_L8_%j.err

module load miniforge
source activate neuralrg

mkdir -p logs

echo "Computing exact L=8 partition function with HS normalization..."
python compute_exact_L8.py

echo "Done."
