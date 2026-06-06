#!/bin/bash -l
#SBATCH --job-name=criticality
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=./logs/criticality_%j.out
#SBATCH --error=./logs/criticality_%j.err

# Criticality witnesses computed directly from the HS dataset:
# Binder cumulant, susceptibility FSS, xi_eff/L, magnetisation
# collapse, and rescaled P(M) at T_c. CPU-only (numpy + matplotlib);
# torch is only used to load the .pt sample tensors.

module load miniforge
source activate neuralrg

mkdir -p logs

python analyzers/criticality_fss/criticality_analysis.py --N 8000

echo "Done."
