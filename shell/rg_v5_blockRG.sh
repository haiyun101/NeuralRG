#!/bin/bash -l
#SBATCH --job-name=rg_v5
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:45:00
#SBATCH --output=./logs/rg_v5_%j.out
#SBATCH --error=./logs/rg_v5_%j.err

# V5 block-RG diagnostic: Wilson-Kadanoff block-average cascade on HS
# data, compared to MERA's forward intermediate y_s (sub-sampled to
# matching resolution) at each scale. This is the ground-truth test
# V4 was missing -- "does the MERA path through scales match real-space
# block RG?".

module load miniforge
source activate neuralrg

mkdir -p logs

python analyzers/rg_fixed_point/rg_v5_blockRG_compare.py --N 2000

echo "Done."
