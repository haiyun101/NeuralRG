#!/bin/bash -l
#SBATCH --job-name=rg_v4
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:45:00
#SBATCH --output=./logs/rg_v4_%j.out
#SBATCH --error=./logs/rg_v4_%j.err

# V4 RG fixed-point probe: feed real HS data forward through each
# trained MERA flow's scale-blocks, recording the field after each
# physical scale. Compare adjacent intermediate fields via marginal
# KS / Wasserstein-1 distance and G(r)/G(0) shape mismatch.

module load miniforge
source activate neuralrg

mkdir -p logs

python analyzers/rg_fixed_point/rg_fixed_point_v4_dataforward.py --N 2000

echo "Done."
