#!/bin/bash -l
#SBATCH --job-name=gauge_v5
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=./logs/gauge_v5_%j.out
#SBATCH --error=./logs/gauge_v5_%j.err

# Gauge-fixed V5: block-RG cross-comparison with per-site Gaussianized
# fields. Requires gauge_transforms.pt at every folder in the V5 FOLDERS
# dict (skipped if missing). After per-site gauge-fix:
#   - KS/W1 trivially ~0 (no marginal info left)
#   - RMS-G measures pure spatial structure mismatch
# Output: analyzers/csv/rg_v5_gauge_compare.csv

module load miniforge
source activate neuralrg

mkdir -p logs

python -u analyzers/rg_fixed_point/gauge_v5_compare.py --N 2000

echo "Done."
