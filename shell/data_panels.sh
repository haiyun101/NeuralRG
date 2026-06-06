#!/bin/bash -l
#SBATCH --job-name=data_panels
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=./logs/data_panels_%j.out
#SBATCH --error=./logs/data_panels_%j.err

# HS-dataset overview panels for each L's temp-sweep report. Runs
# three plot generations per L (configs row, P(M) overlay, G(r)
# overlay with theoretical/fit dashed lines).

module load miniforge
source activate neuralrg

mkdir -p logs

for L in 8 16 32; do
    python analyzers/criticality_fss/make_data_panels.py --L $L
done

echo "Done."
