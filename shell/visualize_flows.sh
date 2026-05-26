#!/bin/bash
#SBATCH --job-name=nrg_flow_vis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=./logs/flow_vis_%j.out
#SBATCH --error=./logs/flow_vis_%j.err

# Render <folder>/flow_samples.png for trained runs: a side-by-side of the
# ACTUAL flow output (x ~ q, sampled from the latest checkpoint) next to the
# HS target data (x ~ p).  This is what proposals_*.png is NOT in a data-driven
# run -- there proposals_*.png shows the input data, not the model.
#
# --no-json: only refresh the PNG; do not overwrite the 8000-sample
# flow_diagnostic.json stats with these cheap 100-sample numbers.

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDERS=(
    data/8Ising_T2.269_hs_dataDriven
    data/16Ising_T2.269_hs_dataDriven
    data/32Ising_T2.269_hs_dataDriven
    data/8Ising_T2.269_sym
    data/16Ising_T2.269_sym
    data/32Ising_T2.269_sym
)

echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
python analyzers/flow_sample_diagnostic.py "${FOLDERS[@]}" -n 100 -b 100 --no-json
echo "Done. flow_samples.png written in each folder above."
