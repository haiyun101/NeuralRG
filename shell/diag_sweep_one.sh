#!/bin/bash -l
#SBATCH --job-name=diag_sweep
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=./logs/diag_sweep_%x_%j.out
#SBATCH --error=./logs/diag_sweep_%x_%j.err

# Flow-diagnostic on one (L, T) HS forward-KL run.
# Loads latest checkpoint, samples q ~ flow + scores HS data, writes
#   data/${L}Ising_T${T}_hs_dataDriven/flow_diagnostic.json
#   data/${L}Ising_T${T}_hs_dataDriven/flow_samples.png
#   data/${L}Ising_T${T}_hs_dataDriven/flow_correlations.png

module load miniforge
source activate neuralrg

mkdir -p logs

L="$1"
T="$2"
N=8000

FOLDER="./data/${L}Ising_T${T}_hs_dataDriven"
if [ ! -d "$FOLDER/savings" ]; then
    echo "ERROR: no savings under $FOLDER"
    exit 1
fi

echo "=========================================="
echo "Diagnostic  |  L=${L}  T=${T}  N=${N}"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python analyzers/flow_sample_diagnostic.py "$FOLDER" -n "$N"

echo "Done (L=${L}, T=${T})."
