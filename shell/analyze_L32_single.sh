#!/bin/bash
#SBATCH --job-name=L32_diag_one
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=./logs/L32_diag_%j.out
#SBATCH --error=./logs/L32_diag_%j.err

# Single-folder L=32 flow-sample diagnostic. Usage:
#   FOLDER=data/32Ising_T2.269_<run>  sbatch shell/analyze_L32_single.sh
# Parallel companion to shell/analyze_L64_single.sh. L=32 needs ~30 min
# per folder so 2h wall is generous. N=8000 (matches the L=32 historical
# diag jobs; double the L=64 default because L=32 has 4x less per-sample
# memory cost).

module load miniforge
source activate neuralrg

mkdir -p logs

: "${FOLDER:?env FOLDER=<path-to-folder> required}"

if [ ! -d "$FOLDER" ]; then
    echo "ERROR: folder not found: $FOLDER"
    exit 1
fi

echo "=========================================="
echo "L=32 single-folder diagnostic  |  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "FOLDER: $FOLDER"
echo "=========================================="

python -u analyzers/flow_sample_diagnostic.py "$FOLDER" -n 8000 -b 512

echo "Done."
