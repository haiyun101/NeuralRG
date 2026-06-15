#!/bin/bash
#SBATCH --job-name=L64_diag_one
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=./logs/L64_diag_%j.out
#SBATCH --error=./logs/L64_diag_%j.err

# Single-folder L=64 flow-sample diagnostic. Usage:
#   FOLDER=data/64Ising_T2.269_<run>  sbatch shell/analyze_L64_single.sh
# Each L=64 bignet folder takes ~3+ hours; running them one-per-sbatch
# parallelises across A100s instead of serializing (the original
# shell/analyze_L64.sh hit a 4h wall after one folder).

module load miniforge
source activate neuralrg

mkdir -p logs

: "${FOLDER:?env FOLDER=<path-to-folder> required}"

if [ ! -d "$FOLDER" ]; then
    echo "ERROR: folder not found: $FOLDER"
    exit 1
fi

echo "=========================================="
echo "L=64 single-folder diagnostic  |  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "FOLDER: $FOLDER"
echo "=========================================="

python -u analyzers/flow_sample_diagnostic.py "$FOLDER" -n 4000 -b 128

echo "Done."
