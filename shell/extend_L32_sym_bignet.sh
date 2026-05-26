#!/bin/bash
#SBATCH --job-name=L32_sb_ext
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=./logs/L32_sb_ext_%j.out
#SBATCH --error=./logs/L32_sb_ext_%j.err

# Phase-2 extension of the reverse-KL L=32 bignet anchor.
# Resumes from data/32Ising_T2.269_sym_bignet/savings/<latest>.saving and
# trains another 6000 epochs (parameters.hdf5 has been patched to epochs=6000).
# Phase-1 records/savings preserved in records_phase1/, savings_phase1/.

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDER="./data/32Ising_T2.269_sym_bignet"
echo "=========================================="
echo "L=32 reverse-KL bignet  EXTENSION (phase 2)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python main.py -load -folder "$FOLDER" -cuda 0 -symmetry -skipHMC

echo "Done (extension)."
