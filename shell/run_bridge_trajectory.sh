#!/bin/bash -l
#SBATCH --job-name=bridge_traj
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:45:00
#SBATCH --output=./logs/bridge_traj_%j.out
#SBATCH --error=./logs/bridge_traj_%j.err

# Iterate every .saving checkpoint in <folder>, sample 2000 from each,
# and write a per-epoch trajectory of <|M|>, Var(M), and bridge fraction.
# Use to verify Phase 2 reverse-KL finetune did NOT collapse the bridge.
#
# Usage: sbatch shell/run_bridge_trajectory.sh <run_folder>

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDER="${1:-./data/32Ising_T2.269_phase2_finetune}"
echo "=========================================="
echo "Bridge trajectory  |  folder: $FOLDER"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="

python analyzers/bridge_trajectory.py "$FOLDER" -n ${N_SAMPLES:-2000} ${FINAL_EPOCH:+--final-epoch $FINAL_EPOCH}

echo "Done."
