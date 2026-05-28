#!/bin/bash -l
#SBATCH --job-name=diag_jsL
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:30:00
#SBATCH --output=./logs/diag_jsL_%j.out
#SBATCH --error=./logs/diag_jsL_%j.err

module load miniforge
source activate neuralrg

mkdir -p logs

F=data/32Ising_T2.269_jsLoss_bignet_long_lam0.5
echo "Diagnostic for $F"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
python analyzers/flow_sample_diagnostic.py "$F" -n 8000
echo "Done."
