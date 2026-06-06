#!/bin/bash -l
#SBATCH --job-name=crit_flow
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --output=./logs/crit_flow_%j.out
#SBATCH --error=./logs/crit_flow_%j.err

# Universality test on flow samples: compute Binder, susceptibility
# (chi/L^(gamma/nu)), xi_eff/L, and P(M)*L^(beta/nu) collapse on
# samples drawn from each trained hs_dataDriven flow (3 L x 5 T).
# Compare side-by-side against criticality_summary.csv (data).

module load miniforge
source activate neuralrg

mkdir -p logs

python analyzers/criticality_fss/criticality_flow.py --N 8000

echo "Done."
