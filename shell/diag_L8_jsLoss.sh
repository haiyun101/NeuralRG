#!/bin/bash -l
#SBATCH --job-name=diag_L8_js
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=./logs/diag_L8_js_%j.out
#SBATCH --error=./logs/diag_L8_js_%j.err

# Post-hoc flow diagnostic on the L=8 jsLoss anchor.
# Both KL(q||p) and KL(p||q) are training-objective for JS, so the diagnostic
# row gives the honest fresh-sample KLs (not training-batch). Compare to the
# pure-mode L=8 diagnostics:
#   sym             on-obj KL_qp=0.71  off-obj KL_pq=2.00
#   hs_dataDriven   on-obj KL_pq=1.35  off-obj KL_qp=0.96
#   THIS RUN        both fresh-sample => are both <2 nat?

module load miniforge
source activate neuralrg

mkdir -p logs

python analyzers/flow_sample_diagnostic.py ./data/8Ising_T2.269_jsLoss_lam0.5/ -n 8000

echo "Done."
