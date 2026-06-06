#!/bin/bash -l
#SBATCH --job-name=diag_L8_pg
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=./logs/diag_L8_pg_%j.out
#SBATCH --error=./logs/diag_L8_pg_%j.err

# Post-hoc flow diagnostic on the new 9800-ep L=8 STL run. Writes
# flow_diagnostic.json (with HS-data-side KL_qp, KL_pq, mag_abs_p,
# xi_p), flow_samples.png, and the log-log flow_correlations.png
# into data/8Ising_T2.269_long9800_pathgrad/.
# Reads HS dataset from data/mcmc_data/hs_L8_T2.269185314213022_N200000.pt.

module load miniforge
source activate neuralrg

mkdir -p logs

python analyzers/flow_sample_diagnostic.py \
    ./data/8Ising_T2.269_long9800_pathgrad/ \
    -n 8000

echo "Done."
