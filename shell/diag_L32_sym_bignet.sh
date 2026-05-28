#!/bin/bash -l
#SBATCH --job-name=diag_L32_sb
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=./logs/diag_L32_sb_%j.out
#SBATCH --error=./logs/diag_L32_sb_%j.err

# Post-hoc flow diagnostic on the L=32 reverse-KL bignet anchor.
# Loads latest checkpoint, samples q ~ flow + scores p_HS data, writes
# data/32Ising_T2.269_sym_bignet/flow_diagnostic.json with KL_qp, KL_pq,
# xi_eff_q vs xi_eff_p, G(L/2)/G(0) for q and p, etc.

module load miniforge
source activate neuralrg

mkdir -p logs

python analyzers/flow_sample_diagnostic.py ./data/32Ising_T2.269_sym_bignet/ -n 8000

echo "Done."
