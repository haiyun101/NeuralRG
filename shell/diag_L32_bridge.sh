#!/bin/bash -l
#SBATCH --job-name=diag_L32_br
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:30:00
#SBATCH --output=./logs/diag_L32_br_%j.out
#SBATCH --error=./logs/diag_L32_br_%j.err

# Post-hoc flow diagnostic on the bridge-upweighted L=32 hs_bignet run.
# Writes flow_diagnostic.json with model-side KL_qp, target-side KL_pq,
# mag_abs_q, G(L/2)/G(0), xi_q for direct comparison to baseline hs_bignet.
# Folder is parameterized via env so multiple W/T sweeps can reuse.

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDER="${FOLDER:-./data/32Ising_T2.269_hsBignet_bridge_w5.0t0.5/}"
python analyzers/flow_sample_diagnostic.py "$FOLDER" -n 8000 -b 256

echo "Done."
