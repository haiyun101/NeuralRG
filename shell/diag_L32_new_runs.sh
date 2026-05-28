#!/bin/bash -l
#SBATCH --job-name=diag_L32new
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:30:00
#SBATCH --output=./logs/diag_L32new_%j.out
#SBATCH --error=./logs/diag_L32new_%j.err

# Generate flow_samples.png + flow_correlations.png + flow_diagnostic.json
# for the three new L=32 runs added to the concise report:
#   * sym_bignet              (reverse-KL bignet, new best reverse-KL)
#   * jsLoss_bignet_long      (symmetric JS, both directions balanced)
#   * phase2_finetune         (fwd->rev workflow)
#
# N=8000 samples gives KL stats accurate to ~0.1 nat.

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDERS=(
    data/32Ising_T2.269_sym_bignet
    data/32Ising_T2.269_jsLoss_bignet_long_lam0.5
    data/32Ising_T2.269_phase2_finetune
)

echo "=========================================="
echo "Diagnostic for L=32 new runs"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="

for F in "${FOLDERS[@]}"; do
    echo ""
    echo "--- $F ---"
    if [ ! -d "$F/savings" ]; then
        echo "skip: no savings/"
        continue
    fi
    python analyzers/flow_sample_diagnostic.py "$F" -n 8000
done

echo ""
echo "Done."
