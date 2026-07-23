#!/bin/bash -l
#SBATCH --job-name=diag_vp_nr2_cont
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/diag_vp_nr2_cont_%j.out
#SBATCH --error=./logs/diag_vp_nr2_cont_%j.err

# Post-training diagnostic + Best-200 rerun for the two nr=2 continuation
# training jobs that finished 2026-07-15 (jobs 41786697 and 41797159,
# both reached ep 15000). Each folder now has fresh Adam-restored
# training up through ep 15000, so re-running diagnostic + Best-200
# gives us the honest sustained-loss for these two arms.

module load miniforge
source activate neuralrg

mkdir -p logs

CELLS=(
    "data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16"
    "data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b16"
)

echo "=========================================="
echo "Post-training diagnostic + Best-200 for vp1e-{3,4} nr=2 continuations"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="
date

for FOLDER in "${CELLS[@]}"; do
    echo
    echo "==================================================================="
    echo "==== $FOLDER"
    echo "==================================================================="
    # 1. Best-200 anchored epoch (recomputed over full record)
    echo
    echo ">> Best-200 (full-record)"
    python3 analyzers/dump_best_200_epochs.py -L 64 -t 2.269 --top 60 2>/dev/null \
        | grep "$(basename $FOLDER)" || echo "(not in top 60 — will still diagnose latest)"

    # 2. flow_sample_diagnostic at latest checkpoint
    echo
    echo ">> flow_sample_diagnostic (latest saving)"
    python -u analyzers/flow_sample_diagnostic.py "$FOLDER" \
        -n 4000 -b 500 \
        || echo "diagnostic FAILED for $FOLDER (continuing)"
done

echo
echo "=========================================="
echo "Done."
echo "=========================================="
date
