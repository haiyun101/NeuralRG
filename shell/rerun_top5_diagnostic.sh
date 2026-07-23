#!/bin/bash -l
#SBATCH --job-name=top5_diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/top5_diag_%j.out
#SBATCH --error=./logs/top5_diag_%j.err

# Rerun flow_sample_diagnostic on the top-5 L=64 T_c models at their
# Best-200 epochs. Produces REAL signed magnetization samples via the
# actual flow — replaces the synthetic bimodal M distribution used in
# the earlier JSON-only regeneration.
#
# Uses the current 2-panel save_corr_png (M histogram + normalized G(r)/G(0)
# log-log with Onsager reference), so the new flow_correlations.png will
# have Z2-symmetric bimodal M peaks from real samples.

module load miniforge
source activate neuralrg

mkdir -p logs

# folder → Best-200 epoch (nearest saving)
CELLS=(
    "data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16 13500"
    "data/64Ising_T2.269_hsBignet_baseline_b16                       15000"
    "data/64Ising_T2.269_hsBignet_i2_stride8h32_b16                  12500"
    "data/64Ising_T2.269_hsBignet_baseline_nr2_b16                   18000"
    "data/64Ising_T2.269_hsBignet_i2_stride8h32_nr2_b16              17800"
)

echo "=========================================="
echo "L=64 T_c top-5 diagnostic rerun with REAL samples"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="
date

for cell in "${CELLS[@]}"; do
    FOLDER=$(echo "$cell" | awk '{print $1}')
    EPOCH=$(echo  "$cell" | awk '{print $2}')
    echo
    echo "==================================================================="
    echo "==== $FOLDER  @ Best-200 ep $EPOCH"
    echo "==================================================================="
    python -u analyzers/flow_sample_diagnostic.py "$FOLDER" \
        --epoch "$EPOCH" -n 4000 -b 500 \
        || echo "diagnostic FAILED for $FOLDER (continuing)"
done

echo
echo "=========================================="
echo "Done. Each top-5 folder now has:"
echo "  - flow_diagnostic.json (KL, |M|, xi, G(r) at Best-200 epoch)"
echo "  - flow_correlations.png (real signed M distribution + normalized G(r)/G(0))"
echo "  - flow_samples.png"
echo "=========================================="
date
