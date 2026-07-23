#!/bin/bash -l
#SBATCH --job-name=L64_family_layer
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/L64_family_layer_%j.out
#SBATCH --error=./logs/L64_family_layer_%j.err

# Extend the champion layer-level analysis (job 41802239) to the two
# Phase-2 comparison references:
#   A = baseline_b16 (Gaussian nr=1)      @ Best-200 ep 14979
#   D = i2_stride8h32_nr2 (Phase-2 winner) @ Best-200 ep 17754
#
# What runs here:
#   1. mera_layer_flow_capture  (both A and D)
#   2. mera_layer_stats         (both A and D)
#
# What we SKIP (and why):
#   - hcg_perscale_similarity: A has no HCG at all; D uses a single CNN
#     (`conditional_gaussian`), not per-scale HCG. N/A for both.
#   - hcg_sigma_law: same reason.
#   - rg_fixed_point (V0-V5): already covered by the champion job's step
#     4, which processes ALL folders in the FOLDERS registry, including
#     A and D. Rerunning would be duplicate work.
#
# Output goes into the folders themselves:
#   data/64Ising_T2.269_hsBignet_baseline_b16/mera_layer_flow_capture.pt
#   data/64Ising_T2.269_hsBignet_baseline_b16/mera_layer_flow_capture.json
#   data/64Ising_T2.269_hsBignet_i2_stride8h32_nr2_b16/mera_layer_flow_capture.pt
#   data/64Ising_T2.269_hsBignet_i2_stride8h32_nr2_b16/mera_layer_flow_capture.json
#
# CPU-only. Runs in parallel with the champion job (no shared state
# beyond the FOLDERS registry read at import time).

module load miniforge
source activate neuralrg

mkdir -p logs

# folder → Best-200 epoch (from dump_best_200_epochs.py -L 64 --top 15)
CELLS=(
    "data/64Ising_T2.269_hsBignet_baseline_b16                14979"
    "data/64Ising_T2.269_hsBignet_i2_stride8h32_nr2_b16       17754"
)

echo "=========================================="
echo "L=64 layer-level analysis — A + D (Phase-2 references)"
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

    # ── (1/2) MERA per-layer activation capture ──
    echo
    echo ">>>>>>>>>> [1/2] mera_layer_flow_capture <<<<<<<<<<"
    date
    python -u analyzers/rg_fixed_point/mera_layer_flow_capture.py \
        --folder "$FOLDER" --epoch "$EPOCH" \
        --N 2000 --device cpu \
        || echo "flow_capture FAILED for $FOLDER (continuing)"

    # ── (2/2) MERA per-layer weight stats ──
    echo
    echo ">>>>>>>>>> [2/2] mera_layer_stats <<<<<<<<<<"
    date
    python -u analyzers/rg_fixed_point/mera_layer_stats.py \
        --folder "$FOLDER" --epoch "$EPOCH" --device cpu \
        || echo "layer_stats FAILED for $FOLDER (continuing)"
done

echo
echo "=========================================="
echo "All folders processed. Outputs:"
for cell in "${CELLS[@]}"; do
    F=$(echo "$cell" | awk '{print $1}')
    echo "  - $F/mera_layer_flow_capture.pt (+ .json)"
    echo "  - stdout: mera_layer_stats table"
done
echo "=========================================="
date
