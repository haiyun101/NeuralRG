#!/bin/bash -l
#SBATCH --job-name=champ_layer_analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=./logs/champ_layer_analysis_%j.out
#SBATCH --error=./logs/champ_layer_analysis_%j.err

# Full layer-level analysis of the L=64 forward-KL champion
#   folder: data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16
#   Best-200 epoch: 13290, checkpoint at ep 13500 (nearest saving)
#
# Four analyses run sequentially:
#  1. mera_layer_flow_capture — per-layer forward+inverse activations with
#     per-site Gaussianization, saved as mera_layer_flow_capture.pt inside
#     the champion folder. Prereq for later cross-scale similarity work.
#  2. mera_layer_stats — L2 norms of every RNVP block's parameters +
#     cosine similarity between the 12 RNVP blocks (are they doing
#     similar work, or is the flow "coarse vs fine" specialized?).
#  3. hcg_perscale_similarity — cross-scale HCG CNN comparison:
#     raw weight cosine, sigma distribution match across levels,
#     cross-application swap test. Tells us if VP pushed per-scale
#     CNNs toward or away from scale-invariance.
#  4. rg_fixed_point.py — V0-V5 RG probe on the L=64 comparison panel
#     (champion vs baseline A vs D). Adjacent-scale MSE plot; a flow at
#     an RG fixed point should show low MSE at deep layers.
#
# Runs on CPU (batch partition, no GPU needed for these analyses).
# 16h walltime — previous attempt at 3h timed out mid-champion.

module load miniforge
source activate neuralrg

mkdir -p logs

CHAMP_FOLDER="data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16"
CHAMP_EPOCH=13290   # Best-200 center; nearest .saving is ep 13500

echo "=========================================="
echo "L=64 champion layer-level analysis battery"
echo "  target: $CHAMP_FOLDER"
echo "  Best-200 epoch: $CHAMP_EPOCH"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="
date

# ── (1/4) MERA per-layer activation capture ─────────────────────────
echo
echo ">>>>>>>>>> [1/4] mera_layer_flow_capture <<<<<<<<<<"
date
python -u analyzers/rg_fixed_point/mera_layer_flow_capture.py \
    --folder "$CHAMP_FOLDER" --epoch "$CHAMP_EPOCH" \
    --N 2000 --device cpu \
    || { echo "STEP 1 FAILED"; exit 1; }

# ── (2/4) MERA per-layer weight stats ───────────────────────────────
echo
echo ">>>>>>>>>> [2/4] mera_layer_stats <<<<<<<<<<"
date
python -u analyzers/rg_fixed_point/mera_layer_stats.py \
    --folder "$CHAMP_FOLDER" --epoch "$CHAMP_EPOCH" --device cpu \
    || echo "STEP 2 FAILED (non-fatal, continuing)"

# ── (3/4) HCG per-scale similarity ──────────────────────────────────
echo
echo ">>>>>>>>>> [3/4] hcg_perscale_similarity <<<<<<<<<<"
date
python -u analyzers/rg_fixed_point/hcg_perscale_similarity.py \
    --folder "$CHAMP_FOLDER" --epoch "$CHAMP_EPOCH" \
    --N 500 --device cpu \
    || echo "STEP 3 FAILED (non-fatal, continuing)"

# ── (4/4) V0-V5 RG probe for L=64 comparison panel ──────────────────
# rg_fixed_point.py uses a hardcoded FOLDERS registry (edited 2026-07-13
# to add the champion + baseline + D). This step processes ALL folders in
# that registry, so the L=32 rows re-run too — output is idempotent.
# Look for figures/rg_fixed_point_L64_champion.png in the outdir.
echo
echo ">>>>>>>>>> [4/4] rg_fixed_point.py (V0-V5 probe, all registered folders) <<<<<<<<<<"
date
python -u analyzers/rg_fixed_point/rg_fixed_point.py \
    --N 10000 --outdir analyzers/rg_fixed_point \
    || echo "STEP 4 FAILED (non-fatal)"

echo
echo "=========================================="
echo "All steps complete. Outputs:"
echo "  - $CHAMP_FOLDER/mera_layer_flow_capture.pt"
echo "  - $CHAMP_FOLDER/mera_layer_flow_capture.json"
echo "  - stdout: mera_layer_stats + hcg_perscale_similarity tables + CSV files"
echo "  - analyzers/rg_fixed_point/figures/rg_fixed_point_L64_champion.png"
echo "  - analyzers/rg_fixed_point/csv/rg_fixed_point_summary.csv"
echo "=========================================="
date
