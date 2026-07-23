#!/bin/bash -l
#SBATCH --job-name=cross_T_diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/cross_T_diag_%j.out
#SBATCH --error=./logs/cross_T_diag_%j.err

# Scoped cross-T analysis on GPU. Previous attempts (jobs 41811610,
# 41814473 step 3) all walltime-cancelled because CPU flow_sample_diagnostic
# on 12 folders + 5 GT was too slow.
#
# Only two steps here (NOT the V-battery from 41814473):
#   1. flow_sample_diagnostic per (T, model) — 12 folders
#      Writes flow_diagnostic.json per folder.
#   2. tier1_observables per (T, model) + L=64 GT at 5 T's
#      Writes [TIER1_ROW] stdout lines.
#
# GPU + N=1000 (up from 500 CPU) should complete in ~1.5 h.
# 3 h walltime with margin.
#
# 3 models × 4 T = 12 folders, all at nr=2 with N=200000 dataset.

module load miniforge
source activate neuralrg

mkdir -p logs

TS=("2.15" "2.22" "2.32" "2.4")
TAGS=("baseline_nr2" "i2_stride8h32_nr2" "hcg_perscale_fixdil_vp1e-3_nr2")

echo "=========================================="
echo "L=64 cross-T diagnostic + Tier-1 (GPU)"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  T's: ${TS[*]}"
echo "  models: ${TAGS[*]}"
echo "=========================================="
date

# ── STEP 1: flow_sample_diagnostic per (T, model) ──
echo
echo ">>>>>>>>>> [1/2] flow_sample_diagnostic <<<<<<<<<<"
date
for T in "${TS[@]}"; do
    for TAG in "${TAGS[@]}"; do
        FOLDER=data/64Ising_T${T}_hsBignet_${TAG}_b16
        [ ! -d "$FOLDER/savings" ] && { echo "SKIP: $FOLDER (missing)"; continue; }
        echo
        echo "--- $FOLDER ---"
        python -u analyzers/flow_sample_diagnostic.py "$FOLDER" \
            -n 1000 -b 500 \
            || echo "diagnostic FAILED for $FOLDER (continuing)"
    done
done

# ── STEP 2: tier1_observables per (T, model) + L=64 GT sweep ──
echo
echo ">>>>>>>>>> [2/2] tier1_observables <<<<<<<<<<"
date

for T in "${TS[@]}"; do
    for TAG in "${TAGS[@]}"; do
        FOLDER=data/64Ising_T${T}_hsBignet_${TAG}_b16
        [ ! -d "$FOLDER/savings" ] && continue
        LABEL="L64_T${T}_${TAG}"
        echo
        echo "--- tier1: $FOLDER ---"
        python -u analyzers/tier1_observables.py \
            --folder "$FOLDER" --T "$T" --N 1000 --batch 500 \
            --label "$LABEL" --device cuda:0 \
            || echo "tier1 FAILED for $FOLDER (continuing)"
    done
done

# L=64 GT sweep
echo
echo "--- L=64 GT sweep ---"
for T in 2.15 2.22 2.269185314213022 2.32 2.4; do
    LABEL=$(python3 -c "print('L64_GT_T'+'${T}'[:5])")
    python -u analyzers/tier1_observables.py \
        --folder GT --L 64 --T "$T" --N 4000 \
        --label "$LABEL" --device cuda:0 \
        || echo "GT tier1 FAILED at T=$T (continuing)"
done

echo
echo "=========================================="
echo "Done. Outputs:"
echo "  - <each folder>/flow_diagnostic.json"
echo "  - stdout: [TIER1_ROW] lines (model + GT)"
echo "=========================================="
date
