#!/bin/bash -l
#SBATCH --job-name=L64_top5_probes
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/L64_top5_probes_%j.out
#SBATCH --error=./logs/L64_top5_probes_%j.err

# Scoped V4 + V5 for the top-5 L=64 models, plus a much lighter cross-T
# pass. Uses the new --filter CLI on rg_fixed_point_v4_dataforward.py /
# rg_v5_blockRG_compare.py.
#
# Top-5 filter regex covers: A (baseline_b16), B (i2_stride8h32_b16
# Phase-2), C (baseline_nr2_b16 P2.x C64), D (i2_stride8h32_nr2_b16
# P2.x D64 ★), champion (fixdil+VP-1e-3 nr=1 ★).
#
# Cross-T (step 3) reduced to N=500 (was 4000 in the failed run) and
# CPU-only. Only 3 flows × 4 T = 12 diagnostic runs.
#
# All CPU, 12h walltime.

module load miniforge
source activate neuralrg

mkdir -p logs
OUTDIR=analyzers/rg_fixed_point

# Regex that matches all five top-model labels
TOP5_FILTER='baseline_b16|Phase-2\)$|P2\.x C64|P2\.x D64|champion'

echo "=========================================="
echo "L=64 scoped V4 + V5 + cross-T diagnostics"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  top-5 filter: $TOP5_FILTER"
echo "=========================================="
date

# ── STEP 1: V4 (data-forward) for L=64 top-5 ──
echo
echo ">>>>>>>>>> [1/3] V4 (v4_dataforward.py) — top-5 filter, N=2000 <<<<<<<<<<"
date
python -u analyzers/rg_fixed_point/rg_fixed_point_v4_dataforward.py \
    --N 2000 --outdir "$OUTDIR" \
    --filter "$TOP5_FILTER" \
    || echo "STEP 1 FAILED (non-fatal)"

# ── STEP 2: V5 (Wilson block-RG) for L=64 top-5 ──
echo
echo ">>>>>>>>>> [2/3] V5 (v5_blockRG_compare.py) — top-5 filter, N=2000 <<<<<<<<<<"
date
python -u analyzers/rg_fixed_point/rg_v5_blockRG_compare.py \
    --N 2000 --outdir "$OUTDIR" \
    --filter "$TOP5_FILTER" \
    || echo "STEP 2 FAILED (non-fatal)"

# ── STEP 3: cross-T flow_diagnostic + tier1 (light, N=500) ──
# Only 12 folders (4 T × 3 models). CPU-bound; N=500 keeps sampling cost low.
TS=("2.15" "2.22" "2.32" "2.4")
TAGS=("baseline_nr2" "i2_stride8h32_nr2" "hcg_perscale_fixdil_vp1e-3_nr2")

echo
echo ">>>>>>>>>> [3/3] cross-T diagnostic + tier1 (12 folders, N=500) <<<<<<<<<<"
date
for T in "${TS[@]}"; do
    for TAG in "${TAGS[@]}"; do
        FOLDER=data/64Ising_T${T}_hsBignet_${TAG}_b16
        [ ! -d "$FOLDER/savings" ] && { echo "SKIP: $FOLDER (missing)"; continue; }
        LABEL="L64_T${T}_${TAG}"
        echo
        echo "--- $FOLDER ---"
        # diagnostic
        python -u analyzers/flow_sample_diagnostic.py "$FOLDER" \
            -n 500 -b 500 \
            || echo "diagnostic FAILED for $FOLDER (continuing)"
        # tier1 physics
        python -u analyzers/tier1_observables.py \
            --folder "$FOLDER" --T "$T" --N 500 --batch 500 \
            --label "$LABEL" --device cpu \
            || echo "tier1 FAILED for $FOLDER (continuing)"
    done
done

# L=64 GT sweep (once — no per-checkpoint sampling)
echo
echo "--- L=64 GT sweep ---"
for T in 2.15 2.22 2.269185314213022 2.32 2.4; do
    LABEL=$(python3 -c "print('L64_GT_T'+'${T}'[:5])")
    python -u analyzers/tier1_observables.py \
        --folder GT --L 64 --T "$T" --N 2000 \
        --label "$LABEL" --device cpu \
        || echo "GT tier1 FAILED at T=$T (continuing)"
done

echo
echo "=========================================="
echo "Done. Outputs (updated in-place where existing):"
echo "  - $OUTDIR/csv/rg_v4_dataforward.csv   (V4 top-5 rows)"
echo "  - $OUTDIR/csv/rg_v5_blockRG_compare.csv (V5 top-5 rows)"
echo "  - <each cross-T folder>/flow_diagnostic.json"
echo "  - stdout: [TIER1_ROW] lines (model + GT)"
echo "=========================================="
date
