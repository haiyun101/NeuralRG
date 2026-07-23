#!/bin/bash -l
#SBATCH --job-name=L64_across_T
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/L64_across_T_%j.out
#SBATCH --error=./logs/L64_across_T_%j.err

# L=64 cross-T analysis battery — three tasks in one shell:
#   1. flow_sample_diagnostic on 12 model folders (4 T × 3 models)
#      → writes flow_diagnostic.json per folder (KL, |M|, xi, G(r))
#   2. tier1_observables on 12 model folders + L=64 GT at 5 T's
#      → [TIER1_ROW] summary lines with chi, U_4, energy, |M|
#   3. rg_fixed_point.py (V0/V1) — will process the 12 new cross-T
#      registry entries added earlier today, plus all pre-existing rows.
#      New PANELS: rg_fixed_point_L64_across_T_D.png and
#                  rg_fixed_point_L64_across_T_VP.png
#
# Models (all nr=2, N=200 000 dataset):
#   C  = baseline_nr2         (Gaussian nr=2)
#   D  = i2_stride8h32_nr2    (Phase-2 reference, conditional_gaussian)
#   VP = hcg_perscale_fixdil_vp1e-3_nr2 (champion-analog with VP)
# Temperatures:
#   2.15 (ordered), 2.22 (near T_c), 2.32 (near T_c), 2.4 (disordered)
# T = 2.269 = T_c already covered elsewhere.
#
# CPU only. 12h walltime.

module load miniforge
source activate neuralrg

mkdir -p logs

TS=("2.15" "2.22" "2.32" "2.4")
TAGS=("baseline_nr2" "i2_stride8h32_nr2" "hcg_perscale_fixdil_vp1e-3_nr2")

echo "=========================================="
echo "L=64 cross-T analysis battery"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  T's: ${TS[*]}"
echo "  models: ${TAGS[*]}"
echo "=========================================="
date

# ── STEP 1: flow_sample_diagnostic (KL, |M|, xi, G(r)) ─────────────────
echo
echo ">>>>>>>>>> [1/3] flow_sample_diagnostic on 12 folders <<<<<<<<<<"
date
for T in "${TS[@]}"; do
    for TAG in "${TAGS[@]}"; do
        FOLDER=data/64Ising_T${T}_hsBignet_${TAG}_b16
        if [ ! -d "$FOLDER/savings" ]; then
            echo "SKIP: $FOLDER (missing)"
            continue
        fi
        echo
        echo "--- $FOLDER ---"
        python -u analyzers/flow_sample_diagnostic.py "$FOLDER" \
            -n 4000 -b 500 \
            || echo "diagnostic FAILED for $FOLDER (continuing)"
    done
done

# ── STEP 2: tier1_observables (chi, U_4, E) ─────────────────────────────
echo
echo ">>>>>>>>>> [2/3] tier1_observables — models + L=64 GT sweep <<<<<<<<<<"
date

# Models
for T in "${TS[@]}"; do
    for TAG in "${TAGS[@]}"; do
        FOLDER=data/64Ising_T${T}_hsBignet_${TAG}_b16
        if [ ! -d "$FOLDER/savings" ]; then
            echo "SKIP tier1: $FOLDER (missing)"
            continue
        fi
        LABEL="L64_T${T}_${TAG}"
        echo
        echo "--- tier1: $FOLDER ---"
        python -u analyzers/tier1_observables.py \
            --folder "$FOLDER" --T "$T" --N 4000 --batch 500 \
            --label "$LABEL" --device cpu \
            || echo "tier1 FAILED for $FOLDER (continuing)"
    done
done

# L=64 GT sweep (may need to be run only where the HS data file exists)
echo
echo "--- tier1: L=64 GT sweep ---"
for T in 2.15 2.22 2.269185314213022 2.32 2.4; do
    LABEL=$(python3 -c "print('L64_GT_T'+'${T}'[:5])")
    python -u analyzers/tier1_observables.py \
        --folder GT --L 64 --T "$T" --N 10000 \
        --label "$LABEL" --device cpu \
        || echo "tier1 GT FAILED for T=$T (continuing)"
done

# ── STEP 3: rg_fixed_point.py (V0/V1 across all registered folders) ────
# 12 new cross-T entries were added to FOLDERS today; running the script
# processes them all + all pre-existing (idempotent for those).
echo
echo ">>>>>>>>>> [3/3] rg_fixed_point.py (V0/V1 across all) <<<<<<<<<<"
date
python -u analyzers/rg_fixed_point/rg_fixed_point.py \
    --N 10000 --outdir analyzers/rg_fixed_point \
    || echo "STEP 3 FAILED (non-fatal)"

echo
echo "=========================================="
echo "Cross-T analysis complete. Outputs:"
echo "  - <each folder>/flow_diagnostic.json (KL, |M|, xi, G(r))"
echo "  - stdout [TIER1_ROW] lines (chi, U_4, energy per model + GT)"
echo "  - analyzers/rg_fixed_point/csv/rg_fixed_point_summary.csv (updated)"
echo "  - analyzers/rg_fixed_point/figures/rg_fixed_point_L64_across_T_D.png"
echo "  - analyzers/rg_fixed_point/figures/rg_fixed_point_L64_across_T_VP.png"
echo "=========================================="
date
