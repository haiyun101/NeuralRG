#!/bin/bash -l
#SBATCH --job-name=champ_V_battery
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=./logs/champ_V_battery_%j.out
#SBATCH --error=./logs/champ_V_battery_%j.err

# Complete the V-battery for the L=64 champion (fixdil+VP-1e-3 nr=1).
# Job 41802239 already covered V0/V1 via rg_fixed_point.py; this job
# adds V2 + V2b + V3 (robustness.py), V4 (v4_dataforward.py),
# and V5 RMS-G + matched-pair MSE (v5_blockRG_compare.py).
#
# Each script iterates its own FOLDERS registry — the champion was
# added to those registries in the same edit (2026-07-14). The scripts
# process all registered folders, so L=64 A + D + i1_df4 rows re-run
# alongside the champion. Idempotent for the already-published ones.
#
# Runs on CPU. 10h walltime is generous — v4/v5 with N=2000 typically
# finish in ~1-2h each.

module load miniforge
source activate neuralrg

mkdir -p logs
OUTDIR=analyzers/rg_fixed_point

echo "=========================================="
echo "L=64 champion V-battery: V2 + V2b + V3 + V4 + V5"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="
date

# ── V2 / V2b / V3 — chained/slot-geometry-corrected + identity residual ──
echo
echo ">>>>>>>>>> [1/3] V2 / V2b / V3 (robustness.py) <<<<<<<<<<"
date
python -u analyzers/rg_fixed_point/rg_fixed_point_robustness.py \
    --N 10000 --outdir "$OUTDIR" \
    || echo "STEP 1 FAILED (non-fatal, continuing)"

# ── V4 — HS data forward adjacent-scale MSE ──
echo
echo ">>>>>>>>>> [2/3] V4 (v4_dataforward.py) <<<<<<<<<<"
date
python -u analyzers/rg_fixed_point/rg_fixed_point_v4_dataforward.py \
    --N 2000 --outdir "$OUTDIR" \
    || echo "STEP 2 FAILED (non-fatal, continuing)"

# ── V5 — Wilson block-RG ground truth comparison ──
echo
echo ">>>>>>>>>> [3/3] V5 (v5_blockRG_compare.py) <<<<<<<<<<"
date
python -u analyzers/rg_fixed_point/rg_v5_blockRG_compare.py \
    --N 2000 --outdir "$OUTDIR" \
    || echo "STEP 3 FAILED (non-fatal)"

echo
echo "=========================================="
echo "V-battery complete. Look for these outputs (updated in-place):"
echo "  - $OUTDIR/csv/rg_fixed_point_robustness.csv    (V2/V2b/V3)"
echo "  - $OUTDIR/csv/rg_v4_dataforward.csv            (V4)"
echo "  - $OUTDIR/csv/rg_v5_blockRG_compare.csv        (V5)"
echo "  - $OUTDIR/figures/rg_fixed_point_robustness.png (+ methods)"
echo "  - $OUTDIR/figures/rg_v4_*.png"
echo "  - $OUTDIR/figures/rg_v5_*.png"
echo "=========================================="
date
