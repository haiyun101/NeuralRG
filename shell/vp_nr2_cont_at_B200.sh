#!/bin/bash -l
#SBATCH --job-name=vp_nr2_at_B200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=./logs/vp_nr2_at_B200_%j.out
#SBATCH --error=./logs/vp_nr2_at_B200_%j.err

# Best-200-anchored diagnostic + Tier-1 physics for the two nr=2 VP
# continuation arms.
#
# Both arms found their Best-200 basin at ~ep 4671 (from the full-record
# trajectory analysis on 2026-07-15). Both then drifted +45 nat over the
# remaining ~10k epochs. The earlier flow_diagnostic run (job 41814471)
# sampled at the LATEST checkpoint (ep 14500) — well past the basin —
# and reported huge KL(q||p) (101 for vp1e-3, 131 for vp1e-4).
#
# This job re-runs the diagnostic + Tier-1 at ep 4500 (nearest saving
# to the Best-200 center of 4671) to get an honest sample of what
# these arms actually learned when the loss was minimal.

module load miniforge
source activate neuralrg

mkdir -p logs

CELLS=(
    "data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16"
    "data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b16"
)
B200_EPOCH=4500   # nearest saving to Best-200 center ep 4671

echo "=========================================="
echo "vp1e-{3,4} nr=2 cont — diagnostic + Tier-1 at Best-200 epoch $B200_EPOCH"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="
date

for FOLDER in "${CELLS[@]}"; do
    echo
    echo "==================================================================="
    echo "==== $FOLDER  @ ep $B200_EPOCH"
    echo "==================================================================="

    # flow_sample_diagnostic at Best-200 checkpoint (writes JSON alongside
    # the current latest one — flow_diagnostic.json gets overwritten).
    echo
    echo ">> flow_sample_diagnostic"
    python -u analyzers/flow_sample_diagnostic.py "$FOLDER" \
        --epoch "$B200_EPOCH" -n 4000 -b 500 \
        || echo "diagnostic FAILED for $FOLDER (continuing)"

    # Tier-1 physics (chi, U_4, energy)
    echo
    echo ">> tier1_observables"
    LABEL="L64_T2.269_$(basename $FOLDER)_B200ep${B200_EPOCH}"
    python -u analyzers/tier1_observables.py \
        --folder "$FOLDER" --epoch "$B200_EPOCH" \
        --N 4000 --batch 500 \
        --label "$LABEL" --device cuda:0 \
        || echo "tier1 FAILED for $FOLDER (continuing)"
done

echo
echo "=========================================="
echo "Done. Updated flow_diagnostic.json in each folder (now B200-anchored)."
echo "Look for [TIER1_ROW] lines in stdout for chi, U_4."
echo "=========================================="
date
