#!/bin/bash
#SBATCH --job-name=nrg_L64_analyze
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=./logs/analyze_L64_%j.out
#SBATCH --error=./logs/analyze_L64_%j.err

# L=64 post-hoc analysis for the 5 completed bignet runs.
# Phase A: flow-sample diagnostic — produces flow_diagnostic.json,
#          flow_samples.png, flow_correlations.png in each folder.
#          These feed the per-method tables + visuals in
#          analyzers/concise_reports/concise_report_L64_T2.269.md
#          (currently a stub for diagnostic rows and visuals).
# Phase B: thermodynamic report — analyzers/loss/loss_analyzer_fixT.py
#          rebuilds analyzers/loss/loss_report_L64_T2.269.md from the
#          freshly-written flow_diagnostic.json's.
#
# Uses gpu partition (CPU-only would take ~hours for L=64 bignet
# 10.9M params x N=4000 samples per folder).

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDERS=(
    data/64Ising_T2.269_sym_bignet
    data/64Ising_T2.269_pathgrad_bignet
    data/64Ising_T2.269_hs_bignet
    data/64Ising_T2.269_jsLoss_bignet_lam0.5
    data/64Ising_T2.269_hsBignet_bridge_w5.0t0.5
)

echo "=========================================="
echo "L=64 Analysis  |  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="

echo ""
echo "--- Phase A: flow-sample diagnostic ---"
# N=4000 (down from L=32's 8000) for memory; L=64 fields are 4x larger.
python -u analyzers/flow_sample_diagnostic.py "${FOLDERS[@]}" -n 4000 -b 128

echo ""
echo "--- Phase B: L=64 thermodynamic report ---"
python -u analyzers/loss/loss_analyzer_fixT.py -L 64 -t 2.269 || \
    echo "(loss_analyzer_fixT may not yet support L=64; skipping)"

echo ""
echo "Done."
