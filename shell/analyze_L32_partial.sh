#!/bin/bash
#SBATCH --job-name=nrg_L32_partial
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=./logs/L32_partial_%j.out
#SBATCH --error=./logs/L32_partial_%j.err

# Preliminary L=32 analysis of the FINISHED forward-KL sweep arms
# (hs_dataDriven baseline, hs_haarPrior, hs_weightTying). hs_bignet is
# still training -> diagnostic skips it, and the report excludes it via
# --exclude bignet. Reverse-KL folders keep their existing
# flow_diagnostic.json from the earlier analyze_L32 run.
#
# The full 4-arm report is produced later by sweep_L32_analyze.sh (job
# chained afterok the bignet run); this just gives an early look.

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDERS=(
    data/32Ising_T2.269_hs_dataDriven
    data/32Ising_T2.269_hs_haarPrior
    data/32Ising_T2.269_hs_weightTying
)

echo "=========================================="
echo "L=32 preliminary analysis  |  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="

echo ""
echo "--- Phase A: flow diagnostic (n=8000) on finished forward-KL arms ---"
python analyzers/flow_sample_diagnostic.py "${FOLDERS[@]}" -n 8000 -b 512

echo ""
echo "--- Phase B: L=32 report (bignet excluded - still training) ---"
python analyzers/loss_analyzer_fixT.py -L 32 -t 2.269 --exclude bignet

echo ""
echo "Done. Report at analyzers/loss_report_L32_T2.269.md"
