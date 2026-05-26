#!/bin/bash
#SBATCH --job-name=nrg_L32_sweep_analyze
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=./logs/L32_sweep_analyze_%j.out
#SBATCH --error=./logs/L32_sweep_analyze_%j.err

# Post-hoc analysis of the L=32 hs_dataDriven architecture sweep.
# Chained afterok the three training jobs.
#
# Phase A: flow-sample diagnostic at n=8000 -- KL(q||p), KL(p||q), and the
#          structural scalars (<|M|>, G(L/2)/G(0), xi_eff).  Also writes
#          flow_samples.png and flow_correlations.png (clean n=8000 P(M)).
# Phase B: regenerate the L=32 thermodynamic report (picks up the new arms).

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDERS=(
    data/32Ising_T2.269_hs_dataDriven     # baseline (nlayers 10, nhidden 64)
    data/32Ising_T2.269_hs_bignet         # deeper + wider
    data/32Ising_T2.269_hs_haarPrior      # Haar majority-vote prior
    data/32Ising_T2.269_hs_weightTying    # scale-invariant weight sharing
)

echo "=========================================="
echo "L=32 sweep analysis  |  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="

echo ""
echo "--- Phase A: flow-sample diagnostic (n=8000) + PNGs ---"
python analyzers/flow_sample_diagnostic.py "${FOLDERS[@]}" -n 8000 -b 512

echo ""
echo "--- Phase B: thermodynamic report ---"
python analyzers/loss_analyzer_fixT.py -L 32 -t 2.269

echo ""
echo "Done. Report at analyzers/loss_report_L32_T2.269.md"
echo "Per-folder: flow_diagnostic.json, flow_samples.png, flow_correlations.png"
