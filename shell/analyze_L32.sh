#!/bin/bash
#SBATCH --job-name=nrg_L32_analyze
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=./logs/analyze_L32_%j.out
#SBATCH --error=./logs/analyze_L32_%j.err

# L=32 post-hoc analysis. Chained afterok the hs_dataDriven training job.
# Phase A: flow-sample diagnostic (<A>_q, H(q), F_c^q, KL(q||p), KL(p||q))
#          for the new hs_dataDriven run + the existing reverse-KL runs.
# Phase B: regenerate the 6-column thermodynamic report. Now that hs_L32
#          samples exist, the theory E_c / S_c columns get filled in too.

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDERS=(
    data/32Ising_T2.269_hs_dataDriven
    data/32Ising_T2.269_sym
    data/32Ising_T2.269_nsym
    data/32Ising_T2.269_sym_longer
    data/32Ising_T2.269_nsym_longer
    data/32Ising_T2.269_nsym_HP
    data/32Ising_T2.269_nsym_WT
)

echo "=========================================="
echo "L=32 Analysis  |  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="

echo ""
echo "--- Phase A: flow-sample diagnostic ---"
python analyzers/flow_sample_diagnostic.py "${FOLDERS[@]}" -n 8000 -b 512

echo ""
echo "--- Phase B: thermodynamic report ---"
python analyzers/loss_analyzer_fixT.py -L 32 -t 2.269

echo ""
echo "Done. Report at analyzers/loss_report_L32_T2.269.md"
