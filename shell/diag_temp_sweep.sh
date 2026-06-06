#!/bin/bash -l
#SBATCH --job-name=diag_sweep
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=./logs/diag_sweep_%j.out
#SBATCH --error=./logs/diag_sweep_%j.err

# Regenerate flow_diagnostic.json + flow_samples.png + flow_correlations.png
# (log-log axes) for the full FSS forward-KL temperature sweep at L=8/16/32.
# Previous PNGs predate the log-log rewrite of save_corr_png(), and the
# existing JSONs at L=8 lack the HS-data-side KL(p||q), CE_pq, mag_abs_p,
# xi_p columns because they were generated before the HS dataset was
# linked in. This job runs the diagnostic with the current code on every
# (L, T) folder used in fss_sweep_KL_v2.csv.

module load miniforge
source activate neuralrg

mkdir -p logs

python analyzers/flow_sample_diagnostic.py \
    data/8Ising_T2.15_hs_dataDriven \
    data/8Ising_T2.22_hs_dataDriven \
    data/8Ising_T2.269_hs_dataDriven \
    data/8Ising_T2.32_hs_dataDriven \
    data/8Ising_T2.4_hs_dataDriven \
    data/16Ising_T2.15_hs_dataDriven_default \
    data/16Ising_T2.22_hs_dataDriven_default \
    data/16Ising_T2.269_hs_dataDriven \
    data/16Ising_T2.32_hs_dataDriven_default \
    data/16Ising_T2.4_hs_dataDriven_default \
    data/32Ising_T2.15_hs_dataDriven \
    data/32Ising_T2.22_hs_dataDriven \
    data/32Ising_T2.269_hs_dataDriven \
    data/32Ising_T2.32_hs_dataDriven \
    data/32Ising_T2.4_hs_dataDriven \
    -n 8000

echo "Done."
