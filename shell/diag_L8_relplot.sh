#!/bin/bash -l
#SBATCH --job-name=diag_L8_replot
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=./logs/diag_L8_replot_%j.out
#SBATCH --error=./logs/diag_L8_replot_%j.err

# Re-run flow_sample_diagnostic.py on the L=8 reference runs so that
# the flow_correlations.png panel uses the current log-log axes + Onsager
# eta=1/4 reference line. The May-21 PNGs were generated before the
# log-log rewrite of save_corr_png().

module load miniforge
source activate neuralrg

mkdir -p logs

python analyzers/flow_sample_diagnostic.py \
    ./data/8Ising_T2.269_sym/ \
    ./data/8Ising_T2.269_hs_dataDriven/ \
    -n 8000

echo "Done."
