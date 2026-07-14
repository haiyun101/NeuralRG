#!/bin/bash -l
#SBATCH --job-name=swa_L64
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=./logs/swa_L64_%j.out
#SBATCH --error=./logs/swa_L64_%j.err

# Post-hoc SWA on L=64 nr=1 HCG per-scale run. Analysis showed best@ep 415
# (L=7599.79) with subsequent random walk to L≈7690 mean. Averaging the
# kept-window checkpoints [200..1400] should either (a) beat 7599.79 by
# noise-averaging into the basin center, or (b) confirm 7599 was a
# lucky-batch outlier.

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDER="data/64Ising_T2.269_hsBignet_hcg_perscale_nodilate_initshared_gc5.0_b16"
DATA="data/mcmc_data/hs_L64_T2.269185314213022_N200000.pt"

python -u analyzers/swa_eval.py \
    --folder "$FOLDER" \
    --data "$DATA" \
    --window best \
    --n 7 \
    --n-batches 100

echo "Done."
