#!/bin/bash -l
#SBATCH --job-name=clean_ep
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/clean_ep_%j.out
#SBATCH --error=./logs/clean_ep_%j.err
module load miniforge
source activate neuralrg
FOLDER=data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-4_nr1_b16
# try 4 progressively earlier checkpoints
for EP in 14500 13500 12500 11000 9500 8000; do
    echo
    echo "=========================================="
    echo "==== $FOLDER  @ ep $EP"
    echo "=========================================="
    python -u analyzers/flow_sample_diagnostic.py "$FOLDER" \
        -n 4000 -b 256 --epoch $EP --no-png --no-json 2>&1 | \
        grep -E "ckpt epoch|F_c\^q|KL\(q\|\||KL\(p\|\||CE = |H\(p"
done
echo "Done."
