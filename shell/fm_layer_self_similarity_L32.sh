#!/bin/bash -l
#SBATCH --job-name=fm_layer_ss
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=./logs/fm_layer_ss_%j.out
#SBATCH --error=./logs/fm_layer_ss_%j.err

# MERAUNet FM per-layer self-similarity analysis on meraFM_L32 @ ep 1200.
# Output analog to cascade section A + B (marginal + adjacent-scale) at
# multiple ODE times so we can directly compare to MERA champion's cascade.

module load miniforge
source activate neuralrg
mkdir -p logs analyzers/csv

python -u analyzers/fm_layer_self_similarity.py \
    --ckpts meraFM_L32_ep1200:data/L32_T2.269_meraFM_h64/savings/fm_L32_T2.269185314213022_epoch1200.pt \
    --L 32 --T 2.269185314213022 --N 500 \
    --nhidden 64 --tembDim 128 --maxChannelMult 4 \
    --tValues "0.1,0.3,0.5,0.7,0.9,1.0" \
    --out analyzers/csv/fm_layer_self_similarity_L32.csv

echo "Done."
date
