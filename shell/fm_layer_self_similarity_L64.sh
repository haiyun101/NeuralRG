#!/bin/bash -l
#SBATCH --job-name=fm_layer_ss_L64
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=./logs/fm_layer_ss_L64_%j.out
#SBATCH --error=./logs/fm_layer_ss_L64_%j.err

# MERAUNet FM per-layer self-similarity — L=64 h128 continuation checkpoint.
# CSV output mirrors L=32 version so we can compare cross-L internal
# representation structure of the same MERAUNet architecture family.

module load miniforge
source activate neuralrg
mkdir -p logs analyzers/csv

python -u analyzers/fm_layer_self_similarity.py \
    --ckpts meraFM_L64_h128_ep50:data/L64_T2.269_meraFM_h128/savings/fm_L64_T2.269185314213022_epoch50.pt \
    --L 64 --T 2.269185314213022 --N 300 \
    --nhidden 128 --tembDim 128 --maxChannelMult 4 \
    --tValues "0.1,0.3,0.5,0.7,0.9,1.0" \
    --out analyzers/csv/fm_layer_self_similarity_L64.csv

echo "Done."
date
