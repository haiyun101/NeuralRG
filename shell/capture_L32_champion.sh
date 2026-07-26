#!/bin/bash -l
#SBATCH --job-name=capture_L32_champ
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/capture_L32_champ_%j.out
#SBATCH --error=./logs/capture_L32_champ_%j.err

# Capture per-layer activations (both directions, per-site z-scored)
# for the L=32 champion: fixdil+VP-1e-3 nr=1  (Best-200 S=1912.56 @ ep 9401).
# Nearest saved checkpoint is epoch 9500 (savePeriod=500).
#
# Output: data/32Ising_.../mera_layer_flow_capture.pt

module load miniforge
source activate neuralrg
mkdir -p logs

FOLDER=data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64
EPOCH=9500

python -u analyzers/rg_fixed_point/mera_layer_flow_capture.py \
    --folder "$FOLDER" --epoch "$EPOCH" --N 4000 --device cpu

echo
echo "===  capture written: $FOLDER/mera_layer_flow_capture.pt"
date
