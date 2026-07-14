#!/bin/bash -l
#SBATCH --job-name=mera_layer_flow
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/mera_layer_flow_%j.out
#SBATCH --error=./logs/mera_layer_flow_%j.err

# Capture per-layer activations in BOTH directions with per-site
# Gaussianization for top L=64 variants at their Best-200 epoch.
# Writes mera_layer_flow_capture.pt inside each folder.
# Used later by cross-layer similarity comparison.

module load miniforge
source activate neuralrg

mkdir -p logs

python3 analyzers/dump_best_200_epochs.py -L 64 -t 2.269 --top 10 > /tmp/b200_capture.txt
echo "Cells:"
cat /tmp/b200_capture.txt
echo

while IFS=$'\t' read -r folder ep S; do
    echo
    echo "==================================================================="
    echo "==== $folder  @ ep $ep"
    echo "==================================================================="
    python -u analyzers/rg_fixed_point/mera_layer_flow_capture.py \
        --folder "$folder" --epoch "$ep" --N 4000 --device cpu
done < /tmp/b200_capture.txt

echo "Done."
