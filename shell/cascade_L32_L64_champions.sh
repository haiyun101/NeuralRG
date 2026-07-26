#!/bin/bash -l
#SBATCH --job-name=cascade_champions
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/cascade_champions_%j.out
#SBATCH --error=./logs/cascade_champions_%j.err

# Cross-L champion-to-champion layer analysis.
# Consumes mera_layer_flow_capture.pt from both champion folders and
# writes cascade_layer_L32vsL64_champions.csv covering:
#   A: per-scale marginal (skew, kurt, KS)
#   B: cross-scale self-similarity within each model
#   C: G(r) axial + xi_s from raw kept-coarse fields
#   D: forward-inverse consistency (y_s vs w_s at same scale)
#   E: cross-model at same scale (L32_champion vs L64_champion)

module load miniforge
source activate neuralrg
mkdir -p logs

L32=data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64/mera_layer_flow_capture.pt
L64=data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16/mera_layer_flow_capture.pt

if [ ! -f "$L32" ]; then
    echo "ERROR: $L32 not found — did capture_L32_champion.sh finish?"
    exit 1
fi
if [ ! -f "$L64" ]; then
    echo "ERROR: $L64 not found"
    exit 1
fi

python -u analyzers/rg_fixed_point/cascade_layer_analysis.py \
    --captures  L32_champion="$L32"  L64_champion="$L64" \
    --out       analyzers/rg_fixed_point/csv/cascade_layer_L32vsL64_champions.csv

echo
echo "===  CSV written: analyzers/rg_fixed_point/csv/cascade_layer_L32vsL64_champions.csv"
date
