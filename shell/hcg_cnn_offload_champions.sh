#!/bin/bash -l
#SBATCH --job-name=hcg_offload_champ
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=./logs/hcg_offload_champ_%j.out
#SBATCH --error=./logs/hcg_offload_champ_%j.err

# V6-style CNN offload metrics for HCG champions (per level).
# Adds to analyzers/csv/rg_v6_hcg_champion_offload.csv:
#   ||μ||/||z||, ⟨σ⟩, KS_raw/whit, W1_raw/whit, KL_gauss_raw/whit (per level)
#
# L=32 champion @ ep 9500 (nearest to Best-200 ep 9401)
# L=64 champion @ ep 13500 (matches existing mera_layer_flow_capture)

module load miniforge
source activate neuralrg
mkdir -p logs analyzers/csv

python -u analyzers/rg_fixed_point/hcg_cnn_offload.py \
    --N 2000 --device cpu \
    --cells \
        L32_champion:data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64:9500 \
        L64_champion:data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16:13500 \
    --csv-out analyzers/csv/rg_v6_hcg_champion_offload.csv

echo "Done."
date
