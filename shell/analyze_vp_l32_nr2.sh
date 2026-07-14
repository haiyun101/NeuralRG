#!/bin/bash
#SBATCH --job-name=analyze_vp32nr2
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=./logs/analyze_vp_L32nr2_%j.out
#SBATCH --error=./logs/analyze_vp_L32nr2_%j.err

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDERS=(
    data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-5_nr2_b64
    data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b64
    data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b64
    data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-2_nr2_b64
)

python -u analyzers/flow_sample_diagnostic.py "${FOLDERS[@]}" -n 8000 -b 512

echo "Done."
