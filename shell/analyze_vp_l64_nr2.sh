#!/bin/bash
#SBATCH --job-name=analyze_vp64nr2
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/analyze_vp_L64nr2_%j.out
#SBATCH --error=./logs/analyze_vp_L64nr2_%j.err

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDERS=(
    data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b16
    data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16
    data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-2_nr2_b16
)

python -u analyzers/flow_sample_diagnostic.py "${FOLDERS[@]}" -n 4000 -b 256

echo "Done."
