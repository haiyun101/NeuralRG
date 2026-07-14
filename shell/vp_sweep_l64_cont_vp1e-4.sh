#!/bin/bash -l
#SBATCH --job-name=vp_L64_cont_e4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=./logs/vp_L64_cont_e4_%j.out
#SBATCH --error=./logs/vp_L64_cont_e4_%j.err

module load miniforge
source activate neuralrg

mkdir -p logs
FOLDER="./data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-4_nr2_b16"
python -u main.py -load -folder "$FOLDER" -cuda 0 -symmetry
echo "Done."
