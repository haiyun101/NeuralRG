#!/bin/bash -l
#SBATCH --job-name=plot_L128_early
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/plot_L128_early_%j.out
#SBATCH --error=./logs/plot_L128_early_%j.err

# Physics visualization for L=128 early-trajectory 4-cell comparison
# (from job 1806651). Multi-model plots for warm/fresh/MERAonly/CNNonly
# at user-requested checkpoints (ep 5, 10, 30, 50, 100).

module load miniforge
source activate neuralrg
cd /cluster/home/hhuang05/NeuralRG
mkdir -p logs

WARM=data/L128_early_warm_from_L64
FRESH=data/L128_early_fresh
MERAONLY=data/L128_early_MERAonly
CNNONLY=data/L128_early_CNNonly

# --- Two comparison views ---

# View 1: warm vs fresh across early epochs (visualizes warm-start advantage)
echo "==================================================================="
echo "=== View 1: warm vs fresh across early epochs"
echo "==================================================================="
python -u analyzers/plot_model_physics.py \
    --N 1000 --batch 32 --device cuda \
    --out figures/L128_early_warm_vs_fresh \
    --cells \
        "warm_ep5:$WARM:5" \
        "warm_ep10:$WARM:10" \
        "warm_ep30:$WARM:30" \
        "warm_ep50:$WARM:50" \
        "warm_ep100:$WARM:100" \
        "fresh_ep5:$FRESH:5" \
        "fresh_ep10:$FRESH:10" \
        "fresh_ep30:$FRESH:30" \
        "fresh_ep50:$FRESH:50" \
        "fresh_ep100:$FRESH:100"

# View 2: 4-way ablation at ep 100 (compare warm/fresh/MERAonly/CNNonly)
echo
echo "==================================================================="
echo "=== View 2: 4-way ablation at ep 100 (both/fresh/MERAonly/CNNonly)"
echo "==================================================================="
python -u analyzers/plot_model_physics.py \
    --N 1000 --batch 32 --device cuda \
    --out figures/L128_early_4way_ep100 \
    --cells \
        "both_transfer:$WARM:100" \
        "fresh:$FRESH:100" \
        "MERAonly:$MERAONLY:100" \
        "CNNonly:$CNNONLY:100"

echo
echo "Done."
date
