#!/bin/bash -l
#SBATCH --job-name=fm_L64_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=./logs/fm_L64_eval_%j.out
#SBATCH --error=./logs/fm_L64_eval_%j.err

# Honest eval of FM L=64 at ep 200 checkpoint from job 1717126.
# Same recipe as L=32 ep 1000 eval:
#   N=4000, RK4-100, GT compare, KL(p||q) via Hutchinson trace ODE.
# GT reference: χ=110.15, U₄=0.611, ⟨|M|⟩=0.614 at L=64 T_c.

module load miniforge
source activate neuralrg
mkdir -p logs

python -u analyzers/fm_evaluate.py \
    --folder data/L64_T2.269_flowmatching_h64 \
    --epoch 200 \
    --N 4000 --batch 500 \
    --steps 100 --solver rk4 \
    --T 2.269185314213022 \
    --gt-compare \
    --compute-kl --kl-samples 1000 --kl-eps 2 \
    --label "fm_L64_h64_ep200"

date
