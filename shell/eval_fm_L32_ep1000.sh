#!/bin/bash -l
#SBATCH --job-name=fm_eval_ep1000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/fm_eval_ep1000_%j.out
#SBATCH --error=./logs/fm_eval_ep1000_%j.err

# Honest evaluation of the FM prototype at ep 1000 (only checkpoint that
# made disk before the 12h walltime cut on job 41818094).
#
# High-quality sampling: N=4000 with 100-step RK4 (much more accurate than
# the periodic-eval's 500-sample 50-step Euler that suggested χ=54.60).
# Plus GT comparison + KL(p||q) computation via Hutchinson-trace ODE.

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDER="data/L32_T2.269_flowmatching_h64"

echo "=========================================="
echo "FM prototype honest evaluation at ep 1000"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="
date

python -u analyzers/fm_evaluate.py \
    --folder "$FOLDER" \
    --epoch 1000 \
    --N 4000 --batch 500 \
    --steps 100 --solver rk4 \
    --T 2.269185314213022 \
    --gt-compare \
    --compute-kl --kl-samples 1000 --kl-eps 2 \
    --label "fm_L32_h64_ep1000"

echo
echo "=========================================="
echo "Done."
date
