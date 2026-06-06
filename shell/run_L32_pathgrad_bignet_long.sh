#!/bin/bash
#SBATCH --job-name=L32_pg_long
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/L32_pg_long_%j.out
#SBATCH --error=./logs/L32_pg_long_%j.err

# Long L=32 STL pilot (5000 epochs, b=128, bignet) to settle the
# "STL converges to a lower asymptote / lower noise floor" question
# at matched epoch count vs the baseline data/32Ising_T2.269_sym_bignet.
#
# The earlier 2000-ep b=128 STL (job 39315244) finished at KL_best_sm=10.75
# vs baseline-at-1950ep KL=11.22 -- slight win but inconclusive on whether
# the gap widens with more epochs (cf. L=8 long pair, which showed STL
# crossing baseline only well past 1500 ep).
#
# Output: ./data/32Ising_T2.269_pathgrad_bignet_long
# Compare against: ./data/32Ising_T2.269_sym_bignet (b=128, std reparam, 5950 ep)

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDER="./data/32Ising_T2.269_pathgrad_bignet_long"
mkdir -p "$FOLDER"

echo "=========================================="
echo "L=32 STL bignet LONG (b=128, 5000 ep, A100-80G)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python -u main.py \
    -L 32 -T 2.269 \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 5000 \
    -batch 128 \
    -nlayers 16 \
    -nmlp 3 \
    -nhidden 128 \
    -nrepeat 1 \
    -savePeriod 50 \
    -symmetry \
    -skipHMC \
    -pathGrad

echo "Done."
