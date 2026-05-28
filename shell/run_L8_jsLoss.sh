#!/bin/bash -l
#SBATCH --job-name=L8_jsLoss
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=./logs/L8_jsLoss_%j.out
#SBATCH --error=./logs/L8_jsLoss_%j.err

# L=8 JS-like (symmetrized KL) training, lambda=0.5.
# loss = 0.5 * KL(q||p) + 0.5 * KL(p||q)
# Reuses HS data (forward-KL term) + flow samples (reverse-KL term) per step.
# Quick L=8 sanity run -- if the loss decreases and final state has KL
# competitive with the L=8 forward/reverse runs, the implementation works.

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
N=200000
HS_PT="data/mcmc_data/hs_L8_T${T}_N${N}.pt"
FOLDER="./data/8Ising_T2.269_jsLoss_lam0.5"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"
echo "=========================================="
echo "L=8 JS-like training  |  lambda=0.5"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python main.py \
    -L 8 -T "$T" \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 10000 \
    -batch 128 \
    -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 \
    -savePeriod 500 \
    -symmetry \
    -skipHMC \
    -jsLoss -jsLambda 0.5 \
    -dataPath "$HS_PT" \
    -noDeq

echo "Done."
