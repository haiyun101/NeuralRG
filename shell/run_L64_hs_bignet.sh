#!/bin/bash -l
#SBATCH --job-name=L64_hs_bignet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L64_hs_%j.out
#SBATCH --error=./logs/L64_hs_%j.err

# L=64 T_c forward-KL on HS continuous-field samples, bignet arch.
# REQUIRES: shell/gen_mcmc_L64.sh has produced
#   data/mcmc_data/hs_L64_T<T>_N200000.pt
# Submit gen_mcmc_L64 first; use --dependency=afterok:<jobid> if you
# launch them together, or just wait for the data file to appear.
#
# batch=64 (down from 128 at L=32) for A100-40G activation memory.
# 20000 epochs aims for the same convergence ratio as L=32 hs_dataDriven
# (20000 ep at L=32 default arch); bignet at L=32 needed 10000-15000.

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T=2.269185314213022
N=200000
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
FOLDER="./data/${L}Ising_T2.269_hs_bignet"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS dataset not found at $HS_PT"
    echo "Submit shell/gen_mcmc_L64.sh first."
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L forward-KL (HS) bignet (b=64, A100)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "data: $HS_PT"
echo "folder: $FOLDER"
echo "=========================================="

python -u main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 20000 \
    -batch 32 \
    -nlayers 16 \
    -nmlp 3 \
    -nhidden 128 \
    -nrepeat 1 \
    -savePeriod 100 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq

echo "Done."
