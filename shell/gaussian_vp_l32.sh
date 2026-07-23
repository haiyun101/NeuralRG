#!/bin/bash -l
#SBATCH --job-name=gauss_vp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/gauss_vp_%j.out
#SBATCH --error=./logs/gauss_vp_%j.err

# Plain Gaussian prior + VP penalty at L=32 T_c.
# Prediction: this fails because Gaussian prior has no learnable σ, so
# MERA is the ONLY thing that can reshape data → prior. VP kills MERA's
# reshape ability → training struggles.
#
# Baseline (Gaussian nr=1, no VP): F ≈ 1899.32 at L=32 T_c.
# Champion (fixdil+VP-1e-3 nr=1): F = 1891.10.
#
# Env vars:
#   VP_LAMBDA (required, e.g. 1e-3)

module load miniforge
source activate neuralrg
mkdir -p logs

L=32
T=2.269185314213022
N=200000
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
VP_LAMBDA="${VP_LAMBDA:?VP_LAMBDA env var required}"
EPOCHS="${EPOCHS:-8000}"

FOLDER="./data/${L}Ising_T2.269_gaussian_vp${VP_LAMBDA}_nr1_b64"

echo "=========================================="
echo "L=$L Plain Gaussian prior + VP-${VP_LAMBDA}"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="
date

python -u main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" -batch 64 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -lr 1e-3 -gradClip 5.0 \
    -savePeriod 500 \
    -symmetry -skipHMC \
    -dataDriven -dataPath "$HS_PT" -noDeq \
    -priorType gaussian \
    -volumePreservingWeight "$VP_LAMBDA"

date
