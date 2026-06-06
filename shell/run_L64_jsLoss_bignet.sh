#!/bin/bash -l
#SBATCH --job-name=L64_jsLoss
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L64_jsLoss_%j.out
#SBATCH --error=./logs/L64_jsLoss_%j.err

# L=64 T_c Jensen-Shannon mixed objective (lam=0.5: 0.5*rev-KL + 0.5*fwd-KL),
# bignet. Companion to the pure rev-KL (sym) and pure fwd-KL (hs) flows.
# REQUIRES the HS dataset (same as run_L64_hs_bignet.sh).

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T=2.269185314213022
N=200000
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
FOLDER="./data/${L}Ising_T2.269_jsLoss_bignet_lam0.5"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS dataset not found at $HS_PT"
    echo "Submit shell/gen_mcmc_L64.sh first."
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L jsLoss bignet (lam=0.5, b=64, A100)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "data: $HS_PT"
echo "folder: $FOLDER"
echo "=========================================="

python -u main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 20000 \
    -batch 16 \
    -nlayers 16 \
    -nmlp 3 \
    -nhidden 128 \
    -nrepeat 1 \
    -savePeriod 100 \
    -symmetry \
    -skipHMC \
    -jsLoss -jsLambda 0.5 \
    -dataPath "$HS_PT" \
    -noDeq

echo "Done."
