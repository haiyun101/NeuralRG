#!/bin/bash -l
#SBATCH --job-name=L64_bridge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L64_bridge_%j.out
#SBATCH --error=./logs/L64_bridge_%j.err

# L=64 T_c forward-KL with bridge-domain upweighting:
#   re-weighted target p̃ ∝ (1 + WEIGHT * 1[|M_i| < THRESH]) p
# Same WEIGHT=5.0, THRESH=0.5 as the L=32 best-structural-match run.
# REQUIRES the HS dataset.

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T=2.269185314213022
N=200000
WEIGHT=5.0
THRESH=0.5
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
FOLDER="./data/${L}Ising_T2.269_hsBignet_bridge_w${WEIGHT}t${THRESH}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS dataset not found at $HS_PT"
    echo "Submit shell/gen_mcmc_L64.sh first."
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L bridge-reweighted fwd-KL bignet (w=$WEIGHT, t=$THRESH, b=64, A100)"
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
    -noDeq \
    -bridgeWeight "$WEIGHT" \
    -bridgeThresh "$THRESH"

echo "Done."
