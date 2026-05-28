#!/bin/bash -l
#SBATCH --job-name=L32_jsLoss
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=5:00:00
#SBATCH --output=./logs/L32_jsLoss_%j.out
#SBATCH --error=./logs/L32_jsLoss_%j.err

# L=32 JS-like (symmetrized KL) training, lambda=0.5, default arch.
# loss = 0.5 * KL(q||p) + 0.5 * KL(p||q)
#
# Science question: at L=32, reverse-KL stalled at KL(q||p) ~ 12 nat
# (mode-dropping floor) while forward-KL achieved KL(p||q) ~ 3.6 nat.
# Does mixing the two losses break the floor?
#
# Default architecture (nlayers=10, nhidden=64) for first test:
#   - JS costs ~2x reverse-KL per step (both flow.sample + flow.logProb)
#   - L=32 forward-KL at default ran in ~6h for 20k epochs
#   - So budget 5h for ~5000 JS epochs on l40
#   - JS should converge faster than either alone (both signals push toward
#     correct shape simultaneously), so 5k epochs should suffice for a
#     first read; if promising, graduate to bignet.

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
N=200000
HS_PT="data/mcmc_data/hs_L32_T${T}_N${N}.pt"
FOLDER="./data/32Ising_T2.269_jsLoss_lam0.5"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"
echo "=========================================="
echo "L=32 JS-like training  |  lambda=0.5  |  default arch"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python main.py \
    -L 32 -T "$T" \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 5000 \
    -batch 128 \
    -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 \
    -savePeriod 100 \
    -symmetry \
    -skipHMC \
    -jsLoss -jsLambda 0.5 \
    -dataPath "$HS_PT" \
    -noDeq

echo "Done."
