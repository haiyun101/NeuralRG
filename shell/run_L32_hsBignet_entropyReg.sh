#!/bin/bash -l
#SBATCH --job-name=L32_hs_ent
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=./logs/L32_hs_ent_%j.out
#SBATCH --error=./logs/L32_hs_ent_%j.err

# L=32 hs_bignet Phase-1 with entropy regularization (-entropyBeta).
# Hypothesis: hs_bignet already matches data bridge_p (0.0095 vs 0.020 data)
# but is still under-density. Adding -beta * H(q) to the MLE loss should
# push q to be broader, increasing bridge mass.
#
# Cost: ~2x base data-driven step (adds a sampling forward+backward pass).
# At L=32 bignet, baseline was ~1s/ep on l40 -> ~2s/ep -> ~5h for 10000 ep.
#
# BETA chosen via env var so we can sweep. Default 0.05 (moderate).
# Try {0.05, 0.10, 0.20} to bracket.

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
N=200000
HS_PT="data/mcmc_data/hs_L32_T${T}_N${N}.pt"
BETA="${BETA:-0.05}"
EPOCHS="${EPOCHS:-10000}"
FOLDER="./data/32Ising_T2.269_hsBignet_ent${BETA}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"
echo "=========================================="
echo "L=32 hs_bignet + entropy reg  (beta=$BETA, epochs=$EPOCHS)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python main.py \
    -L 32 -T "$T" \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" \
    -batch 128 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -savePeriod 200 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq \
    -entropyBeta "$BETA"

echo "Done."
