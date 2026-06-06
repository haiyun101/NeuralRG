#!/bin/bash -l
#SBATCH --job-name=L32_hs_bridge
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=./logs/L32_hs_bridge_%j.out
#SBATCH --error=./logs/L32_hs_bridge_%j.err

# L=32 hs_bignet with bridge-targeted upweighting (-bridgeWeight).
# Replaces the failed entropy-reg approach (see project_entropy_reg_review):
# entropy reg fattened marginal tails globally, not the bridge specifically.
#
# Here: each training sample x_i with |M_i| < bridgeThresh gets effective
# weight (1 + bridgeWeight) in the MLE loss. ENTROPY column stays the
# unweighted -E_data[log q] so we can compare LOSS to the no-bridge baseline.
#
# Tunables via env:
#   WEIGHT=5.0      (extra multiplicative weight for bridge samples)
#   THRESH=0.5      (|M_i| < THRESH counts as bridge; M is per-config mean(x))
#   EPOCHS=2000     (matched to entropy reg β=0.005 trajectory)
#   FOLDER_SUFFIX=  (optional folder suffix, e.g. "_w10t0.3")

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
N=200000
HS_PT="data/mcmc_data/hs_L32_T${T}_N${N}.pt"
WEIGHT="${WEIGHT:-5.0}"
THRESH="${THRESH:-0.5}"
EPOCHS="${EPOCHS:-2000}"
FOLDER_SUFFIX="${FOLDER_SUFFIX:-_w${WEIGHT}t${THRESH}}"
FOLDER="./data/32Ising_T2.269_hsBignet_bridge${FOLDER_SUFFIX}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=32 hs_bignet + bridge upweighting"
echo "  bridgeWeight=$WEIGHT  bridgeThresh=$THRESH  epochs=$EPOCHS"
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
    -bridgeWeight "$WEIGHT" \
    -bridgeThresh "$THRESH"

echo "Done."
