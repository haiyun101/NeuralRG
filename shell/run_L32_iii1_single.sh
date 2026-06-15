#!/bin/bash -l
#SBATCH --job-name=L32_iii1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/L32_iii1_%j.out
#SBATCH --error=./logs/L32_iii1_%j.err

# L=32 hs_bignet baseline + III.1 multi-scale loss (single-variable ablation).
# Scheme III.1 from analyzers/rg_fixed_point/improvements_zh.md:
#   loss += lambda_scale * sum_s MSE(zscore(y_s[::2^(s+1), ::2^(s+1)]),
#                                    zscore(y_{s+1}[::2^(s+1), ::2^(s+1)]))
# Penalises mismatch between adjacent MERA scales' kept-coarse subsets
# after per-sample z-scoring. Targets the rev-KL deep-block collapse and
# fwd-KL deep-block inflation pathologies described in
# rg_fixed_point_report_zh.md.
#
# Tunables via env:
#   LAMBDA_SCALE=1.0     (multi-scale loss coefficient; 0 = baseline)
#   EPOCHS=20000
#   FOLDER_SUFFIX=       (optional; defaults to _lam${LAMBDA_SCALE})
#
# This is a SINGLE-VARIABLE run: same as hs_bignet (run_L32_hsBignet*.sh)
# except for the added -scaleLoss flag, so the diff vs baseline isolates
# the multi-scale loss contribution to V3/V4/V5 metrics.

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
N=200000
HS_PT="data/mcmc_data/hs_L32_T${T}_N${N}.pt"
LAMBDA_SCALE="${LAMBDA_SCALE:-1.0}"
EPOCHS="${EPOCHS:-20000}"
# Batch halved vs hs_bignet baseline (128 -> 64) because scaleLoss adds
# a third forward-graph (its own pass + symmetry's 2 internal passes +
# alpha-penalty's 2 internal passes = 5 graphs at peak). At batch=128
# the 40 GiB A100 OOMs at the alpha-penalty step. batch=64 fits.
BATCH="${BATCH:-64}"
FOLDER_SUFFIX="${FOLDER_SUFFIX:-_lam${LAMBDA_SCALE}_b${BATCH}}"
FOLDER="./data/32Ising_T2.269_hsBignet_iii1${FOLDER_SUFFIX}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=32 hs_bignet + III.1 multi-scale loss"
echo "  lambda_scale=$LAMBDA_SCALE  epochs=$EPOCHS"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="

python main.py \
    -L 32 -T "$T" \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" \
    -batch "$BATCH" \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -savePeriod 200 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq \
    -scaleLoss "$LAMBDA_SCALE"

echo "Done."
