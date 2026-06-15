#!/bin/bash -l
#SBATCH --job-name=L64_iii1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L64_iii1_%j.out
#SBATCH --error=./logs/L64_iii1_%j.err

# L=64 hs_bignet baseline + III.1 multi-scale loss (single-variable
# ablation at L=64). Same flow as run_L32_iii1_single.sh — scheme III.1
# from analyzers/rg_fixed_point/improvements_zh.md adds
#   loss += lambda_scale * MSE(zscore(y_s[::2,::2]), zscore(y_{s+1}))
# at every MERA scale pair via forward_with_intermediates.
#
# batch=16 (vs L=32's 64) because at L=64 the activation footprint per
# forward pass is 4x, and scaleLoss adds a third forward graph.

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T=2.269185314213022
N=200000
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
LAMBDA_SCALE="${LAMBDA_SCALE:-1.0}"
EPOCHS="${EPOCHS:-20000}"
BATCH="${BATCH:-16}"
FOLDER_SUFFIX="${FOLDER_SUFFIX:-_lam${LAMBDA_SCALE}_b${BATCH}}"
FOLDER="./data/${L}Ising_T2.269_hsBignet_iii1${FOLDER_SUFFIX}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L hs_bignet + III.1 multi-scale loss"
echo "  lambda_scale=$LAMBDA_SCALE  batch=$BATCH  epochs=$EPOCHS"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="

python -u main.py \
    -L $L -T $T \
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
