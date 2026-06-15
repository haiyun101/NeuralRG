#!/bin/bash -l
#SBATCH --job-name=L32_i1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/L32_i1_%j.out
#SBATCH --error=./logs/L32_i1_%j.err

# L=32 hs_bignet baseline + I.1 Student-t prior.
# Scheme I.1 from analyzers/rg_fixed_point/improvements_zh.md — the
# *negation* experiment: Student-t has heavier tails than Gaussian but
# still factorises across sites (P(z) = prod_i p_t(z_i)). If V5 KS
# improves but V5 RMS-G stays put, the bottleneck is spatial structure
# (need I.2/A or I.3/B), not marginal tails. If neither moves, the
# Gaussian-prior hypothesis is the wrong frame entirely.
#
# Tunables via env:
#   DF=4.0               (degrees-of-freedom; >2 for finite variance)
#   EPOCHS=20000
#   BATCH=128            (same as hs_bignet baseline — no extra forward graph)
#   FOLDER_SUFFIX=       (defaults to _df${DF})

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
N=200000
HS_PT="data/mcmc_data/hs_L32_T${T}_N${N}.pt"
DF="${DF:-4.0}"
EPOCHS="${EPOCHS:-20000}"
BATCH="${BATCH:-128}"
FOLDER_SUFFIX="${FOLDER_SUFFIX:-_df${DF}}"
FOLDER="./data/32Ising_T2.269_hsBignet_i1${FOLDER_SUFFIX}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=32 hs_bignet + I.1 Student-t prior  (df=$DF)"
echo "  batch=$BATCH  epochs=$EPOCHS"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="

python -u main.py \
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
    -priorType studentT \
    -priorDf "$DF"

echo "Done."
