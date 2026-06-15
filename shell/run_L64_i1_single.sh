#!/bin/bash -l
#SBATCH --job-name=L64_i1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=./logs/L64_i1_%j.out
#SBATCH --error=./logs/L64_i1_%j.err

# L=64 hs_bignet baseline + I.1 Student-t prior.
# Cross-L parallel of the L=32 negation experiment. Same flow as
# run_L32_i1_single.sh; only the lattice size and the matched-batch
# (16, to match baseline_b16 / iii1_b16 / i2_b16) differ.

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T=2.269185314213022
N=200000
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
DF="${DF:-4.0}"
EPOCHS="${EPOCHS:-20000}"
BATCH="${BATCH:-16}"
FOLDER_SUFFIX="${FOLDER_SUFFIX:-_df${DF}_b${BATCH}}"
FOLDER="./data/${L}Ising_T2.269_hsBignet_i1${FOLDER_SUFFIX}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L hs_bignet + I.1 Student-t prior  (df=$DF)"
echo "  batch=$BATCH  epochs=$EPOCHS"
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
    -priorType studentT \
    -priorDf "$DF"

echo "Done."
