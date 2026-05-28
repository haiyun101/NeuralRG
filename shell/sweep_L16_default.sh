#!/bin/bash -l
#SBATCH --job-name=L16_def
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=./logs/L16_def_%x_%j.out
#SBATCH --error=./logs/L16_def_%x_%j.err

# Re-run L=16 sweep with DEFAULT arch (nlayers=10, nhidden=64) instead of
# the original "midbig" (l=12, h=96). At T_c, default beats midbig by
# 1.6 nat (LOSS 477.51 vs 479.09). Investigating whether the same holds
# off-T_c so the FSS L=16 curve becomes clean.
#
# Usage:  sbatch --job-name=L16d_T<T> shell/sweep_L16_default.sh <T>
# Writes to: ./data/16Ising_T<T>_hs_dataDriven_default/

module load miniforge
source activate neuralrg

mkdir -p logs

T="$1"
if [ -z "$T" ]; then
    echo "ERROR: usage  sbatch shell/sweep_L16_default.sh <T>"
    exit 1
fi

N=200000
EPOCHS=20000
HS_PT="data/mcmc_data/hs_L16_T${T}_N${N}.pt"
FOLDER="./data/16Ising_T${T}_hs_dataDriven_default"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"
echo "=========================================="
echo "L=16 hs_dataDriven  DEFAULT arch  |  T=${T}"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python main.py \
    -L 16 -T "$T" \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" \
    -batch 128 \
    -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 \
    -savePeriod 500 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq

echo "Done (L=16 default, T=${T})."
