#!/bin/bash
#SBATCH --job-name=nrg_sweep_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=./logs/sweep_train_%x_%j.out
#SBATCH --error=./logs/sweep_train_%x_%j.err

# Train HS forward-KL flow at one (L, T) pair.
# Usage: sbatch --job-name=tr_L{L}_T{T} shell/sweep_temp_train.sh <L> <T>
# Architecture is chosen by L (per the L=32 bignet capacity finding):
#   L=8   -> default  (nlayers=10  nhidden=64)
#   L=16  -> midbig   (nlayers=12  nhidden=96)
#   L=32  -> bignet   (nlayers=16  nhidden=128)
# Output folder: ./data/${L}Ising_T${T}_hs_dataDriven

module load miniforge
source activate neuralrg

mkdir -p logs

L="$1"
T="$2"
N=200000
EPOCHS=20000

if [ -z "$L" ] || [ -z "$T" ]; then
    echo "ERROR: usage  sbatch shell/sweep_temp_train.sh <L> <T>"
    exit 1
fi

HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
FOLDER="./data/${L}Ising_T${T}_hs_dataDriven"

case "$L" in
    8)  ARCH="-nlayers 10 -nmlp 3 -nhidden 64  -nrepeat 1 -symmetry" ;;
    16) ARCH="-nlayers 12 -nmlp 3 -nhidden 96  -nrepeat 1 -symmetry" ;;
    32) ARCH="-nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 -symmetry" ;;
    *)  echo "ERROR: unsupported L=$L (expected 8, 16, or 32)"; exit 1 ;;
esac

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    echo "(the data-gen job should produce this; check dependency chain)"
    exit 1
fi

mkdir -p "$FOLDER"
echo "=========================================="
echo "HS forward-KL  |  L=${L}  T=${T}"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "arch:   $ARCH"
echo "epochs: $EPOCHS"
echo "data:   $HS_PT"
echo "=========================================="

python main.py \
    -L "$L" -T "$T" \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" \
    -batch 128 \
    -savePeriod 500 \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq \
    $ARCH

echo ""
echo "Done (L=${L}, T=${T})."
