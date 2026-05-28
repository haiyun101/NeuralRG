#!/bin/bash -l
#SBATCH --job-name=L32_hs_nsf
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/L32_hs_nsf_%j.out
#SBATCH --error=./logs/L32_hs_nsf_%j.err

# Stage-2 NSF on L=32 hs_dataDriven.
# Two configs (selected via ARCH env var):
#   ARCH=default : nlayers=10 nhidden=64  (head-to-head vs L=32 RNVP hs_dataDriven)
#   ARCH=bignet  : nlayers=16 nhidden=128 (head-to-head vs hs_bignet, KL_fwd=16.37 nat)
#
# RNVP baselines for reference:
#   hs_dataDriven           : LOSS=~1922  -> KL_fwd~19 nat (default arch)
#   hs_bignet               : LOSS=1919.35 -> KL_fwd=16.37 nat (bignet arch)
#
# K=8 bins, bound=20 (HS L=32 T_c tails ~ |x|<15).

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
N=200000
HS_PT="data/mcmc_data/hs_L32_T${T}_N${N}.pt"
ARCH="${ARCH:-default}"
EPOCHS="${EPOCHS:-8000}"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

if [ "$ARCH" = "bignet" ]; then
    NLAYERS=16; NHIDDEN=128
elif [ "$ARCH" = "default" ]; then
    NLAYERS=10; NHIDDEN=64
else
    echo "ERROR: ARCH must be 'default' or 'bignet'"; exit 1
fi

FOLDER="./data/32Ising_T2.269_hs_nsf_${ARCH}"
mkdir -p "$FOLDER"

echo "=========================================="
echo "L=32 hs NSF Stage-2  (arch=$ARCH, K=8, bound=20)"
echo "nlayers=$NLAYERS nhidden=$NHIDDEN epochs=$EPOCHS"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python main.py \
    -L 32 -T "$T" \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" \
    -batch 128 \
    -nlayers "$NLAYERS" -nmlp 3 -nhidden "$NHIDDEN" -nrepeat 1 \
    -savePeriod 200 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq \
    -flowType nsf \
    -nsfBins 8 \
    -nsfBound 20.0

echo "Done."
