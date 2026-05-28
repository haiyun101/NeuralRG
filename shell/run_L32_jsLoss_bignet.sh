#!/bin/bash -l
#SBATCH --job-name=L32_js_big
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=5:00:00
#SBATCH --output=./logs/L32_js_big_%j.out
#SBATCH --error=./logs/L32_js_big_%j.err

# L=32 JS-like (symmetrized KL) training with BIGNET architecture.
# loss = 0.5 * KL(q||p) + 0.5 * KL(p||q),  nlayers=16, nhidden=128 (~10.9M params).
#
# Default-arch L=32 jsLoss saturated at KL_qp=17.3, KL_pq=16.4 by epoch ~3000.
# Both errors capped by capacity, not by the objective. Bignet should let JS
# drive both KL terms down to single digits in parallel -- IF the objective
# really can support a balanced solution at this scale.
#
# Reference (L=32 T_c at bignet capacity):
#   hs_bignet   (forward-KL):  KL(p||q) = 3.6  (KL(q||p) diag = 21.3)
#   sym_bignet  (reverse-KL):  KL(q||p) = 9.7
#   THIS RUN    (JS, lam=0.5): predicted KL(q||p) ~5-8, KL(p||q) ~5-8  IF JS+capacity wins.
#
# Budget: 3000 epochs. JS ~2x reverse-KL cost; bignet L=32 reverse-KL was
# ~50 min / 1000 epochs on l40, so ~3000 epochs ~5h on l40 (tight but ok).

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
N=200000
HS_PT="data/mcmc_data/hs_L32_T${T}_N${N}.pt"
FOLDER="./data/32Ising_T2.269_jsLoss_bignet_lam0.5"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"
echo "=========================================="
echo "L=32 JS-like training (BIGNET)  |  lambda=0.5"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $FOLDER"
echo "=========================================="

python main.py \
    -L 32 -T "$T" \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 3000 \
    -batch 128 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 1 \
    -savePeriod 50 \
    -symmetry \
    -skipHMC \
    -jsLoss -jsLambda 0.5 \
    -dataPath "$HS_PT" \
    -noDeq

echo "Done."
