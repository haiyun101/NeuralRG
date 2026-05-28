#!/bin/bash -l
#SBATCH --job-name=L8_hs_nsf
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=./logs/L8_hs_nsf_%j.out
#SBATCH --error=./logs/L8_hs_nsf_%j.err

# Stage-1 sanity: train an NSF-coupling MERA on L=8 HS data (forward KL,
# 3000 epochs is enough to converge at this size). Compare KL_fwd to the
# existing L=8 RNVP baseline.
#
# Default architecture (nlayers=8, nhidden=64) so total params are roughly
# comparable to the RNVP baseline (NSF MLP last layer is wider but the
# rest is identical).

module load miniforge
source activate neuralrg

mkdir -p logs

T="2.269185314213022"
HS_PT="data/mcmc_data/hs_L8_T${T}_N200000.pt"
if [ ! -f "$HS_PT" ]; then
    echo "ERROR: no HS L=8 data found in data/mcmc_data/"
    exit 1
fi

FOLDER="./data/8Ising_T2.269_hs_nsf"
mkdir -p "$FOLDER"

echo "=========================================="
echo "L=8 hs NSF sanity (3000 ep, K=8, bound=20)"
echo "data: $HS_PT"
echo "folder: $FOLDER"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "=========================================="

# Bound = 20 because HS at L=8 T_c has tails out to ~|x|<15. Generous safety.
python main.py \
    -L 8 -T "$T" \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 3000 \
    -batch 128 \
    -nlayers 8 -nmlp 3 -nhidden 64 -nrepeat 1 \
    -savePeriod 100 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq \
    -flowType nsf \
    -nsfBins 8 \
    -nsfBound 20.0

echo "Done."
