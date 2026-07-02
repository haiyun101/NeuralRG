#!/bin/bash -l
#SBATCH --job-name=hcg_L8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=./logs/hcg_L8_%j.out
#SBATCH --error=./logs/hcg_L8_%j.err

# HCG sanity check at L=8: 500 epochs, scale-shared CNN, verify loss goes
# down and no NaN. Small architecture (nlayers=8, nhidden=32) since L=8 has
# only 64 sites and 3 hierarchy levels [4, 2, 1].

module load miniforge
source activate neuralrg

mkdir -p logs

L=8
T=2.269185314213022
N=200000
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
FOLDER="./data/8Ising_T2.269_hcg_sanity"

if [ ! -f "$HS_PT" ]; then
    echo "ERROR: HS data missing: $HS_PT"
    exit 1
fi

mkdir -p "$FOLDER"

echo "=========================================="
echo "HCG sanity check at L=$L, T=T_c"
echo "  scale_shared=1, hidden=32, dilated=1, circular=1"
echo "  epochs=500 batch=64"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="

python -u main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs 500 \
    -batch 64 \
    -nlayers 8 -nmlp 3 -nhidden 32 -nrepeat 1 \
    -lr 1e-3 \
    -savePeriod 100 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 1 \
    -hcgHidden 32 \
    -hcgDilated 1 \
    -hcgCircular 1

echo "Done."
