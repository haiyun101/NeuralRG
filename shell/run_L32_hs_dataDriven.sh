#!/bin/bash
#SBATCH --job-name=nrg_L32_hs
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=./logs/L32_hs_%j.out
#SBATCH --error=./logs/L32_hs_%j.err

module load miniforge
source activate neuralrg

mkdir -p logs

T=2.269185314213022
L=32
EPOCHS=30000
DISC_PT="data/mcmc_data/mcmc_wolff_L32_T2.269_N200000.pt"
HS_PT="data/mcmc_data/hs_L32_T${T}_N200000.pt"
FOLDER="./data/32Ising_T2.269_hs_dataDriven"

echo "=========================================="
echo "L=32 HS Data-Driven Training"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "=========================================="

# Phase 1: Convert discrete samples to exact HS continuous samples (x ~ N(Ks, K))
echo ""
echo "--- Phase 1: Convert discrete MCMC → HS continuous samples ---"
if [ -f "$HS_PT" ]; then
    echo "HS samples already exist at $HS_PT — skipping conversion."
else
    python generate_hs_samples.py -L $L -T $T \
        --in_path $DISC_PT \
        --out_dir data/mcmc_data
fi
echo "Conversion done."

# Phase 2: Train on HS samples (forward KL / MLE, exact continuous target, no dequantization)
mkdir -p $FOLDER
echo ""
echo "--- Phase 2: Train on HS samples (forward KL, exact continuous target) ---"
python main.py \
    -L $L \
    -T $T \
    -folder $FOLDER \
    -cuda 0 \
    -epochs $EPOCHS \
    -batch 128 \
    -nlayers 10 \
    -nmlp 3 \
    -nhidden 64 \
    -nrepeat 1 \
    -savePeriod 500 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath $HS_PT \
    -noDeq
echo "Training done."

echo ""
echo "=========================================="
echo "All done."
echo "=========================================="
