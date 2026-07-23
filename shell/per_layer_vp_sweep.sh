#!/bin/bash -l
#SBATCH --job-name=perlayer_vp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/perlayer_vp_%j.out
#SBATCH --error=./logs/perlayer_vp_%j.err

# Per-layer VP sweep at L=32 T_c.
#
# Tests whether per-layer VP (penalizes Σ_block (log|det J_block|)² individually,
# preventing expand/contract cancellation between adjacent blocks) fixes the
# nr=2 Group-B collapse pattern.
#
# Each variant is one job — submit multiple via env-var CONFIG:
#   CONFIG=A_nr1     — Gaussian prior nr=1 (baseline sanity check)
#   CONFIG=champion_nr1 — HCG fixdil nr=1 (already known Group A, sanity check)
#   CONFIG=champion_nr2 — HCG fixdil nr=2 (the KEY test: does per-layer VP
#                                          push it to Group A?)
#   CONFIG=D_nr2     — i2 conditional_gaussian nr=2 (Phase-2 reference,
#                                                   currently Group B)
#
# Common: VP_LAMBDA=1e-3, EPOCHS=8000, 3h wall.

module load miniforge
source activate neuralrg
mkdir -p logs

L=32
T=2.269185314213022
N=200000
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
VP_LAMBDA="${VP_LAMBDA:-1e-3}"
EPOCHS="${EPOCHS:-8000}"
CONFIG="${CONFIG:?CONFIG env var required (A_nr1/champion_nr1/champion_nr2/D_nr2)}"

case "$CONFIG" in
    A_nr1)
        FOLDER="./data/${L}Ising_T2.269_gaussian_perLayerVP${VP_LAMBDA}_nr1_b64"
        PRIOR_ARGS="-priorType gaussian"
        NREPEAT=1
        ;;
    champion_nr1)
        FOLDER="./data/${L}Ising_T2.269_hcg_perscale_fixdil_perLayerVP${VP_LAMBDA}_nr1_b64"
        PRIOR_ARGS="-priorType hierarchical_conditional_gaussian -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1"
        NREPEAT=1
        ;;
    champion_nr2)
        FOLDER="./data/${L}Ising_T2.269_hcg_perscale_fixdil_perLayerVP${VP_LAMBDA}_nr2_b64"
        PRIOR_ARGS="-priorType hierarchical_conditional_gaussian -hcgScaleShared 0 -hcgHidden 32 -hcgDilated 1 -hcgCircular 1"
        NREPEAT=2
        ;;
    D_nr2)
        FOLDER="./data/${L}Ising_T2.269_i2_stride8h32_perLayerVP${VP_LAMBDA}_nr2_b64"
        PRIOR_ARGS="-priorType conditional_gaussian -condPriorSlowStride 8 -condPriorHidden 32"
        NREPEAT=2
        ;;
    *)
        echo "Unknown CONFIG: $CONFIG"; exit 1 ;;
esac

echo "=========================================="
echo "Per-layer VP experiment: $CONFIG"
echo "  L=$L T=$T  nr=$NREPEAT  VP=$VP_LAMBDA"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  folder: $FOLDER"
echo "=========================================="
date

python -u main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" -batch 64 \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat $NREPEAT \
    -lr 1e-3 -gradClip 5.0 \
    -savePeriod 500 \
    -symmetry -skipHMC \
    -dataDriven -dataPath "$HS_PT" -noDeq \
    $PRIOR_ARGS \
    -volumePreservingWeight "$VP_LAMBDA" \
    -volumePreservingPerLayer 1

date
