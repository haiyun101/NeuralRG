#!/bin/bash -l
#SBATCH --job-name=vpL64_fixdil_nr2_iS
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=./logs/vpL64_fixdil_nr2_iS_%j.out
#SBATCH --error=./logs/vpL64_fixdil_nr2_iS_%j.err

# L=64 fixdil nr=2 + VP, WARM-STARTED from the trained shared nr=2 checkpoint.
# Rationale: cold-start nr=2 fixdil+VP ranks below cold-start nr=1 fixdil+VP
# on Best-200 (7690-7716 vs 7658). Warm-starting the per-scale CNNs from
# shared's trained CNN gives them a scale-invariant basin to start from; VP
# then applies pressure to keep the flow near volume-preserving. Isolates
# whether the nr=2 gap was compute-limited (from-scratch) or capacity-mismatched.
#
# Uses main.py's existing -hcgInitFromShared plumbing:
#   - loads shared nr=2's MERA + Symmetrized state (10.9 M params) directly
#   - duplicates shared CNN weights into each per-scale CNN slot
#   - expands shared Adam moments to all per-scale CNN slots
# Then applies -volumePreservingWeight to the training loss.
#
# Env vars:
#   VP_LAMBDA  (required, e.g. 1e-3)
#   EPOCHS     (default 15000)

module load miniforge
source activate neuralrg

mkdir -p logs

L=64
T="2.269185314213022"
N="200000"
HS_PT="data/mcmc_data/hs_L${L}_T${T}_N${N}.pt"
VP_LAMBDA="${VP_LAMBDA:?VP_LAMBDA env var required (e.g. 1e-3, 1e-4)}"
EPOCHS="${EPOCHS:-15000}"
# batch=8 gradAccum=2 (effective batch 16, same as nr=1) — halved batch
# fits nr=2 activations in 40 GB A100; batch=16 nr=2 OOMs (verified by
# job 41798360 earlier this session).
BATCH="${BATCH:-8}"
GRADACCUM="${GRADACCUM:-2}"

SHARED_FOLDER="./data/64Ising_T2.269_hsBignet_hcg_shared_nr2_b16"
if [ ! -d "$SHARED_FOLDER/savings" ]; then
    echo "ERROR: shared nr=2 checkpoint folder missing: $SHARED_FOLDER"
    exit 1
fi

FOLDER="./data/${L}Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp${VP_LAMBDA}_initshared_nr2_b${BATCH}"
mkdir -p "$FOLDER"

echo "=========================================="
echo "L=$L fixdil nr=2 + VP-${VP_LAMBDA}, warm-start from shared nr=2"
echo "  Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "  init-from-shared source: $SHARED_FOLDER"
echo "  target folder:           $FOLDER"
echo "=========================================="

python -u main.py \
    -L $L -T $T \
    -folder "$FOLDER" \
    -cuda 0 \
    -epochs "$EPOCHS" \
    -batch "$BATCH" \
    -gradAccum "$GRADACCUM" \
    -nlayers 16 -nmlp 3 -nhidden 128 -nrepeat 2 \
    -lr 1e-3 \
    -gradClip 5.0 \
    -savePeriod 500 \
    -symmetry \
    -skipHMC \
    -dataDriven \
    -dataPath "$HS_PT" \
    -noDeq \
    -priorType hierarchical_conditional_gaussian \
    -hcgScaleShared 0 \
    -hcgHidden 32 \
    -hcgDilated 1 \
    -hcgCircular 1 \
    -hcgInitFromShared "$SHARED_FOLDER" \
    -volumePreservingWeight "$VP_LAMBDA"

echo "Done."
