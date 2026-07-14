#!/bin/bash -l
#SBATCH --job-name=vp_L64_cont
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=./logs/vp_L64_cont_%j.out
#SBATCH --error=./logs/vp_L64_cont_%j.err

# Continue the fixdil+VP-1e-3 nr=2 L=64 run from its ep 7000 checkpoint
# with 36h walltime so the deep basin (log-observed F=7541.58 @ ep 7234)
# gets captured in the record file. Uses -load with same hyperparams
# (batch=8 gradAccum=2, matching the original OOM-avoidance recipe).

module load miniforge
source activate neuralrg

mkdir -p logs

FOLDER="./data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr2_b16"

# main.py's -load reads most args from parameters.hdf5, BUT `-symmetry`
# is a boolean that only picks up the CLI flag — not restored from HDF5.
# The saved checkpoint was built by a Symmetrized-wrapped model, so we
# MUST pass -symmetry here or state_dict keys don't match.
python -u main.py \
    -load -folder "$FOLDER" -cuda 0 -symmetry

echo "Done."
