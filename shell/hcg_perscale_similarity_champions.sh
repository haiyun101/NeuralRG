#!/bin/bash -l
#SBATCH --job-name=hcg_sim_champ
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=./logs/hcg_sim_champ_%j.out
#SBATCH --error=./logs/hcg_sim_champ_%j.err

# HCG per-scale similarity probe on the fixdil+VP-1e-3 nr=1 champions.
# Produces:
#   A. weight cos-sim + L2 norm across CNN levels
#   B. per-level σ output distribution (mean, std, hist)
#   C. cross-application swap test (CNN_k applied at level k' — is HCG scale-invariant?)
#
# Snapshot epochs (nearest saved checkpoint to Best-200 rolling min):
#   L=32 champion @ ep 9500   (Best-200 ep 9401)
#   L=64 champion @ ep 13500  (Best-200 anchor, matches existing capture)
#
# Output: stdout log — parse into cross-L table for the report.

module load miniforge
source activate neuralrg
mkdir -p logs

echo "==================================================================="
echo "=== L=32 champion  (fixdil+VP-1e-3 nr=1) @ ep 9500"
echo "==================================================================="
python -u analyzers/rg_fixed_point/hcg_perscale_similarity.py \
    --N 500 --device cpu --epoch 9500 \
    --folder data/32Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_b64

echo
echo "==================================================================="
echo "=== L=64 champion  (fixdil+VP-1e-3 nr=1) @ ep 13500"
echo "==================================================================="
python -u analyzers/rg_fixed_point/hcg_perscale_similarity.py \
    --N 500 --device cpu --epoch 13500 \
    --folder data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16

echo
echo "Done."
date
