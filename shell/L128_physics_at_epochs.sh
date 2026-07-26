#!/bin/bash -l
#SBATCH --job-name=L128_phys_ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=./logs/L128_phys_ep_%j.out
#SBATCH --error=./logs/L128_phys_ep_%j.err

# Physics sampling from L=128 warm-start + fresh init at multiple epochs.
# Answers: does warm-start from L=64 alone give physics-meaningful samples,
# or does it need extra L=128-specific training?
#
# For each cell/epoch: draw N=1000 samples, compute χ, U₄, |M|, energy.
# Output: JSON in each folder's flow_diagnostic_epoch{N}.json.

module load miniforge
source activate neuralrg
mkdir -p logs

WARM=data/L128_T2.269_champion_from_L64
FRESH=data/L128_T2.269_champion_freshInit

# L=64 reference (for GT check on the physics-sampling pipeline)
L64_CHAMP=data/64Ising_T2.269_hsBignet_hcg_perscale_fixdil_vp1e-3_nr1_b16

# Sample sizes / batch (L=128 GPU sampling)
N=1000
B=32

echo "==================================================================="
echo "L=64 champion @ ep 9500 (reference — expect close to GT)"
echo "==================================================================="
python -u analyzers/flow_sample_diagnostic.py $L64_CHAMP \
    -n $N -b $B --seed 0 --epoch 9500 --no-png

for ep in 200 2000 5000 9800; do
  echo
  echo "==================================================================="
  echo "L=128 warm-start @ ep $ep"
  echo "==================================================================="
  python -u analyzers/flow_sample_diagnostic.py $WARM \
      -n $N -b $B --seed 0 --epoch $ep --no-png
done

echo
echo "==================================================================="
echo "L=128 fresh init @ ep 9800 (control for warm-start)"
echo "==================================================================="
python -u analyzers/flow_sample_diagnostic.py $FRESH \
    -n $N -b $B --seed 0 --epoch 9800 --no-png

echo
echo "Done."
date
