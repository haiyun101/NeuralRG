#!/bin/bash -l
#SBATCH --job-name=rg_v6
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=./logs/rg_v6_%j.out
#SBATCH --error=./logs/rg_v6_%j.err

# V6 CNN-offload probe: does the conditional-Gaussian prior absorb
# Ising structure (cleaning it up at the final latent), so that MERA's
# intermediate y_s shown by V5 "look non-Wilson" not because MERA learned
# a different fixed point but because the work was split MERA <-> CNN?
#
# Per cell we measure CNN strength (||μ||/||z||, σ stats), raw vs CNN-
# whitened latent distance to N(0,1), and the improvement (raw - whit).
#
# Cells: P2.x set covering A baseline, B i2-only, C nrepeat-only,
# D i2+nrepeat=2 at both L=32 and L=64.

module load miniforge
source activate neuralrg

mkdir -p logs

CELLS=(
    "L=32 baseline_b64"
    "L=32 i2_stride8h32_b64 (+I.2 cond)"
    "L=32 i2_stride8h32_nr2_b64 (P2.x D32)"
    "L=64 baseline_b16"
    "L=64 i2_stride16h32_b16 (+I.2)"
    "L=64 baseline_nr2_b16 (P2.x C64)"
    "L=64 i2_stride8h32_nr2_b16 (P2.x D64 ★)"
)

python -u analyzers/rg_fixed_point/rg_v6_cnn_offload.py \
    --N 2000 \
    --device cpu \
    --csv-out analyzers/csv/rg_v6_cnn_offload.csv \
    --cells "${CELLS[@]}"

echo "Done."
