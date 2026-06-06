#!/bin/bash
#SBATCH --job-name=L32_pg_ext
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80G
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=./logs/L32_pg_ext_%j.out
#SBATCH --error=./logs/L32_pg_ext_%j.err

# Path-gradient (STL) continuation past the 5000-epoch endpoint of
# data/32Ising_T2.269_pathgrad_bignet_long (job 39351158). Submitted
# with --dependency=afterok so it fires only if the upstream STL run
# completed normally. Adam burn-in (~500 ep) matches the symmetric
# burn-in cost of the baseline extension (job 39356349), keeping the
# STL-vs-baseline asymptote comparison fair.
#
# Output: data/32Ising_T2.269_pathgrad_bignet_long_ext/

module load miniforge
source activate neuralrg

mkdir -p logs

SRC="./data/32Ising_T2.269_pathgrad_bignet_long"
DST="./data/32Ising_T2.269_pathgrad_bignet_long_ext"

mkdir -p "$DST/savings" "$DST/records"

# Seed the load with the latest STL checkpoint AND the saved
# hyperparameters (main.py reads parameters.hdf5 when -load is set,
# and the cmdline -epochs is ignored in that branch -- patch the copy
# to set the desired extension length).
LATEST=$(ls -t "$SRC/savings/"*.saving | head -1)
cp "$LATEST" "$DST/savings/"
cp "$SRC/parameters.hdf5" "$DST/parameters.hdf5"
python -c "
import h5py, numpy as np
with h5py.File('$DST/parameters.hdf5','r+') as f:
    if 'epochs' in f: del f['epochs']
    f.create_dataset('epochs', data=np.int64(5000))
print('Patched epochs -> 5000 in $DST/parameters.hdf5')
"
echo "Seeded $DST from $LATEST"

echo "=========================================="
echo "L=32 STL bignet EXTENSION (+5000 ep past ep 5000)"
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "folder: $DST"
echo "=========================================="

python -u main.py \
    -L 32 -T 2.269 \
    -folder "$DST" \
    -cuda 0 \
    -load \
    -epochs 5000 \
    -batch 128 \
    -nlayers 16 \
    -nmlp 3 \
    -nhidden 128 \
    -nrepeat 1 \
    -savePeriod 50 \
    -symmetry \
    -skipHMC \
    -pathGrad

echo "Done."
