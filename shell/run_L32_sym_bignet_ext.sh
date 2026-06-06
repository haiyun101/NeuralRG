#!/bin/bash
#SBATCH --job-name=L32_sym_ext
#SBATCH --partition=preempt
#SBATCH --gres=gpu:l40:1
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=./logs/L32_sym_ext_%j.out
#SBATCH --error=./logs/L32_sym_ext_%j.err

# Baseline (reverse-KL bignet) continuation past sym_bignet's 5951-epoch
# endpoint. Goal: drive baseline closer to asymptote so the STL-vs-baseline
# comparison can be done at convergence rather than at a still-drifting
# matched-epoch snapshot.
#
# Mechanism:
#   1) Copy the latest .saving from sym_bignet (ep 5950) into the new
#      _ext folder's savings/ dir.
#   2) Run main.py with -load: it loads weights from that file, then
#      trains 5000 more epochs into the _ext folder (records start at
#      ep 0 of the new run, globally ~5950..10950). Adam moments are
#      NOT preserved by -load (see project_resume_optimizer_state
#      memory), so first ~500 ep are burn-in waste -- symmetric with
#      the STL extension to keep the comparison fair.
#
# Output: data/32Ising_T2.269_sym_bignet_ext/

module load miniforge
source activate neuralrg

mkdir -p logs

SRC="./data/32Ising_T2.269_sym_bignet"
DST="./data/32Ising_T2.269_sym_bignet_ext"

mkdir -p "$DST/savings" "$DST/records"

# Seed the load with the latest baseline checkpoint AND the saved
# hyperparameters (main.py reads parameters.hdf5 when -load is set,
# and the cmdline -epochs is ignored in that branch -- we must patch
# the copy to set the desired extension length).
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
echo "L=32 sym_bignet EXTENSION (reverse-KL, +5000 ep past ep 5950)"
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
    -skipHMC

echo "Done."
