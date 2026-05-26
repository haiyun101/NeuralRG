# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Tufts HPC cluster. No GPU on the login node — submit GPU jobs via Slurm.

```bash
module load miniforge
source activate neuralrg   # PyTorch, h5py, numba
```

GPU jobs: `--partition=gpu --gres=gpu:a100:1`. Results land in `./data/<run_name>/` or `./opt/<run_name>/`.

## Running Training

**Standard energy-based (reverse KL):**
```bash
python main.py -L 32 -T 2.269 -cuda 0 -epochs 80000 -batch 128 \
  -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 -savePeriod 100 \
  -symmetry -skipHMC -folder ./opt/MyRun/
```

**Data-driven (forward KL / MLE — requires pre-generated MCMC data):**
```bash
# Step 1: generate dataset (CPU, uses Numba-JIT Wolff algorithm)
python generate_mcmc_data.py -L 32 -T 2.269 -N 50000
# outputs: ./data/mcmc_data/mcmc_wolff_L32_T2.269_N50000.pt

# Step 2: train
python main.py -L 32 -T 2.269 -dataDriven -cuda 0 -epochs 500000 \
  -batch 128 -nlayers 10 -nmlp 3 -nhidden 64 -nrepeat 1 \
  -savePeriod 100 -symmetry -skipHMC -folder ./opt/MyRun/
```
If `-dataPath` is omitted, `main.py` auto-searches `./data/mcmc_data/mcmc_wolff_L{L}_T{T}_N*.pt`.

**Resume from checkpoint:**
```bash
python main.py -load -folder ./opt/MyRun/ [same flags as original run]
```

**Slurm wrappers** in `shell/`:
- `shell/run_data_driven.sh` — single data-driven job
- `shell/scan_temps.sh [--wt] [--hp] [--sym] T1 T2 ...` — sweep over temperatures, one Slurm job per T; supports `--wt` (weight tying), `--hp` (Haar prior), `--sym` (Z2 symmetry)
- `shell/run_neuralrg32_crit.sh` and variants — pre-configured L=32 energy-based jobs

**Multi-GPU hyperparameter sweep** (CPU process pool):
```bash
# Edit setting.py to configure parameters and GPU assignments, then:
python core.py
```

## Running Tests

```bash
# From repo root (no test runner configured — run directly)
python test/test_mera.py
python test/test_rnvp.py
python test/test_symmetry.py
# etc.
```

## Architecture

```
main.py               Entry point: parses args, builds model, calls learnInterface
generate_mcmc_data.py Wolff-cluster MCMC sampler (Numba JIT) → saves .pt datasets

source/               Target distributions
  ising.py            2D Ising model: Gaussian-approximation log-probability + HMC energy
  gaussian.py         Isotropic Gaussian (latent prior)

flow/                 Normalizing flow building blocks
  rnvp.py             Real-NVP affine coupling layer
  flow.py             Base Flow class (forward/inverse/logProbability/sample)
  hierarchy/
    mera.py           MERA: applies RNVP blocks at successive coarse-graining scales
    template.py       HierarchyBijector: im2col-based patch extraction + reassembly
    im2col.py         Index computation for hierarchical patch access

train/
  learn.py            symmetryMERAInit() builds the flow;
                      learnInterface() runs the training loop (reverse-KL or MLE)
                      HaarRNVP wrapper applies Haar majority-vote prior
  symmetry.py         Symmetrized wrapper (averages over Z2 spin-flip symmetry)

utils/
  layers/             MLP, CNN, ScalableTanh, squeezing utilities
  mc/                 HMC and Metropolis samplers
  savefolder.py       createWorkSpace() — creates opt/<run>/savings/ and records/

analyzers/            Post-training analysis scripts (standalone, not imported)
  plot_dist_hdf5.py   Distribution plots + physics observables from HDF5 records
  loss_report.py      Comparison of training loss vs exact free energy (etc/exactz.md)
  MCMC_Wolff.py       Reference MCMC for comparison
  calc_exact_loss.py  Exact partition-function loss calculator

setting.py / core.py  Multi-GPU sweep: setting.py configures commands/parameters,
                      core.py runs them in parallel via multiprocessing.Pool

shell/                Slurm job scripts
etc/exactz.md         Exact 2D Ising free energies for L=4..64 (reference values)
```

## Training Loop Details

`learnInterface` in `train/learn.py` handles both modes:

- **Reverse KL** (default): samples from the flow, minimizes `KL(q||p) = E_q[log q - log p]`. The `alpha` flag adds a Z2 symmetry-breaking penalty.
- **Forward KL / MLE** (`-dataDriven`): loads MCMC data, minimizes `-E_data[log q(x)]`. Dequantization noise (±0.1) is added to smooth discrete spin states. Optional alpha-weighted symmetry penalty also applies.

Checkpoints saved to `<folder>/savings/<name>Saving_epoch{N}.saving` (PyTorch state dict) and `<folder>/records/<name>Record_epoch{N}.hdf5` (loss/energy/entropy/HMC stats).

## Physical Priors

Two optional architectural priors controlled by flags:
- `-weightTying`: all MERA layers share a single RNVP parameter set (scale invariance)
- `-haarPrior`: wraps each RNVP block with a fixed 4×4 Haar orthogonal transform before the coupling, separating slow (majority-vote) and fast modes
- `-symmetry`: wraps the full flow in `Symmetrized` to enforce Z2 (spin-flip) symmetry at inference

These can be combined. See `train/learn.py:symmetryMERAInit()` for construction logic.

## Output Structure

Each run folder contains:
```
parameters.hdf5           Hyperparameters (written once at start, used for -load)
proposals_{epoch}.png     Sample spin configurations
savings/                  Model checkpoints (.saving files)
records/                  HDF5 files with loss/energy/entropy/HMC acceptance arrays
```
