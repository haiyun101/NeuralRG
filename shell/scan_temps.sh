#!/bin/bash

# 1. Initialize variables
WT_SUFFIX=""
WT_FLAG=""
HP_SUFFIX=""
HP_FLAG=""
SYM_SUFFIX="_nsym" # Default is nsym
SYM_FLAG=""

# 2. Parse optional flags
while [[ "$1" == --* ]]; do
    case "$1" in
        --wt)
            WT_SUFFIX="_WT"
            WT_FLAG="-weightTying"
            ;;
        --hp)
            HP_SUFFIX="_HP"
            HP_FLAG="-haarPrior"
            ;;
        --sym)
            SYM_SUFFIX="_sym"
            SYM_FLAG="-symmetry"
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--wt] [--hp] [--sym] temp1 temp2 ..."
            exit 1
            ;;
    esac
    shift
done

# 3. Check if any temperatures were provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [--wt] [--hp] [--sym] temp1 temp2 ..."
    exit 1
fi

temps=("$@")

for T in "${temps[@]}"
do
    # 4. Construct Job Name and Directory using all suffixes
    # Resulting order: T2.269_sym_WT_HP (depending on what you toggle)
    SUFFIX="${SYM_SUFFIX}${WT_SUFFIX}${HP_SUFFIX}"
    JOB_NAME="T${T}${SUFFIX}"
    OUT_DIR="./data/32Ising_T${T}${SUFFIX}_longer"

    mkdir -p $OUT_DIR
    echo "Submitting: $JOB_NAME"

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=3:00:00
#SBATCH --output=${OUT_DIR}/${JOB_NAME}_%j.log

module load miniforge
source activate neuralrg

python ./main.py \\
    -L 32 \\
    -T ${T} \\
    -folder ${OUT_DIR} \\
    -batch 128 \\
    -epochs 1600 \\
    -nlayers 10 \\
    -nmlp 3 \\
    -nhidden 64 \\
    -nrepeat 1 \\
    -savePeriod 10 \\
    -cuda 0 \\
    ${WT_FLAG} \\
    ${HP_FLAG} \\
    ${SYM_FLAG}
EOT
done