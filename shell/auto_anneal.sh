#!/bin/bash
#SBATCH --job-name=Auto_Anneal_WT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/auto_%j.log

module load miniforge
source activate neuralrg

# Params: $1="NextTemp:Epochs Remaining..." $2="CurrentFolderName"
SCHEDULE=($1)
CURRENT_DIR=$2

if [ ${#SCHEDULE[@]} -eq 0 ]; then
    echo "Annealing path complete."
    exit 0
fi

# 1. Extract the "Last Temp" from the current folder name
# This regex looks for a number like 2.6 in "32Ising_T2.6_weightTying"
LAST_TEMP=$(echo $CURRENT_DIR | grep -oP 'T\K[0-9.]+')

# 2. Parse current step info
STEP_INFO=${SCHEDULE[0]}
NEXT_TEMP=$(echo $STEP_INFO | cut -d':' -f1)
NEXT_EPOCHS=$(echo $STEP_INFO | cut -d':' -f2)

# 3. Create the new "Path-Based" folder name
# Example: 32Ising_T2.6_to_2.5_weightTying
NEW_DIR="32Ising_T${LAST_TEMP}_to_${NEXT_TEMP}_weightTying"
REMAINING_SCHEDULE="${SCHEDULE[@]:1}"

echo "========================================="
echo "Annealing: $LAST_TEMP -> $NEXT_TEMP"
echo "Folder: $NEW_DIR"
echo "========================================="

# 4. Transfer weights
# Assumes anneal_trans.sh takes: source_dir dest_dir target_temp
bash anneal_trans.sh opt/$CURRENT_DIR opt/$NEW_DIR $NEXT_TEMP

# 5. Run Training
python ./main.py \
    -L 32 \
    -T $NEXT_TEMP \
    -folder ./opt/$NEW_DIR \
    -load \
    -lr 0.0001 \
    -batch 128 \
    -epochs $NEXT_EPOCHS \
    -nlayers 10 \
    -nmlp 3 \
    -nhidden 64 \
    -nrepeat 1 \
    -savePeriod 100 \
    -symmetry \
    -cuda 0

# 6. Self-Submit next step
if [ $? -eq 0 ] && [ ${#SCHEDULE[@]} -gt 1 ]; then
    echo "Step successful. Submitting next job..."
    sbatch auto_anneal.sh "$REMAINING_SCHEDULE" "$NEW_DIR"
else
    echo "Pipeline finished or training failed."
fi