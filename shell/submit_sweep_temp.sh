#!/bin/bash
# Submit the temperature-sweep FSS run: 5 T x 3 L = 15 (train) + up to 12 (data).
#
# For each (L, T):
#   1. Sbatch the data-gen job   -> CPU partition (skips phases whose output exists)
#   2. Sbatch the training job   -> gpu/a100, --dependency=afterok:<data_jid>
#
# The (L, T=2.27) slot reuses the existing T=2.269185314213022 dataset (the
# exact critical temperature) rather than running fresh MCMC at T=2.27 --
# these are physically the same point. Set REUSE_TC=0 to disable.

set -euo pipefail
cd "$(dirname "$0")/.."

REUSE_TC=${REUSE_TC:-1}

LS=(8 16 32)
TS=(2.15 2.22 2.27 2.32 2.40)
TC_STR="2.269185314213022"
N=200000

echo "Temperature sweep submission"
echo "  L values:        ${LS[*]}"
echo "  T values:        ${TS[*]}"
echo "  N (samples):     $N"
echo "  Reuse T_c data:  $REUSE_TC  (T=2.27 -> $TC_STR)"
echo ""

DATA_JIDS=()
TRAIN_JIDS=()

for L in "${LS[@]}"; do
    for T_REQ in "${TS[@]}"; do

        # Decide effective T (filename T). T=2.27 reuses the existing T_c data.
        if [[ "$T_REQ" == "2.27" && "$REUSE_TC" == "1" ]]; then
            T_USE="$TC_STR"
        else
            T_USE="$T_REQ"
        fi

        HS_PT="data/mcmc_data/hs_L${L}_T${T_USE}_N${N}.pt"
        DATA_TAG="data_L${L}_T${T_USE}"
        TRAIN_TAG="tr_L${L}_T${T_USE}"

        echo "------------------------------------------------------------"
        echo "(L=$L, T_req=$T_REQ, T_use=$T_USE)"

        if [[ -f "$HS_PT" ]]; then
            echo "  HS data already exists -> no data job."
            DATA_DEP=""
        else
            DJID=$(sbatch --parsable \
                --job-name="$DATA_TAG" \
                shell/sweep_temp_data.sh "$L" "$T_USE")
            echo "  data job:  $DJID  ($DATA_TAG)"
            DATA_JIDS+=("$DJID")
            DATA_DEP="--dependency=afterok:$DJID"
        fi

        TJID=$(sbatch --parsable $DATA_DEP \
            --job-name="$TRAIN_TAG" \
            shell/sweep_temp_train.sh "$L" "$T_USE")
        echo "  train job: $TJID  ($TRAIN_TAG)${DATA_DEP:+  [waits on $DJID]}"
        TRAIN_JIDS+=("$TJID")
    done
done

echo ""
echo "============================================================"
echo "Submitted ${#DATA_JIDS[@]} data jobs and ${#TRAIN_JIDS[@]} train jobs."
echo "Data  JIDs: ${DATA_JIDS[*]:-(none -- all reused existing data)}"
echo "Train JIDs: ${TRAIN_JIDS[*]}"
echo ""
echo "Monitor: squeue -u \$USER"
