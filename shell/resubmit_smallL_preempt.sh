#!/bin/bash
# Resubmit L=8 and L=16 HS forward-KL training on preempt/l40, since gpu/a100
# is bottlenecked (only cc1gpu003/005 up, mixed). L=32 jobs stay on gpu/a100.
# Override partition/gres/time via sbatch CLI flags (these win over the
# #SBATCH directives in sweep_temp_train.sh).

set -euo pipefail
cd "$(dirname "$0")/.."

LS=(8 16)
TS=(2.15 2.22 2.27 2.32 2.40)
TC_STR="2.269185314213022"

JIDS=()
for L in "${LS[@]}"; do
    for T_REQ in "${TS[@]}"; do
        if [[ "$T_REQ" == "2.27" ]]; then T_USE="$TC_STR"; else T_USE="$T_REQ"; fi
        JID=$(sbatch --parsable \
            --partition=preempt \
            --gres=gpu:l40:1 \
            --time=3:00:00 \
            --job-name="tr_L${L}_T${T_USE}" \
            shell/sweep_temp_train.sh "$L" "$T_USE")
        echo "  $JID  tr_L${L}_T${T_USE}  (preempt/l40)"
        JIDS+=("$JID")
    done
done

echo ""
echo "Resubmitted ${#JIDS[@]} jobs on preempt/l40."
echo "JIDs: ${JIDS[*]}"
