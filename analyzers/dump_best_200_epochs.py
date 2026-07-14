"""Emit `<folder> <best_200_epoch>` lines for each variant. Consumed by
shell scripts that need to rerun tier1/diagnostic on the Best-200 checkpoint
so all downstream analysis matches the loss ranking.

Usage:  python analyzers/dump_best_200_epochs.py -L 64 -t 2.269
Output: one line per method: `<folder>\\t<best_200_epoch>\\t<S>`
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "loss"))

from loss_analyzer_fixT import (
    collect_experimental_results_for_T, get_theoretical_values, EXACT_FILE, DATA_DIR,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-L", type=int, required=True)
    ap.add_argument("-t", "--target_t", type=float, default=2.269185314213022)
    ap.add_argument("--top", type=int, default=15,
                    help="dump only top-N variants (by Best-200 S). Default 15.")
    args = ap.parse_args()

    results = collect_experimental_results_for_T(DATA_DIR, args.target_t, L=args.L)
    # Keep only forward-KL (hsBignet_*) — reverse-KL runs use a different objective
    fwd = {k: v for k, v in results.items() if k.startswith("hs")}
    # Sort by best_200_entropy ascending
    def key(kv):
        v = kv[1].get("best_200_entropy")
        return v if isinstance(v, float) else float("inf")
    rows = sorted(fwd.items(), key=key)[:args.top]
    for method, d in rows:
        folder = d.get("folder", "")
        ep = d.get("best_200_epoch", None)
        S = d.get("best_200_entropy", None)
        if folder and ep is not None and S is not None:
            print(f"{folder}\t{ep}\t{S:.2f}")
