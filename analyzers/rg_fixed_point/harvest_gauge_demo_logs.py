"""Harvest V4-like adj-scale gauge MSE from gauge_fix.py demo stdout logs.

`gauge_fix.py` printed a table of `pair / zscore MSE / gauge MSE / gauge/zscore ratio`
to stdout per folder. There are ~26 such logs in logs/gauge_demo_*.out from runs
on 25 different folders (sometimes retries). We harvest the latest log per
folder, parse the table, and write a CSV.

Output: analyzers/csv/rg_v4_gauge_demo.csv with columns
    label, folder, L, T, epoch, pair, zscore_mse, gauge_mse, ratio
"""
import csv
import glob
import os
import re
import sys


def parse_one(path):
    """Parse a single gauge_demo log; return dict of folder + table rows or None."""
    with open(path) as f:
        text = f.read()

    folder_m = re.search(r"Gauge-fix demo on (\S+)", text)
    if not folder_m:
        return None
    folder = folder_m.group(1).rstrip("/")

    L_m = re.search(r"L=(\d+),\s*T=([\d.]+),\s*ep=(\d+)", text)
    if not L_m:
        return None
    L = int(L_m.group(1))
    T = float(L_m.group(2))
    epoch = int(L_m.group(3))

    # Parse the Step-3 table block:
    #         pair   zscore MSE    gauge MSE   gauge/zscore
    # ------------------------------------------------------------
    #   f_1 → f_2       0.7189       0.5034          0.700
    rows = []
    for m in re.finditer(
        r"f_(\d+)\s*→\s*f_(\d+)\s+([0-9.]+(?:[eE][+-]?\d+)?)\s+([0-9.]+(?:[eE][+-]?\d+)?)\s+([0-9.]+(?:[eE][+-]?\d+)?)",
        text,
    ):
        s, sp1 = int(m.group(1)), int(m.group(2))
        z_mse = float(m.group(3))
        g_mse = float(m.group(4))
        ratio = float(m.group(5))
        rows.append((f"f_{s}->f_{sp1}", z_mse, g_mse, ratio))

    if not rows:
        return None
    return dict(folder=folder, L=L, T=T, epoch=epoch, rows=rows, path=path)


def best_for_folder(parsed):
    """Pick the latest (max epoch) parsed log per folder."""
    by_folder = {}
    for p in parsed:
        f = p["folder"]
        if f not in by_folder or p["epoch"] > by_folder[f]["epoch"]:
            by_folder[f] = p
    return list(by_folder.values())


def label_from_folder(folder):
    """Map folder path → human-readable label (mirroring FOLDERS dict)."""
    base = os.path.basename(folder)
    L_m = re.search(r"^(\d+)Ising", base)
    L = int(L_m.group(1)) if L_m else 0
    if "hs_dataDriven" in base:
        T_m = re.search(r"T([\d.]+)_hs_dataDriven", base)
        T = T_m.group(1) if T_m else "?"
        if T == "2.15" or T == "2.4":
            kind = "low T, ordered" if T == "2.15" else "high T, disorder"
            return f"T = {T}  ({kind})"
        return f"T = 2.269 (T_c, hs_dataDriven)"
    if "sym_bignet" in base: return "T_c sym_bignet (rev-KL)"
    if "pathgrad" in base:   return "T_c pathgrad_bignet_long_ext (STL)"
    if "hs_bignet" in base and "hsBignet" not in base:
        return "T = 2.269 (T_c, hs_bignet)"
    # Phase-1 / Phase-2 hsBignet variants
    rest = base.replace(f"{L}Ising_T2.269_hsBignet_", "")
    # Add a "Phase-2" marker for the new capacity scan stride×hidden cells
    is_phase2 = bool(re.search(r"i2_stride\d+h\d+_b(16|64)$", rest)) and "stride8h32_b64" not in rest and "stride16h32_b16" not in rest
    suffix = " (Phase-2)" if is_phase2 else ""
    # Decorate known Phase-1 names
    if rest.startswith("iii1"):    suffix = " (+III.1)" if "lam1.0" in rest and "combined" not in rest else ""
    if rest.startswith("i2_stride8h32_b64"): suffix = " (+I.2 cond)"
    if rest.startswith("i2_stride16h32_b16"): suffix = " (+I.2)"
    if rest.startswith("combined"): suffix = " (I.2+III.1)"
    if rest.startswith("i1_df4.0_b16"): suffix = " (Student-t)"
    if rest.startswith("i1_df4.0_b128"): suffix = " (Student-t b128)"
    if rest.startswith("i2_stride") and "b128" in rest: suffix = " (b128)"
    label_core = rest.replace("_b16","_b16").replace("_b64","_b64").replace("_b128","_b128")
    return f"L={L} {label_core}{suffix}"


def main():
    logs = sorted(glob.glob("logs/gauge_demo_*.out"))
    print(f"found {len(logs)} demo logs", flush=True)

    parsed = [p for p in (parse_one(L) for L in logs) if p is not None]
    print(f"  parsed OK: {len(parsed)}", flush=True)

    chosen = best_for_folder(parsed)
    print(f"  unique folders: {len(chosen)}", flush=True)

    out_path = "analyzers/csv/rg_v4_gauge_demo.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "folder", "L", "T", "epoch",
                    "pair", "zscore_mse", "gauge_mse", "ratio"])
        for r in sorted(chosen, key=lambda x: x["folder"]):
            label = label_from_folder(r["folder"])
            for pair, zm, gm, ratio in r["rows"]:
                w.writerow([label, r["folder"], r["L"], r["T"],
                            r["epoch"], pair, zm, gm, ratio])
    print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
