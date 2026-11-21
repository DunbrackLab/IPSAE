#!/usr/bin/env python3
"""
Gather per-arm long-format ipsae CSVs (from ipsae_batch.py) into:
  (a) a merged long CSV across all arms, and
  (b) a wide per-design CSV with one row per (arm, design_id) and pivoted
      metrics: ipSAE_A_B, ipSAE_B_A, ipSAE_max, ipTM_d0chn_*, plus the
      symmetric fields (pDockQ, pDockQ2, LIS).

The wide CSV is what you want to join against binder_ptm.csv / scRMSD results.

Usage:
    python ipsae_gather.py \
        --in-dir ~/Foundry/ClusterCentroids_MDresults_MatureAlpha/Equilibrated/ipsae_out \
        --pattern 'ipsae_*_10_10.csv' \
        --out-long ipsae_all_long_10_10.csv \
        --out-wide ipsae_all_wide_10_10.csv \
        --binder-chain B
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


LONG_COPY_COLS = [
    "Chn1", "Chn2", "PAE", "Dist", "Type",
    "ipSAE", "ipSAE_d0chn", "ipSAE_d0dom",
    "ipTM_af", "ipTM_d0chn",
    "pDockQ", "pDockQ2", "LIS",
    "n0res", "n0chn", "n0dom",
    "d0res", "d0chn", "d0dom",
    "nres1", "nres2", "dist1", "dist2",
    "Model", "pae_cutoff", "dist_cutoff",
]

ASYM_METRICS = ["ipSAE", "ipSAE_d0chn", "ipSAE_d0dom", "ipTM_d0chn"]
MAX_METRICS = ["ipSAE", "ipSAE_d0chn", "ipSAE_d0dom",
               "ipTM_af", "ipTM_d0chn",
               "pDockQ", "pDockQ2", "LIS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--pattern", default="ipsae_*_10_10.csv")
    ap.add_argument("--out-long", required=True)
    ap.add_argument("--out-wide", required=True)
    ap.add_argument("--target-chain", default="A")
    ap.add_argument("--binder-chain", default="B")
    args = ap.parse_args()

    files = sorted(Path(args.in_dir).expanduser().glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matched {args.pattern} in {args.in_dir}")

    long_rows = []
    for f in files:
        with open(f) as fh:
            r = csv.DictReader(fh)
            for row in r:
                long_rows.append(row)
    print(f"read {len(long_rows)} rows from {len(files)} files")

    long_fields = ["arm", "design_id"] + LONG_COPY_COLS
    with open(args.out_long, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=long_fields,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(long_rows)
    print(f"wrote long CSV: {args.out_long}")

    # Pivot to wide: one row per (arm, design_id).
    T, B = args.target_chain, args.binder_chain
    wide = defaultdict(dict)
    for row in long_rows:
        key = (row["arm"], row["design_id"])
        wide[key].setdefault("arm", row["arm"])
        wide[key].setdefault("design_id", row["design_id"])
        wide[key].setdefault("pae_cutoff", row["pae_cutoff"])
        wide[key].setdefault("dist_cutoff", row["dist_cutoff"])
        wide[key].setdefault("nres1", row["nres1"])
        wide[key].setdefault("nres2", row["nres2"])

        rtype = row["Type"]
        c1, c2 = row["Chn1"], row["Chn2"]

        if rtype == "asym":
            # Target-aligned vs Binder-aligned
            if c1 == T and c2 == B:
                tag = f"{T}_{B}"       # target-aligned: how well target scores binder
            elif c1 == B and c2 == T:
                tag = f"{B}_{T}"       # binder-aligned: how well binder scores target
            else:
                tag = f"{c1}_{c2}"
            for m in ASYM_METRICS:
                wide[key][f"{m}_{tag}"] = row[m]
            wide[key][f"dist1_{tag}"] = row["dist1"]
            wide[key][f"dist2_{tag}"] = row["dist2"]
        elif rtype == "max":
            for m in MAX_METRICS:
                wide[key][f"{m}_max"] = row[m]
            wide[key]["dist1_max"] = row["dist1"]
            wide[key]["dist2_max"] = row["dist2"]

    # Build the wide header deterministically
    tag_TB = f"{T}_{B}"
    tag_BT = f"{B}_{T}"
    wide_fields = ["arm", "design_id", "pae_cutoff", "dist_cutoff",
                   "nres1", "nres2"]
    for m in ASYM_METRICS:
        wide_fields += [f"{m}_{tag_TB}", f"{m}_{tag_BT}"]
    wide_fields += [f"dist1_{tag_TB}", f"dist2_{tag_TB}",
                    f"dist1_{tag_BT}", f"dist2_{tag_BT}"]
    for m in MAX_METRICS:
        wide_fields.append(f"{m}_max")
    wide_fields += ["dist1_max", "dist2_max"]

    with open(args.out_wide, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=wide_fields,
                           extrasaction="ignore")
        w.writeheader()
        for key in sorted(wide):
            w.writerow(wide[key])
    print(f"wrote wide CSV: {args.out_wide} ({len(wide)} designs)")


if __name__ == "__main__":
    main()
