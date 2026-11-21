#!/usr/bin/env python3
"""
Batch driver for DunbrackLab/IPSAE across a Boltz-2 co-folding arm.

Walks an arm's output root, locates every (pae_<DID>_model_0.npz, <DID>_model_0.pdb)
pair, calls ipsae.py on it, parses the resulting <stem>_<PAE>_<DIST>.txt, and writes
one long-format CSV covering every chain-pair row for the arm.

Handles the two different Boltz output tree depths (templated vs untemplated) via
rglob on the file name, so no path assumptions are needed.

Idempotent: if the per-prediction .txt already exists it is re-parsed, not re-run.
Pass --no-skip-existing to force a re-run.

Usage:
    python ipsae_batch.py --arm c1 \
        --root ~/Foundry/ClusterCentroids_MDresults_MatureAlpha/Equilibrated/out_cluster1 \
        --ipsae-py ~/IPSAE_Loop1_Binders/ipsae.py \
        --pae-cutoff 10 --dist-cutoff 10 \
        --out ipsae_c1_10_10.csv \
        --jobs 4
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


IPSAE_HEADER = [
    "Chn1", "Chn2", "PAE", "Dist", "Type",
    "ipSAE", "ipSAE_d0chn", "ipSAE_d0dom",
    "ipTM_af", "ipTM_d0chn",
    "pDockQ", "pDockQ2", "LIS",
    "n0res", "n0chn", "n0dom",
    "d0res", "d0chn", "d0dom",
    "nres1", "nres2", "dist1", "dist2",
    "Model",
]

PAE_RE = re.compile(r"^pae_(.+?)_model_0\.npz$")


def find_predictions(root: Path):
    """Yield (design_id, pae_path, structure_path) under root (any depth)."""
    for pae in root.rglob("pae_*_model_0.npz"):
        m = PAE_RE.match(pae.name)
        if not m:
            continue
        did = m.group(1)
        pdb = pae.with_name(f"{did}_model_0.pdb")
        if not pdb.exists():
            cif = pae.with_name(f"{did}_model_0.cif")
            if not cif.exists():
                continue
            pdb = cif
        yield did, pae, pdb


def parse_ipsae_txt(txt_path: Path):
    """Return list of dict rows from an ipsae.py chain-pair output file."""
    rows = []
    with open(txt_path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("Chn1"):
                continue
            parts = s.split()
            if len(parts) < len(IPSAE_HEADER):
                continue
            rows.append(dict(zip(IPSAE_HEADER, parts[:len(IPSAE_HEADER)])))
    return rows


def run_one(ipsae_py: str, pae: Path, pdb: Path,
            pae_cut: float, dist_cut: float,
            skip_existing: bool):
    """Run ipsae.py once and return (parsed_rows, err_text_or_None)."""
    pdb_stem = pdb.stem  # e.g. <DID>_model_0
    out_txt = pdb.parent / f"{pdb_stem}_{int(pae_cut):02d}_{int(dist_cut):02d}.txt"

    if not (skip_existing and out_txt.exists()):
        proc = subprocess.run(
            [sys.executable, ipsae_py, str(pae), str(pdb),
             str(pae_cut), str(dist_cut)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not out_txt.exists():
            return None, (proc.stdout + "\n--STDERR--\n" + proc.stderr)[-1200:]

    try:
        return parse_ipsae_txt(out_txt), None
    except Exception as exc:  # defensive; parser is simple
        return None, f"parse error: {exc}"


def _worker(args_tuple):
    did, pae, pdb, ipsae_py, pae_cut, dist_cut, skip_existing = args_tuple
    rows, err = run_one(ipsae_py, pae, pdb, pae_cut, dist_cut, skip_existing)
    return did, str(pae), rows, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="Label for this arm (written to csv)")
    ap.add_argument("--root", required=True, help="Arm output root (recursed by name)")
    ap.add_argument("--ipsae-py", required=True, help="Path to ipsae.py")
    ap.add_argument("--pae-cutoff", type=float, default=10.0)
    ap.add_argument("--dist-cutoff", type=float, default=10.0)
    ap.add_argument("--out", required=True, help="Long-format output CSV")
    ap.add_argument("--manifest", help="Optional CSV with design_id column; filter to these")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-skip-existing", action="store_true")
    ap.add_argument("--fail-log", default=None)
    ap.add_argument("--limit", type=int, default=0, help="If >0, process only this many (for smoke test)")
    args = ap.parse_args()

    ipsae_py = os.path.abspath(args.ipsae_py)
    if not os.path.exists(ipsae_py):
        sys.exit(f"ipsae.py not found at {ipsae_py}")
    root = Path(args.root).expanduser()
    if not root.exists():
        sys.exit(f"arm root does not exist: {root}")

    manifest_ids = None
    if args.manifest:
        with open(args.manifest) as fh:
            rdr = csv.DictReader(fh)
            if "design_id" not in rdr.fieldnames:
                sys.exit(f"manifest {args.manifest} has no design_id column")
            manifest_ids = {r["design_id"] for r in rdr}

    preds = list(find_predictions(root))
    if manifest_ids is not None:
        before = len(preds)
        preds = [p for p in preds if p[0] in manifest_ids]
        print(f"[{args.arm}] filtered by manifest: {before} -> {len(preds)}", flush=True)
    else:
        print(f"[{args.arm}] found {len(preds)} predictions under {root}", flush=True)

    if args.limit > 0:
        preds = preds[:args.limit]
        print(f"[{args.arm}] --limit truncated to {len(preds)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail_log = Path(args.fail_log) if args.fail_log else out_path.with_suffix(".failures.log")

    csv_cols = ["arm", "design_id"] + IPSAE_HEADER + ["pae_cutoff", "dist_cutoff"]
    skip_existing = not args.no_skip_existing

    n_ok = n_fail = n_rows = 0
    with open(out_path, "w", newline="") as fout, open(fail_log, "w") as flog:
        w = csv.DictWriter(fout, fieldnames=csv_cols)
        w.writeheader()

        payload = [(did, pae, pdb, ipsae_py,
                    args.pae_cutoff, args.dist_cutoff, skip_existing)
                   for (did, pae, pdb) in preds]

        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                for did, pae_str, rows, err in ex.map(_worker, payload, chunksize=4):
                    n_ok, n_fail, n_rows = _emit(
                        did, pae_str, rows, err, w, flog, args, n_ok, n_fail, n_rows)
        else:
            for item in payload:
                did, pae_str, rows, err = _worker(item)
                n_ok, n_fail, n_rows = _emit(
                    did, pae_str, rows, err, w, flog, args, n_ok, n_fail, n_rows)

    print(f"[{args.arm}] done. ok={n_ok} fail={n_fail} rows={n_rows}", flush=True)
    print(f"[{args.arm}] CSV: {out_path}", flush=True)
    if n_fail:
        print(f"[{args.arm}] failures logged to: {fail_log}", flush=True)


def _emit(did, pae_str, rows, err, writer, flog, args,
          n_ok, n_fail, n_rows):
    if err or not rows:
        flog.write(f"{did}\t{pae_str}\t{(err or 'empty_output').replace(chr(10), ' | ')[:600]}\n")
        flog.flush()
        return n_ok, n_fail + 1, n_rows
    for row in rows:
        writer.writerow({
            "arm": args.arm, "design_id": did,
            "pae_cutoff": args.pae_cutoff,
            "dist_cutoff": args.dist_cutoff,
            **row,
        })
    return n_ok + 1, n_fail, n_rows + len(rows)


if __name__ == "__main__":
    main()
