# ipSAE scoring for the ATP5A1 loop-1 co-folding arms

Cheaha-side drivers around `../ipsae.py` (DunbrackLab/IPSAE v4,
2026-01-03 build with the Boltz-2 fixes). They produce one extra
scoring metric to sit alongside `binder_ptm.csv` and scRMSD.

## What it computes

For every `(target, binder)` complex in each arm, `ipsae.py` writes a
three-row file: target-aligned asym, binder-aligned asym, and the
pairwise `max`. We collect all three directions for all designs across
all five arms into one CSV, plus a wide per-design CSV for easy joining.

Alongside `ipSAE` you also get `ipSAE_d0chn`, `ipSAE_d0dom`,
`ipTM_d0chn` (the re-scored ipTM), `pDockQ`, `pDockQ2`, and `LIS`.
Binder-aligned direction (`B_A`) is usually the most informative for
small binders against a large target; `ipSAE_max` is the safe default
scalar if you want one number per design.

## Files here

| file | what it does |
|---|---|
| `ipsae_batch.py` | drives `ipsae.py` across one arm; writes long CSV |
| `ipsae_gather.py` | merges per-arm long CSVs into one, plus a wide pivot |
| `ipsae_array.slurm` | SLURM array, one task per arm (5 tasks) |

## Inputs the batch driver assumes

`ipsae.py` auto-detects Boltz from a `.npz` PAE. Beside each
`pae_<DID>_model_0.npz` it also reads (in the same directory):

- `plddt_<DID>_model_0.npz`  (npz key `plddt`)
- `confidence_<DID>_model_0.json` (JSON key `pair_chains_iptm`)

Both are standard Boltz-2 v2.2.1 outputs and are listed in the
co-folding catalogue (§2). **Do not split these files from the PDB** —
if you tar an arm to a package, keep the four files together.

The batch driver locates predictions by `rglob('pae_*_model_0.npz')`,
so it handles both tree depths (templated arms vs the extra `<DID>/`
wrapper in `crystal_MSA`) with no configuration.

## Run it

Edit the three paths at the top of `ipsae_array.slurm`:

- `WORKDIR` — your Equilibrated dir
- `IPSAE_REPO` — where you cloned your fork
- the `source activate boltz2` line — your conda env (anything with numpy works)

Then:

```bash
cd $HOME/IPSAE_Loop1_Binders/cheaha
sbatch ipsae_array.slurm
```

The array runs five tasks (indexes 0..4 = crystal_MSA, crystaltmpl,
c1, c2, c3). `crystal_MSA` is auto-filtered to the 900-row RFD3
manifest, so BindCraft orphans are excluded.

Outputs land in `$WORKDIR/ipsae_out/`:

- `ipsae_<arm>_10_10.csv` — per-arm long CSV (3 rows per design)
- `ipsae_<arm>_10_10.failures.log` — any DIDs that `ipsae.py` failed on

### Smoke test first

Before the array, do one arm at 20 designs:

```bash
cd $HOME/IPSAE_Loop1_Binders
python cheaha/ipsae_batch.py \
    --arm c1 \
    --root $HOME/Foundry/ClusterCentroids_MDresults_MatureAlpha/Equilibrated/out_cluster1 \
    --ipsae-py $PWD/ipsae.py \
    --pae-cutoff 10 --dist-cutoff 10 \
    --jobs 4 --limit 20 \
    --out /tmp/ipsae_c1_smoke.csv
head /tmp/ipsae_c1_smoke.csv
```

If the CSV has `arm, design_id, Chn1, Chn2, ... ipSAE, ...` columns and
~60 rows (3 per design), you're good.

## Merge after the array finishes

```bash
cd $HOME/IPSAE_Loop1_Binders
python cheaha/ipsae_gather.py \
    --in-dir $HOME/Foundry/ClusterCentroids_MDresults_MatureAlpha/Equilibrated/ipsae_out \
    --pattern 'ipsae_*_10_10.csv' \
    --out-long ipsae_all_long_10_10.csv \
    --out-wide ipsae_all_wide_10_10.csv \
    --target-chain A --binder-chain B
```

`ipsae_all_wide_10_10.csv` has one row per `(arm, design_id)` with
columns including `ipSAE_A_B`, `ipSAE_B_A`, `ipSAE_max`,
`ipTM_d0chn_max`, `pDockQ_max`, `pDockQ2_max`, `LIS_max`. Join it to
`binder_ptm.csv` on `(arm, design_id)`.

## Second cutoff pair (optional)

The Dunbrack README's AF2 examples use `15 15`; its AF3 and Boltz
examples use `10 10`. If you want both, re-run the array with
`PAE_CUT=15 DIST_CUT=15` — outputs will land in a separate
`ipsae_<arm>_15_15.csv` and the per-prediction `.txt` files beside each
PDB have the cutoffs in their names, so nothing collides.

## Gotchas

- **Boltz confidence JSON**: `ipsae.py` reads `pair_chains_iptm`
  (pairwise ipTM), not `chains_ptm` (per-chain pTM). Boltz-2 v2.2.1
  writes both; if you ever see
  `Warning: 'pair_chains_iptm' key not found`, the ipTM-dependent
  columns go to 0 and `ipsae.py` exits soon after with a KeyError
  because of a post-warning re-read in the script. Our fork can patch
  that line if it ever bites, but v2.2.1 outputs are fine.
- **Outputs write next to each PDB** (`<DID>_model_0_10_10.txt`,
  `_byres.txt`, `.pml`). ~900 × 3 small files per arm. The batch driver
  skips re-running a design whose `.txt` already exists, so re-runs are
  cheap and crash-safe.
- **Model column in the output** is `pdb_stem` (`<DID>_model_0`), which
  we strip back to `<DID>` in the CSV's `design_id` column.
- **`crystal_MSA`**: the manifest filter is applied by the batch
  driver; otherwise you'd score 21 BindCraft orphans into the arm CSV.
- **Threads**: `--jobs 4` gives roughly linear speedup on the array
  tasks; one arm finishes in a few minutes on 4 CPUs, so the whole
  campaign is a single short SLURM burst.
