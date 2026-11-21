#!/bin/bash
#SBATCH --job-name=ipsae
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --partition=amd-hdr100
#SBATCH --output=ipsae_%A_%a.out
#SBATCH --error=ipsae_%A_%a.err

# ipsae_array.slurm
# One SLURM array task per co-folding arm. ipSAE scoring is CPU-only
# and small (numpy ops over a ~524x524 PAE), so 4 CPUs/task is plenty.
#
# Submit from the repo's cheaha/ directory:
#   cd $HOME/IPSAE_Loop1_Binders/cheaha
#   sbatch ipsae_array.slurm
#
# Edit IPSAE_REPO, WORKDIR, and the env activation line to match your layout.

set -euo pipefail

# ---------------------- paths and knobs ----------------------
WORKDIR="$HOME/Foundry/ClusterCentroids_MDresults_MatureAlpha/Equilibrated"
IPSAE_REPO="$HOME/IPSAE_Loop1_Binders"
IPSAE_PY="$IPSAE_REPO/ipsae.py"
BATCH="$IPSAE_REPO/cheaha/ipsae_batch.py"

MANIFEST="$HOME/Foundry/rfd3_manifest.csv"
OUT_DIR="$WORKDIR/ipsae_out"
PAE_CUT=10
DIST_CUT=10

mkdir -p "$OUT_DIR"

# ---------------------- environment --------------------------
# ipsae.py needs only numpy, so any env with numpy works.
# Easiest: reuse your boltz2 env, since Boltz already had numpy.
module load Anaconda3/2023.07-2
source activate boltz    # conda env with numpy; any numpy env is fine

# ---------------------- arm table ---------------------------
# index: ARMS[i] uses ROOTS[i] and, for crystal_MSA only, filters by the manifest.
ARMS=(crystal_MSA crystaltmpl c1 c2 c3)
ROOTS=(
  "$HOME/Foundry/boltz_jobs/out"
  "$WORKDIR/out_clustercrystal"
  "$WORKDIR/out_cluster1"
  "$WORKDIR/out_cluster2"
  "$WORKDIR/out_cluster3"
)

i="${SLURM_ARRAY_TASK_ID}"
ARM="${ARMS[$i]}"
ROOT="${ROOTS[$i]}"
OUT="$OUT_DIR/ipsae_${ARM}_${PAE_CUT}_${DIST_CUT}.csv"

MANIFEST_ARG=()
if [[ "$ARM" == "crystal_MSA" ]]; then
  # crystal_MSA has 921 (900 RFD3 + 21 BindCraft orphans) -- filter to the 900.
  MANIFEST_ARG=(--manifest "$MANIFEST")
fi

echo "[$ARM] root=$ROOT out=$OUT cuts=${PAE_CUT}/${DIST_CUT}"

srun python "$BATCH" \
    --arm "$ARM" \
    --root "$ROOT" \
    --ipsae-py "$IPSAE_PY" \
    --pae-cutoff "$PAE_CUT" \
    --dist-cutoff "$DIST_CUT" \
    --jobs "${SLURM_CPUS_PER_TASK:-1}" \
    "${MANIFEST_ARG[@]}" \
    --out "$OUT"

echo "[$ARM] array task $i finished."
