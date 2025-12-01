#!/usr/bin/env bash
set -euo pipefail
if [ "${DEBUG:-0}" -eq 1 ]; then
    set -x
fi

self_dir=$(dirname "$(realpath "$0")")
ipsae_py_path="${self_dir}/../ipsae.py"

# Run on AF2 input files
# echo '========== Running tests on AF2 input files =========='
# af2_dir="${self_dir}/AF2"
# if test -d "${af2_dir}"; then
#     rm -rf "${af2_dir}"
# fi

# mkdir -p "${af2_dir}"
# af2_sample_id='RAF1_KSR1_MEK1_9f755'
# af2_suffix='alphafold2_multimer_v3_model_1_seed_000'

# gunzip -k "${self_dir}/${af2_sample_id}_scores_${af2_suffix}.json.gz" -c > "${af2_dir}/${af2_sample_id}_scores_${af2_suffix}.json"
# cp "${self_dir}/${af2_sample_id}_unrelaxed_${af2_suffix}.pdb" "${af2_dir}/"

# cd "${af2_dir}"
# python "${ipsae_py_path}" \
#     "${af2_sample_id}_scores_${af2_suffix}.json" \
#     "${af2_sample_id}_unrelaxed_${af2_suffix}.pdb" \
#     15 15 \
#     -o "${af2_dir}"

# cd "${self_dir}"
# fd --exact-depth 1 -t f '.*_15_15.*' -x code --diff {} AF2/{}


# Run on AF3 input files
echo '========== Running tests on AF3 input files =========='
af3_dir="${self_dir}/AF3"
if test -d "${af3_dir}"; then
    rm -rf "${af3_dir}"
fi

mkdir -p "${af3_dir}"
af3_sample_id='aurka_0_tpx2_0'

cp "${self_dir}/fold_${af3_sample_id}_model_0.cif" "${af3_dir}/"
cp "${self_dir}/fold_${af3_sample_id}_full_data_0.json" "${af3_dir}/"

cd "${af3_dir}"
python "${ipsae_py_path}" \
    "fold_${af3_sample_id}_full_data_0.json" \
    "fold_${af3_sample_id}_model_0.cif" \
    10 10 \
    -o "${af3_dir}"

cd "${self_dir}"
fd --exact-depth 1 -t f '.*_10_10.*' -x code --diff {} AF3/{}
