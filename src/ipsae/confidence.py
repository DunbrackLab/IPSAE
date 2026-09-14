# confidence.py
# Loading of AF2 / AF3 / Boltz confidence data (PAE matrices, pLDDT, ipTM) for the ipsae package.
#
# Derived from ipsae.py by Roland Dunbrack, Fox Chase Cancer Center.
# https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2
# MIT license: script can be modified and redistributed for non-commercial and
# commercial use, as long as this information is reproduced.

import gzip
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .utils import init_chainpairdict_zeros

MODEL_TYPES = ("af2", "af3", "boltz", "esmfold2")


def load_json_file(path):
    """Load a JSON file, transparently handling gzip-compressed ``.json.gz`` files."""
    if path.endswith(".gz"):
        with gzip.open(path, 'rt') as file:
            return json.load(file)
    with open(path, 'r') as file:
        return json.load(file)


def _sibling_path(path, old, new):
    """Replace ``old`` with ``new`` in the basename of ``path`` only.

    Keeps directory names untouched so that e.g. a folder named ``pae`` is not
    rewritten when deriving the Boltz pLDDT/confidence file names.
    """
    directory, basename = os.path.split(path)
    return os.path.join(directory, basename.replace(old, new))


@dataclass
class ConfidenceData:
    """PAE matrix, pLDDT values, and AlphaFold/Boltz ipTM values for one model."""

    model_type: str                 # 'af2', 'af3', 'boltz', or 'esmfold2'
    pae_matrix: np.ndarray          # numres x numres residue PAE matrix
    plddt: np.ndarray               # per-residue pLDDT (CA atoms)
    cb_plddt: np.ndarray            # per-residue pLDDT (CB atoms; for pDockQ)
    iptm_af2: float = -1.0          # AF2 whole-complex ipTM (same for all chain pairs)
    ptm_af2: float = -1.0           # AF2 whole-complex pTM
    iptm_pairs: Optional[Dict] = field(default=None)  # AF3/Boltz chain-pair ipTM values

    def get_iptm_af(self, chain1, chain2):
        """Return the AlphaFold/Boltz-reported ipTM for a chain pair."""
        if self.model_type == "af2":
            return self.iptm_af2
        if self.iptm_pairs is not None:
            return self.iptm_pairs[chain1][chain2]
        return 0.0


def load_af2_confidence(pae_file_path, structure):
    """Load AF2 confidence data from a scores ``.json``/``.json.gz`` or ``.pkl`` file."""
    if not os.path.exists(pae_file_path):
        raise FileNotFoundError(f"AF2 PAE file does not exist: {pae_file_path}")

    if pae_file_path.endswith('.pkl'):
        data = np.load(pae_file_path, allow_pickle=True)
    else:
        data = load_json_file(pae_file_path)

    if 'iptm' in data:
        iptm_af2 = float(data['iptm'])
    else:
        iptm_af2 = -1.0
    if 'ptm' in data:
        ptm_af2 = float(data['ptm'])
    else:
        ptm_af2 = -1.0

    if 'plddt' in data:
        plddt = np.array(data['plddt'])
        cb_plddt = np.array(data['plddt'])  # for pDockQ
    else:
        plddt = np.zeros(structure.numres)
        cb_plddt = np.zeros(structure.numres)

    if 'pae' in data:
        pae_matrix = np.array(data['pae'])
    elif 'predicted_aligned_error' in data:
        pae_matrix = np.array(data['predicted_aligned_error'])
    else:
        raise ValueError(f"No PAE data ('pae' or 'predicted_aligned_error') in AF2 file: {pae_file_path}")

    return ConfidenceData(
        model_type='af2',
        pae_matrix=pae_matrix,
        plddt=plddt,
        cb_plddt=cb_plddt,
        iptm_af2=iptm_af2,
        ptm_af2=ptm_af2,
    )


def load_af3_confidence(pae_file_path, structure):
    """Load AF3 confidence data from a full-data/confidences ``.json``/``.json.gz`` file.

    The chain-pair ipTM matrix is read from the companion summary confidences
    file when it can be found next to the PAE file.
    """
    # Example AlphaFold3 server filenames
    #   fold_aurka_0_tpx2_0_full_data_0.json
    #   fold_aurka_0_tpx2_0_summary_confidences_0.json
    #   fold_aurka_0_tpx2_0_model_0.cif
    # Example AlphaFold3 downloadable code filenames
    #   confidences.json
    #   summary_confidences.json
    #   model1.cif
    if not os.path.exists(pae_file_path):
        raise FileNotFoundError(f"AF3 PAE file does not exist: {pae_file_path}")

    data = load_json_file(pae_file_path)

    if "atom_plddts" in data:
        atom_plddts = np.array(data['atom_plddts'])
        plddt = atom_plddts[structure.CA_atom_num]      # residue plddts from Calpha atoms
        cb_plddt = atom_plddts[structure.CB_atom_num]   # residue plddts from Cbeta atoms for pDockQ
    else:
        plddt = np.zeros(structure.numres)
        cb_plddt = np.zeros(structure.numres)

    # Get pairwise residue PAE matrix by identifying one token per protein residue.
    # Modified residues have separate tokens for each atom, so need to pull out Calpha atom as token.
    # Skip ligands.
    if 'pae' in data:
        pae_matrix_af3 = np.array(data['pae'])
    else:
        raise ValueError(f"No PAE data in AF3 json file: {pae_file_path}")

    token_array = structure.token_array
    pae_matrix = pae_matrix_af3[np.ix_(token_array.astype(bool), token_array.astype(bool))]

    # Get iptm matrix from AF3 summary_confidences file
    unique_chains = structure.unique_chains
    iptm_af3 = init_chainpairdict_zeros(unique_chains)

    summary_file_path = None
    if "confidences" in os.path.basename(pae_file_path):
        summary_file_path = _sibling_path(pae_file_path, "confidences", "summary_confidences")
    elif "full_data" in os.path.basename(pae_file_path):
        summary_file_path = _sibling_path(pae_file_path, "full_data", "summary_confidences")

    if summary_file_path is not None and not os.path.exists(summary_file_path):
        # allow mixed compressed/uncompressed PAE and summary files
        alternate = summary_file_path[:-3] if summary_file_path.endswith(".gz") else summary_file_path + ".gz"
        if os.path.exists(alternate):
            summary_file_path = alternate

    if summary_file_path is not None and os.path.exists(summary_file_path):
        data_summary = load_json_file(summary_file_path)
        af3_chain_pair_iptm_data = data_summary['chain_pair_iptm']
        for nchain1, chain1 in enumerate(unique_chains):
            for nchain2, chain2 in enumerate(unique_chains):
                if chain1 == chain2:
                    continue
                iptm_af3[chain1][chain2] = af3_chain_pair_iptm_data[nchain1][nchain2]
    else:
        print("AF3 summary file does not exist: ", summary_file_path)

    return ConfidenceData(
        model_type='af3',
        pae_matrix=pae_matrix,
        plddt=plddt,
        cb_plddt=cb_plddt,
        iptm_pairs=iptm_af3,
    )


def _extract_esmfold2_plddt(data, structure, pae_file_path):
    """Extract per-residue pLDDT for ESMfold2 from residue- or atom-level arrays."""
    if not isinstance(data, dict):
        return np.zeros(structure.numres), np.zeros(structure.numres)

    if "plddt" not in data:
        return np.zeros(structure.numres), np.zeros(structure.numres)

    raw_plddt = np.array(data["plddt"])
    if raw_plddt.ndim != 1:
        raise ValueError(f"ESMfold2 pLDDT must be a 1D array in: {pae_file_path}")

    # Convert normalized [0,1] values to [0,100] when needed (NaN-tolerant).
    finite_plddt = raw_plddt[np.isfinite(raw_plddt)]
    if finite_plddt.size > 0 and np.max(finite_plddt) <= 1.0:
        raw_plddt = 100.0 * raw_plddt

    if raw_plddt.shape[0] == structure.numres:
        plddt = raw_plddt
        cb_plddt = raw_plddt
    elif raw_plddt.shape[0] > np.max(structure.CB_atom_num):
        plddt = raw_plddt[structure.CA_atom_num]
        cb_plddt = raw_plddt[structure.CB_atom_num]
    else:
        raise ValueError(
            f"ESMfold2 pLDDT length ({raw_plddt.shape[0]}) does not match the number of residues "
            f"({structure.numres}) and is too short for atom indexing in: {pae_file_path}")

    return plddt, cb_plddt


def _extract_esmfold2_pae(data, structure, pae_file_path):
    """Extract an ESMfold2 residue-residue PAE matrix and align to structure residues."""
    if isinstance(data, list):
        pae_raw = np.array(data)
    elif isinstance(data, dict):
        if "predicted_aligned_error" in data:
            pae_raw = np.array(data["predicted_aligned_error"])
        elif "pae" in data:
            pae_raw = np.array(data["pae"])
        else:
            raise ValueError(
                f"No PAE data ('predicted_aligned_error' or 'pae') in ESMfold2 file: {pae_file_path}")
    else:
        raise ValueError(
            f"Unsupported top-level JSON type for ESMfold2 PAE file (expected list or dict): {pae_file_path}")

    if pae_raw.shape == (structure.numres, structure.numres):
        return pae_raw

    token_array = structure.token_array
    if pae_raw.shape == (len(token_array), len(token_array)):
        return pae_raw[np.ix_(token_array.astype(bool), token_array.astype(bool))]

    raise ValueError(
        f"ESMfold2 PAE matrix shape {pae_raw.shape} does not match residues "
        f"({structure.numres}, {structure.numres}) or token-array shape "
        f"({len(token_array)}, {len(token_array)}) for: {pae_file_path}")


def load_esmfold2_confidence(pae_file_path, structure):
    """Load ESMfold2 confidence data from a PAE ``.json``/``.json.gz`` file.

    ESMfold2 outputs may provide ``predicted_aligned_error`` (or ``pae``) and
    residue-level ``plddt`` values; pairwise chain ipTM values are not expected.
    """
    if not os.path.exists(pae_file_path):
        raise FileNotFoundError(f"ESMfold2 PAE file does not exist: {pae_file_path}")

    data = load_json_file(pae_file_path)
    plddt, cb_plddt = _extract_esmfold2_plddt(data, structure, pae_file_path)
    pae_matrix = _extract_esmfold2_pae(data, structure, pae_file_path)

    unique_chains = structure.unique_chains
    iptm_esmfold2 = init_chainpairdict_zeros(unique_chains)

    return ConfidenceData(
        model_type='esmfold2',
        pae_matrix=pae_matrix,
        plddt=plddt,
        cb_plddt=cb_plddt,
        iptm_pairs=iptm_esmfold2,
    )


def detect_cif_json_model_type(pae_file_path):
    """Detect model type for ``.cif`` + JSON inputs ('af3' vs 'esmfold2')."""
    basename = os.path.basename(pae_file_path).lower()
    if "full_data" in basename or "confidences" in basename:
        return "af3"
    if "esmfold" in basename:
        return "esmfold2"

    if not os.path.exists(pae_file_path):
        raise FileNotFoundError(f"PAE file does not exist: {pae_file_path}")

    data = load_json_file(pae_file_path)
    if isinstance(data, list):
        return "esmfold2"
    if isinstance(data, dict):
        if "atom_plddts" in data or "chain_pair_iptm" in data or "atom_chain_ids" in data:
            return "af3"
        if "predicted_aligned_error" in data:
            return "esmfold2"
        if "pae" in data:
            return "esmfold2"
        raise ValueError(
            f"Cannot detect model type from .cif + JSON schema in: {pae_file_path}; "
            "use --model {af2,af3,boltz2,esmfold2}.")
    raise ValueError(
        f"Unsupported top-level JSON type for model detection in: {pae_file_path}; "
        "use --model {af2,af3,boltz2,esmfold2}.")


def load_boltz_confidence(pae_file_path, structure):
    """Load Boltz1/2 confidence data from ``pae_*.npz`` plus companion files.

    The pLDDT vector is read from the sibling ``plddt_*.npz`` file and the
    chain-pair ipTM values from the sibling ``confidence_*.json`` file when
    they are present; otherwise zeros are used.
    """
    # Boltz filenames:
    # AURKA_TPX2_model_0.cif
    # confidence_AURKA_TPX2_model_0.json
    # pae_AURKA_TPX2_model_0.npz
    # plddt_AURKA_TPX2_model_0.npz
    token_array = structure.token_array
    ntokens = structure.ntokens

    plddt_file_path = _sibling_path(pae_file_path, "pae", "plddt")
    if os.path.exists(plddt_file_path):
        data_plddt = np.load(plddt_file_path)

        raw_plddt = data_plddt['plddt']
        # Only multiply by 100 if the max value is <= 1.0 (meaning it's normalized)
        if np.max(raw_plddt) <= 1.0:
            plddt_boltz = np.array(100.0 * raw_plddt)
        else:
            plddt_boltz = np.array(raw_plddt)

        plddt = plddt_boltz[np.ix_(token_array.astype(bool))]
        cb_plddt = plddt_boltz[np.ix_(token_array.astype(bool))]
    else:
        plddt = np.zeros(ntokens)
        cb_plddt = np.zeros(ntokens)

    if not os.path.exists(pae_file_path):
        raise FileNotFoundError(f"Boltz PAE file does not exist: {pae_file_path}")

    data_pae = np.load(pae_file_path)
    pae_matrix_boltz = np.array(data_pae['pae'])
    pae_matrix = pae_matrix_boltz[np.ix_(token_array.astype(bool), token_array.astype(bool))]

    summary_file_path = _sibling_path(pae_file_path, "pae", "confidence")
    summary_file_path = summary_file_path.replace(".npz", ".json")
    unique_chains = structure.unique_chains
    iptm_boltz = init_chainpairdict_zeros(unique_chains)
    if os.path.exists(summary_file_path):
        data_summary = load_json_file(summary_file_path)

        if 'pair_chains_iptm' in data_summary:
            boltz_chain_pair_iptm_data = data_summary['pair_chains_iptm']
            for nchain1, chain1 in enumerate(unique_chains):
                for nchain2, chain2 in enumerate(unique_chains):
                    if chain1 == chain2:
                        continue
                    iptm_boltz[chain1][chain2] = boltz_chain_pair_iptm_data.get(str(nchain1), {}).get(str(nchain2), 0)
        else:
            # Boltz missing key fallback
            print(f"Warning: 'pair_chains_iptm' key not found in {summary_file_path}. ipTM scores will be 0.")
    else:
        print("Boltz summary file does not exist: ", summary_file_path)

    return ConfidenceData(
        model_type='boltz',
        pae_matrix=pae_matrix,
        plddt=plddt,
        cb_plddt=cb_plddt,
        iptm_pairs=iptm_boltz,
    )


def load_confidence(pae_file_path, structure, model_type):
    """Load the confidence data for ``model_type`` ('af2', 'af3', 'boltz', 'esmfold2')."""
    if model_type == 'af2':
        return load_af2_confidence(pae_file_path, structure)
    if model_type == 'af3':
        return load_af3_confidence(pae_file_path, structure)
    if model_type == 'boltz':
        return load_boltz_confidence(pae_file_path, structure)
    if model_type == 'esmfold2':
        return load_esmfold2_confidence(pae_file_path, structure)
    raise ValueError(f"Unknown model type: {model_type} (expected one of {MODEL_TYPES})")
