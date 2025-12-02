"""Refactored version of https://github.com/DunbrackLab/IPSAE/blob/main/ipsae.py.

Commit hash: b0a54939973a4a4389d0f89492d03e509b22a38f

Changes:

- Included chain index fix from https://github.com/DunbrackLab/IPSAE/pull/19
- Refactored the script into functions for better modularity.
- Vectorized calculations where possible for performance improvements.
- Supported specifying model type from command line arguments.

Original Script Description
=============================
Script for calculating the ipSAE score for scoring pairwise protein-protein interactions
in AlphaFold2 and AlphaFold3 models.
https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1

Also calculates:
- pDockQ: Bryant, Pozotti, and Eloffson. https://www.nature.com/articles/s41467-022-28865-w
- pDockQ2: Zhu, Shenoy, Kundrotas, Elofsson. https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714
- LIS: Kim, Hu, Comjean, Rodiger, Mohr, Perrimon. https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1

Roland Dunbrack
Fox Chase Cancer Center
version 3
April 6, 2025
MIT license: script can be modified and redistributed for non-commercial and commercial use,
as long as this information is reproduced.

Includes support for Boltz1 structures and structures with nucleic acids.

It may be necessary to install numpy with the following command:
     pip install numpy

Usage:

 python ipsae.py <path_to_af2_pae_file>     <path_to_af2_pdb_file>     <pae_cutoff> <dist_cutoff>
 python ipsae.py <path_to_af3_pae_file>     <path_to_af3_cif_file>     <pae_cutoff> <dist_cutoff>
 python ipsae.py <path_to_boltz1_pae_file>  <path_to_boltz1_cif_file>  <pae_cutoff> <dist_cutoff>

All output files will be in same path/folder as cif or pdb file
"""

# ruff: noqa: C901, PLR0912, PLR0915
import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
logger = logging.getLogger("ipSAE")

# Constants
LIS_PAE_CUTOFF = 12

RESIDUE_SET = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "DA",
    "DC",
    "DT",
    "DG",
    "A",
    "C",
    "U",
    "G",
}

NUC_RESIDUE_SET = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}

CHAIN_COLOR = {
    "A": "magenta",
    "B": "marine",
    "C": "lime",
    "D": "orange",
    "E": "yellow",
    "F": "cyan",
    "G": "lightorange",
    "H": "pink",
    "I": "deepteal",
    "J": "forest",
    "K": "lightblue",
    "L": "slate",
    "M": "violet",
    "N": "arsenic",
    "O": "iodine",
    "P": "silver",
    "Q": "red",
    "R": "sulfur",
    "S": "purple",
    "T": "olive",
    "U": "palegreen",
    "V": "green",
    "W": "blue",
    "X": "palecyan",
    "Y": "limon",
    "Z": "chocolate",
}


@dataclass
class Residue:
    """Represents a residue with its coordinates and metadata.

    Attributes:
    ----------
        atom_num: Atom serial number of the representative atom (CA or CB).
        coor: Numpy array of coordinates [x, y, z].
        res: Residue name (3-letter code).
        chainid: Chain identifier.
        resnum: Residue sequence number.
        residue_str: Formatted string representation for output.
    """

    atom_num: int
    coor: np.ndarray
    res: str
    chainid: str
    resnum: int
    residue_str: str


@dataclass
class StructureData:
    """Container for parsed structure data.

    Attributes:
    ----------
        residues: List of Residue objects (CA atoms).
        cb_residues: List of Residue objects (CB atoms, or CA for Glycine).
        chains: Array of chain identifiers for each residue.
        unique_chains: Array of unique chain identifiers.
        token_mask: Array indicating valid residues (1) vs ligands/others (0).
        residue_types: Array of residue names.
        coordinates: Array of coordinates (CB atoms).
        distances: Pairwise distance matrix between residues.
        chain_pair_type: Dictionary mapping chain pairs to type ('protein' or 'nucleic_acid').
        numres: Total number of residues.
    """

    residues: list[Residue]  # [n_res]
    cb_residues: list[Residue]  # [n_res]
    chains: np.ndarray  # [n_res,]
    unique_chains: np.ndarray  # [n_chains,]
    token_mask: np.ndarray  # [n_res,]
    residue_types: np.ndarray  # [n_res,]
    coordinates: np.ndarray  # [n_res, 3]
    distances: np.ndarray  # [n_res, n_res]
    chain_pair_type: dict[str, dict[str, str]]
    numres: int


@dataclass
class PAEData:
    """Container for PAE and confidence data.

    Attributes:
    ----------
        pae_matrix: Predicted Aligned Error matrix.
        plddt: Array of pLDDT scores (CA atoms).
        cb_plddt: Array of pLDDT scores (CB atoms).
        iptm_dict: Dictionary of ipTM scores for chain pairs.
        ptm: Global PTM score (if available).
        iptm: Global ipTM score (if available).
    """

    pae_matrix: np.ndarray
    plddt: np.ndarray
    cb_plddt: np.ndarray
    iptm_dict: dict[str, dict[str, float]]
    ptm: float = -1.0
    iptm: float = -1.0


@dataclass
class ScoreResults:
    """Container for calculated scores and output data.

    Attributes:
    ----------
        ipsae_scores: Dictionary of ipSAE scores (by residue).
        iptm_scores: Dictionary of ipTM scores (by residue).
        pdockq_scores: Dictionary of pDockQ scores (by chain pair).
        pdockq2_scores: Dictionary of pDockQ2 scores (by chain pair).
        lis_scores: Dictionary of LIS scores (by chain pair).
        metrics: Dictionary of summary metrics for each chain pair.
        by_res_data: List of formatted strings for per-residue output file.
        summary_lines: List of formatted strings for summary output file.
        pymol_script: List of formatted strings for PyMOL script.
    """

    ipsae_scores: dict
    iptm_scores: dict
    pdockq_scores: dict
    pdockq2_scores: dict
    lis_scores: dict
    metrics: dict
    by_res_data: list[str]  # Storing the formatted lines for the by-res output file
    summary_lines: list[str]  # Storing the formatted lines for the summary output file
    pymol_script: list[str]


# Helper Functions
def ptm_func(x: float, d0: float) -> float:
    """Calculate the TM-score term: 1 / (1 + (x/d0)^2).

    Args:
        x: Distance or error value.
        d0: Normalization factor.

    Returns:
    -------
        The calculated term.
    """
    return 1.0 / (1 + (x / d0) ** 2.0)


ptm_func_vec = np.vectorize(ptm_func)


def calc_d0(length: int | float, pair_type: str) -> float:
    """Calculate the normalization factor d0 for TM-score.

    Formula from Yang and Skolnick, PROTEINS: Structure, Function, and Bioinformatics 57:702-710 (2004).

    d0 = 1.24 * (L - 15)^(1/3) - 1.8
    Minimum value is 1.0 (or 2.0 for nucleic acids).

    Args:
        length: Length (number of residues).
        pair_type: Type of chain pair ('protein' or 'nucleic_acid').

    Returns:
    -------
        The calculated d0 value.
    """
    length = max(length, 27)
    min_value = 1.0
    if pair_type == "nucleic_acid":
        min_value = 2.0
    d0 = 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8
    return max(min_value, d0)


def calc_d0_array(length: list[float] | np.ndarray, pair_type: str) -> np.ndarray:
    """Vectorized version of calc_d0.

    Args:
        length: Array of lengths.
        pair_type: Type of chain pair ('protein' or 'nucleic_acid').

    Returns:
    -------
        Array of calculated d0 values.
    """
    length_arr = np.array(length, dtype=float)
    length_arr = np.maximum(27, length_arr)
    min_value = 1.0
    if pair_type == "nucleic_acid":
        min_value = 2.0
    return np.maximum(min_value, 1.24 * (length_arr - 15) ** (1.0 / 3.0) - 1.8)


def parse_pdb_atom_line(line: str) -> dict | None:
    """Parse a line from a PDB file.

    Args:
        line: A line from a PDB file starting with ATOM or HETATM.

    Returns:
    -------
        A dictionary containing atom details, or None if parsing fails.
    """
    try:
        atom_num = int(line[6:11].strip())
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21].strip()
        residue_seq_num = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
    except ValueError as e:
        logger.debug(f"Failed to parse PDB line: {line.strip()} with error: {e}")
        return None
    else:
        return {
            "atom_num": atom_num,
            "atom_name": atom_name,
            "residue_name": residue_name,
            "chain_id": chain_id,
            "residue_seq_num": residue_seq_num,
            "x": x,
            "y": y,
            "z": z,
        }


def parse_cif_atom_line(line: str, fielddict: dict[str, int]) -> dict | None:
    """Parse a line from an mmCIF file.

    Note that ligands do not have residue numbers, but modified residues do.
    We return `None` for ligands.

    Args:
        line: A line from an mmCIF file.
        fielddict: Dictionary mapping field names to column indices.

    Returns:
    -------
        A dictionary containing atom details, or None if parsing fails or it's a ligand.
    """
    linelist = line.split()
    try:
        residue_seq_num_str = linelist[fielddict["label_seq_id"]]
        if residue_seq_num_str == ".":
            return None  # ligand

        atom_num = int(linelist[fielddict["id"]])
        atom_name = linelist[fielddict["label_atom_id"]]
        residue_name = linelist[fielddict["label_comp_id"]]
        chain_id = linelist[fielddict["label_asym_id"]]
        residue_seq_num = int(residue_seq_num_str)
        x = float(linelist[fielddict["Cartn_x"]])
        y = float(linelist[fielddict["Cartn_y"]])
        z = float(linelist[fielddict["Cartn_z"]])
    except (ValueError, IndexError, KeyError) as e:
        logger.debug(f"Failed to parse mmCIF line: {line.strip()} with error: {e}")
        return None
    else:
        return {
            "atom_num": atom_num,
            "atom_name": atom_name,
            "residue_name": residue_name,
            "chain_id": chain_id,
            "residue_seq_num": residue_seq_num,
            "x": x,
            "y": y,
            "z": z,
        }


def contiguous_ranges(numbers: set[int]) -> str | None:
    """Format a set of numbers into a string of contiguous ranges.

    This is for printing out residue ranges in PyMOL scripts.

    Example: {1, 2, 3, 5, 7, 8} -> "1-3+5+7-8"

    Args:
        numbers: A set of integers.

    Returns:
    -------
        A formatted string representing the ranges, or None if empty.
    """
    if not numbers:
        return None
    sorted_numbers = sorted(numbers)
    start = sorted_numbers[0]
    end = start
    ranges = []

    def format_range(s, e) -> str:
        return f"{s}" if s == e else f"{s}-{e}"

    for number in sorted_numbers[1:]:
        if number == end + 1:
            end = number
        else:
            ranges.append(format_range(start, end))
            start = end = number
    ranges.append(format_range(start, end))
    return "+".join(ranges)


@overload
def init_chainpairdict_zeros(
    chainlist: list[str] | np.ndarray, zero: int
) -> dict[str, dict[str, int]]: ...
@overload
def init_chainpairdict_zeros(
    chainlist: list[str] | np.ndarray, zero: float
) -> dict[str, dict[str, float]]: ...
@overload
def init_chainpairdict_zeros(
    chainlist: list[str] | np.ndarray, zero: str
) -> dict[str, dict[str, str]]: ...
def init_chainpairdict_zeros(chainlist, zero=0):
    """Initialize a nested dictionary for chain pairs with zero values."""
    return {c1: {c2: zero for c2 in chainlist if c1 != c2} for c1 in chainlist}


def init_chainpairdict_npzeros(
    chainlist: list[str] | np.ndarray, arraysize: int
) -> dict[str, dict[str, np.ndarray]]:
    """Initialize a nested dictionary for chain pairs with numpy arrays of zeros."""
    return {
        c1: {c2: np.zeros(arraysize) for c2 in chainlist if c1 != c2}
        for c1 in chainlist
    }


def init_chainpairdict_set(
    chainlist: list[str] | np.ndarray,
) -> dict[str, dict[str, set]]:
    """Initialize a nested dictionary for chain pairs with empty sets."""
    return {c1: {c2: set() for c2 in chainlist if c1 != c2} for c1 in chainlist}


def classify_chains(chains: np.ndarray, residue_types: np.ndarray) -> dict[str, str]:
    """Classify chains as 'protein' or 'nucleic_acid' based on residue types for d0 calculation.

    Args:
        chains: Array of chain identifiers.
        residue_types: Array of residue names.

    Returns:
    -------
        Dictionary mapping chain ID to type ('protein' or 'nucleic_acid').
    """
    chain_types = {}
    unique_chains = np.unique(chains)
    for chain in unique_chains:
        indices = np.where(chains == chain)[0]
        chain_residues = residue_types[indices]
        nuc_count = sum(r in NUC_RESIDUE_SET for r in chain_residues)
        chain_types[chain] = "nucleic_acid" if nuc_count > 0 else "protein"
    return chain_types


def load_structure(struct_path: Path) -> StructureData:
    """Parse a PDB or mmCIF file to extract structure data.

    Reads the file to identify residues, coordinates (CA and CB), and chains.
    Calculates the pairwise distance matrix between residues.
    Classifies chain pairs as protein-protein, protein-nucleic acid, etc.

    Args:
        struct_path: Path to the PDB or mmCIF file.

    Returns:
    -------
        A StructureData object containing the parsed information.
    """
    residues = []
    cb_residues = []
    chains_list = []

    # For af3 and boltz1: need mask to identify CA atom tokens in plddt vector and pae matrix;
    # Skip ligand atom tokens and non-CA-atom tokens in PTMs (those not in RESIDUE_SET)
    token_mask = []
    atomsitefield_dict = {}
    atomsitefield_num = 0

    is_cif = struct_path.suffix == ".cif"

    with struct_path.open() as f:
        for raw_line in f:
            # mmCIF _atom_site loop headers
            if raw_line.startswith("_atom_site."):
                line = raw_line.strip()
                parts = line.split(".")
                if len(parts) == 2:
                    atomsitefield_dict[parts[1]] = atomsitefield_num
                    atomsitefield_num += 1

            # Atom coordinates
            if raw_line.startswith(("ATOM", "HETATM")):
                if is_cif:
                    atom = parse_cif_atom_line(raw_line, atomsitefield_dict)
                else:
                    atom = parse_pdb_atom_line(raw_line)

                if atom is None:  # Ligand or parse error
                    token_mask.append(0)
                    continue

                # CA or C1' (nucleic acid)
                if (atom["atom_name"] == "CA") or ("C1" in atom["atom_name"]):
                    token_mask.append(1)
                    res_obj = Residue(
                        atom_num=atom["atom_num"],
                        coor=np.array([atom["x"], atom["y"], atom["z"]]),
                        res=atom["residue_name"],
                        chainid=atom["chain_id"],
                        resnum=atom["residue_seq_num"],
                        residue_str=f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}",
                    )
                    residues.append(res_obj)
                    chains_list.append(atom["chain_id"])

                # CB or C3' or GLY CA
                if (
                    (atom["atom_name"] == "CB")
                    or ("C3" in atom["atom_name"])
                    or (atom["residue_name"] == "GLY" and atom["atom_name"] == "CA")
                ):
                    res_obj = Residue(
                        atom_num=atom["atom_num"],
                        coor=np.array([atom["x"], atom["y"], atom["z"]]),
                        res=atom["residue_name"],
                        chainid=atom["chain_id"],
                        resnum=atom["residue_seq_num"],
                        residue_str=f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}",
                    )
                    cb_residues.append(res_obj)

                # Non-CA/C1' atoms in standard residues -> token 0
                # Nucleic acids and non-CA atoms in PTM residues to tokens (as 0), whether labeled as "HETATM" (af3) or as "ATOM" (boltz1)
                if (
                    (atom["atom_name"] != "CA")
                    and ("C1" not in atom["atom_name"])
                    and (atom["residue_name"] not in RESIDUE_SET)
                ):
                    token_mask.append(0)

    logger.debug(f"Parsed _atom_site fields: {atomsitefield_dict}")

    # Convert structure information to numpy arrays
    numres = len(residues)
    coordinates = np.array([r.coor for r in cb_residues])
    chains = np.array(chains_list)
    unique_chains = np.unique(chains)  # TODO: does the order matter?
    token_array = np.array(token_mask)
    residue_types = np.array([r.res for r in residues])

    chain_dict = classify_chains(chains, residue_types)
    chain_pair_type = init_chainpairdict_zeros(unique_chains, "0")
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            if chain_dict[c1] == "nucleic_acid" or chain_dict[c2] == "nucleic_acid":
                chain_pair_type[c1][c2] = "nucleic_acid"
            else:
                chain_pair_type[c1][c2] = "protein"

    # Distance matrix
    if len(coordinates) > 0:
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        distances = np.sqrt((diff**2).sum(axis=2))
    else:
        distances = np.zeros((0, 0), dtype=float)

    return StructureData(
        residues=residues,
        cb_residues=cb_residues,
        chains=chains,
        unique_chains=unique_chains,
        token_mask=token_array,
        residue_types=residue_types,
        coordinates=coordinates,
        distances=distances,
        chain_pair_type=chain_pair_type,
        numres=numres,
    )


def load_pae_data(
    pae_path: Path, structure_data: StructureData, model_type: str
) -> PAEData:
    """Load PAE, pLDDT, and other scores from various file formats.

    In addition to `pae_path` passed by the user, we also try to load score
    files based on inferred model types.

    | Model type | File type | Filename pattern                          |
    |-----------:|:----------|:------------------------------------------|
    | AF3 server | Structure | fold_[name]_model_0.cif                   |
    |            | PAE       | fold_[name]_full_data_0.json              |
    |            | ipTM      | fold_[name]_summary_confidences_0.json    |
    | AF3 local  | Structure | model1.cif                                |
    |            | PAE       | confidences.json                          |
    |            | ipTM      | summary_confidences.json                  |
    | Boltz      | Structure | [name]_model_0.cif                        |
    |            | PAE       | pae_[name]_model_0.npz                    |
    |            | ipTM      | confidence_[name]_model_0.json            |
    |            | plDDT     | plddt_[name]_model_0.npz                  |
    | AF2        | Structure | *.pdb                                     |
    |            | PAE       | *.json                                    |

    TODO: support Chai-1 models.
        plDDT needs to be extracted from cif structure files.
        pae needs to be dumped from Chai-1 into (n_samples, N, N) npy files.
        ptm, iptm, per_chain_pair_iptm are in the scores npz files.

    Args:
        pae_path: Path to the PAE file.
        structure_data: Parsed structure data (needed for mapping atoms/residues).
        model_type: Type of model ('af2', 'af3', 'boltz1').

    Returns:
    -------
        A PAEData object containing the loaded scores.
    """
    if not pae_path.exists():
        raise FileNotFoundError(f"PAE file not found: {pae_path}")

    unique_chains = structure_data.unique_chains
    numres = structure_data.numres
    token_array = structure_data.token_mask
    mask_bool = token_array.astype(bool)

    # Initialize scores to be loaded
    pae_matrix = np.zeros((numres, numres))
    plddt = np.zeros(numres)
    cb_plddt = np.zeros(numres)
    iptm_dict = init_chainpairdict_zeros(unique_chains, 0.0)
    iptm_val = -1.0
    ptm_val = -1.0

    if model_type == "af2":
        # Load all scores from input PAE file
        if pae_path.suffix == ".pkl":
            data = np.load(pae_path, allow_pickle=True)
        else:
            with pae_path.open() as f:
                data = json.load(f)

        iptm_val = float(data.get("iptm", -1.0))
        ptm_val = float(data.get("ptm", -1.0))

        if "plddt" in data:
            plddt = np.array(data["plddt"])
            cb_plddt = np.array(data["plddt"])  # for pDockQ
        else:
            logger.warning(f"pLDDT scores not found in AF2 PAE file: {pae_path}")

        if "pae" in data:
            pae_matrix = np.array(data["pae"])
        elif "predicted_aligned_error" in data:
            pae_matrix = np.array(data["predicted_aligned_error"])
        else:
            logger.warning(f"PAE matrix not found in AF2 PAE file: {pae_path}")

    elif model_type == "boltz1":
        # Load pLDDT if file exists
        plddt_path = pae_path.with_name(pae_path.name.replace("pae", "plddt"))
        if plddt_path.exists():
            data_plddt = np.load(plddt_path)
            # Boltz plddt is 0-1, convert to 0-100
            plddt_boltz = np.array(100.0 * data_plddt["plddt"])

            # Filter by token mask
            plddt = plddt_boltz[np.ix_(mask_bool)]
            cb_plddt = plddt_boltz[np.ix_(mask_bool)]
        else:
            logger.warning(f"Boltz1 pLDDT file not found: {plddt_path}")
            ntokens = np.sum(token_array)
            plddt = np.zeros(ntokens)
            cb_plddt = np.zeros(ntokens)

        # Load PAE matrix
        data_pae = np.load(pae_path)
        pae_full = np.array(data_pae["pae"])
        pae_matrix = pae_full[np.ix_(mask_bool, mask_bool)]

        # Load ipTM scores if summary file exists
        summary_path = pae_path.with_name(
            pae_path.name.replace("pae", "confidence")
        ).with_suffix(".json")
        if summary_path.exists():
            with summary_path.open() as f:
                data_summary = json.load(f)
                if "pair_chains_iptm" in data_summary:
                    boltz_iptm = data_summary["pair_chains_iptm"]
                    # Map indices to chains
                    # TODO: is this the right order?
                    for i, c1 in enumerate(unique_chains):
                        for j, c2 in enumerate(unique_chains):
                            if c1 == c2:
                                continue
                            # Keys in json are strings of indices
                            iptm_dict[c1][c2] = boltz_iptm[str(i)][str(j)]
        else:
            logger.warning(f"Boltz1 confidence summary file not found: {summary_path}")

    elif model_type == "af3":
        with pae_path.open() as f:
            data = json.load(f)

        atom_plddts = np.array(data["atom_plddts"])

        # Derive atom indices from structure data
        # Cbeta plDDTs are needed for pDockQ
        ca_indices = [r.atom_num - 1 for r in structure_data.residues]
        cb_indices = [r.atom_num - 1 for r in structure_data.cb_residues]

        plddt = atom_plddts[ca_indices]
        cb_plddt = atom_plddts[cb_indices]

        # Get pairwise residue PAE matrix by identifying one token per protein residue.
        # Modified residues have separate tokens for each atom, so need to pull out Calpha atom as token
        if "pae" in data:
            pae_full = np.array(data["pae"])
            pae_matrix = pae_full[np.ix_(mask_bool, mask_bool)]
        else:
            raise ValueError(f"PAE matrix not found in AF3 PAE file: {pae_path}")

        # Get iptm matrix from AF3 summary_confidences file
        summary_path = None
        pae_filename = pae_path.name
        if "confidences" in pae_filename:  # AF3 local
            summary_path = pae_path.with_name(
                pae_filename.replace("confidences", "summary_confidences")
            )
        elif "full_data" in pae_filename:  # AF3 server
            summary_path = pae_path.with_name(
                pae_filename.replace("full_data", "summary_confidences")
            )

        if summary_path and summary_path.exists():
            with summary_path.open() as f:
                data_summary = json.load(f)
            if "chain_pair_iptm" in data_summary:
                af3_iptm = data_summary["chain_pair_iptm"]
                for i, c1 in enumerate(unique_chains):
                    for j, c2 in enumerate(unique_chains):
                        if c1 == c2:
                            continue
                        iptm_dict[c1][c2] = af3_iptm[i][j]
        else:
            logger.warning("AF3 summary confidences file not found")

    return PAEData(
        pae_matrix=pae_matrix,
        plddt=plddt,
        cb_plddt=cb_plddt,
        iptm_dict=iptm_dict,
        ptm=ptm_val,
        iptm=iptm_val,
    )


def calculate_pdockq_scores(
    chains: np.ndarray,
    unique_chains: np.ndarray,
    distances: np.ndarray,
    pae_matrix: np.ndarray,
    cb_plddt: np.ndarray,
    pdockq_dist_cutoff: float = 8.0,
):
    """Calculate pDockQ and pDockQ2 scores for all chain pairs."""
    pDockQ = init_chainpairdict_zeros(unique_chains, 0.0)
    pDockQ2 = init_chainpairdict_zeros(unique_chains, 0.0)
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue

            # Vectorized approach for speed
            c1_indices = np.where(chains == c1)[0]
            c2_indices = np.where(chains == c2)[0]

            if len(c1_indices) == 0 or len(c2_indices) == 0:
                continue

            # Submatrix of distances
            dists_sub = distances[np.ix_(c1_indices, c2_indices)]
            valid_mask = dists_sub <= pdockq_dist_cutoff
            npairs = np.sum(valid_mask)

            if npairs > 0:
                # Identify interface residues on c1 and c2
                # Any column in valid_mask with at least one True
                c1_interface_mask = valid_mask.any(axis=1)
                c1_interface_indices = c1_indices[c1_interface_mask]
                c2_interface_mask = valid_mask.any(axis=0)
                c2_interface_indices = c2_indices[c2_interface_mask]

                mean_plddt = cb_plddt[
                    np.hstack([c1_interface_indices, c2_interface_indices])
                ].mean()
                x = mean_plddt * np.log10(npairs)
                pDockQ[c1][c2] = 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

                # Also find sub-matrix for pDockQ2 calculation
                pae_sub = pae_matrix[np.ix_(c1_indices, c2_indices)]
                pae_valid = pae_sub[valid_mask]
                pae_ptm_sum = ptm_func_vec(pae_valid, 10.0).sum()

                mean_ptm = pae_ptm_sum / npairs
                x = mean_plddt * mean_ptm
                pDockQ2[c1][c2] = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
            # else:
            #     pDockQ[c1][c2] = 0.0
            #     pDockQ2[c1][c2] = 0.0

    return pDockQ, pDockQ2


def calculate_lis(
    chains: np.ndarray, unique_chains: np.ndarray, pae_matrix: np.ndarray
):
    """Calculate LIS scores for all chain pairs."""
    LIS = init_chainpairdict_zeros(unique_chains, 0.0)
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            c1_indices = np.where(chains == c1)[0]
            c2_indices = np.where(chains == c2)[0]

            if len(c1_indices) == 0 or len(c2_indices) == 0:
                continue

            pae_sub = pae_matrix[np.ix_(c1_indices, c2_indices)]
            valid_pae = pae_sub[pae_sub <= LIS_PAE_CUTOFF]
            if valid_pae.size > 0:
                scores = (LIS_PAE_CUTOFF - valid_pae) / LIS_PAE_CUTOFF
                LIS[c1][c2] = np.mean(scores)
            else:
                LIS[c1][c2] = 0.0

    return LIS


def calculate_scores(
    structure: StructureData,
    pae_data: PAEData,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
    pdb_stem: str = "model",
) -> ScoreResults:
    """Calculate chain-pair-specific ipSAE, ipTM, pDockQ, pDockQ2, and LIS scores.

    This is the main calculation engine. It iterates over all chain pairs and computes:
    - ipSAE: Inter-protein Predicted Aligned Error score.
    - ipTM: Inter-protein Template Modeling score.
    - pDockQ: Predicted DockQ score (Bryant et al.).
    - pDockQ2: Improved pDockQ score (Zhu et al.).
    - LIS: Local Interaction Score (Kim et al.).

    Nomenclature:
    - iptm_d0chn: calculate iptm from PAEs with no PAE cutoff
        d0 = numres in chain pair = len(chain1) + len(chain2)
    - ipsae_d0chn: calculate ipsae from PAEs with PAE cutoff
        d0 = numres in chain pair = len(chain1) + len(chain2)
    - ipsae_d0dom: calculate ipsae from PAEs with PAE cutoff
        d0 from number of residues in chain1 and chain2 that have interchain PAE<cutoff
    - ipsae_d0res: calculate ipsae from PAEs with PAE cutoff
        d0 from number of residues in chain2 that have interchain PAE<cutoff given residue in chain1

    for each chain_pair iptm/ipsae, there is (for example)
    - ipsae_d0res_byres: by-residue array;
    - ipsae_d0res_asym: asymmetric pair value (A->B is different from B->A)
    - ipsae_d0res_max: maximum of A->B and B->A value
    - ipsae_d0res_asymres: identify of residue that provides each asym maximum
    - ipsae_d0res_maxres: identify of residue that provides each maximum over both chains

    - n0num: number of residues in whole complex provided by AF2 model
    - n0chn: number of residues in chain pair = len(chain1) + len(chain2)
    - n0dom: number of residues in chain pair that have good PAE values (<cutoff)
    - n0res: number of residues in chain2 that have good PAE residues for each residue of chain1

    Args:
        structure: Parsed structure data.
        pae_data: Loaded PAE and pLDDT data.
        pae_cutoff: Cutoff for PAE to consider a residue pair "good" (default: 10.0).
        dist_cutoff: Distance cutoff for contact definition (default: 10.0).
        pdb_stem: Stem of the PDB filename (for output labeling).

    Returns:
    -------
        A ScoreResults object containing all calculated scores and output strings.
    """
    chains = structure.chains
    unique_chains = structure.unique_chains
    distances = structure.distances
    pae_matrix = pae_data.pae_matrix
    cb_plddt = pae_data.cb_plddt

    # Calculate pDockQ and LIS scores
    pDockQ, pDockQ2 = calculate_pdockq_scores(
        chains, unique_chains, distances, pae_matrix, cb_plddt
    )
    LIS = calculate_lis(chains, unique_chains, pae_matrix)

    # --- ipTM / ipSAE ---
    residues = structure.residues
    plddt = pae_data.plddt
    numres = structure.numres

    # Initialize containers
    iptm_d0chn_byres = init_chainpairdict_npzeros(unique_chains, numres)
    ipsae_d0chn_byres = init_chainpairdict_npzeros(unique_chains, numres)
    ipsae_d0dom_byres = init_chainpairdict_npzeros(unique_chains, numres)
    ipsae_d0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

    n0chn = init_chainpairdict_zeros(unique_chains, 0)
    d0chn = init_chainpairdict_zeros(unique_chains, 0.0)
    n0dom = init_chainpairdict_zeros(unique_chains, 0)
    d0dom = init_chainpairdict_zeros(unique_chains, 0.0)
    n0res_byres = init_chainpairdict_npzeros(unique_chains, numres)
    d0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

    n0res_max = init_chainpairdict_zeros(unique_chains, 0)
    n0dom_max = init_chainpairdict_zeros(unique_chains, 0)
    d0res_max = init_chainpairdict_zeros(unique_chains, 0.0)
    d0dom_max = init_chainpairdict_zeros(unique_chains, 0.0)

    unique_residues_chain1 = init_chainpairdict_set(unique_chains)
    unique_residues_chain2 = init_chainpairdict_set(unique_chains)
    dist_unique_residues_chain1 = init_chainpairdict_set(unique_chains)
    dist_unique_residues_chain2 = init_chainpairdict_set(unique_chains)

    valid_pair_counts = init_chainpairdict_zeros(unique_chains, 0)
    dist_valid_pair_counts = init_chainpairdict_zeros(unique_chains, 0)

    # First pass: d0chn
    # Calculate ipTM/ipSAE with and without PAE cutoff
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue

            c1_indices = np.where(chains == c1)[0]
            c2_indices = np.where(chains == c2)[0]

            n0chn[c1][c2] = len(c1_indices) + len(c2_indices)  # Total #res in chain1+2
            d0chn[c1][c2] = calc_d0(n0chn[c1][c2], structure.chain_pair_type[c1][c2])

            # Precompute PTM matrix for this d0
            ptm_matrix_d0chn = ptm_func_vec(
                pae_matrix[np.ix_(c1_indices, c2_indices)], d0chn[c1][c2]
            )

            # ipTM uses all of chain 2, ipSAE uses PAE cutoff
            iptm_d0chn_byres[c1][c2][c1_indices] = ptm_matrix_d0chn.mean(axis=1)

            valid_pairs_mask = pae_matrix[np.ix_(c1_indices, c2_indices)] < pae_cutoff
            ipsae_d0chn_byres[c1][c2][c1_indices] = np.ma.masked_where(
                ~valid_pairs_mask, ptm_matrix_d0chn
            ).mean(axis=1)

            # n0res and d0res by residue
            n0res_byres[c1][c2][c1_indices] = valid_pairs_mask.sum(axis=1)
            d0res_byres[c1][c2][c1_indices] = calc_d0_array(
                n0res_byres[c1][c2][c1_indices], structure.chain_pair_type[c1][c2]
            )

            # Track unique residues contributing to the ipSAE for c1,2
            valid_pair_counts[c1][c2] = np.sum(valid_pairs_mask)
            c1_contrib_residues = set(
                residues[c1_indices[i]].resnum
                for i in np.where(valid_pairs_mask.any(axis=1))[0]
            )
            unique_residues_chain1[c1][c2].update(c1_contrib_residues)

            c2_contrib_residues = set(
                residues[c2_indices[j]].resnum
                for j in np.where(valid_pairs_mask.any(axis=0))[0]
            )
            unique_residues_chain2[c1][c2].update(c2_contrib_residues)

            # Track unique residues contributing to ipTM in interface
            c2_valid_dist_mask = (valid_pairs_mask) & (
                distances[np.ix_(c1_indices, c2_indices)] < dist_cutoff
            )
            dist_valid_pair_counts[c1][c2] = np.sum(c2_valid_dist_mask)
            c1_dist_contrib_residues = set(
                residues[c1_indices[i]].resnum
                for i in np.where(c2_valid_dist_mask.any(axis=1))[0]
            )
            dist_unique_residues_chain1[c1][c2].update(c1_dist_contrib_residues)

            c2_dist_contrib_residues = set(
                residues[c2_indices[j]].resnum
                for j in np.where(c2_valid_dist_mask.any(axis=0))[0]
            )
            dist_unique_residues_chain2[c1][c2].update(c2_dist_contrib_residues)

    # Second pass: d0dom and d0res
    by_res_lines = []
    by_res_lines.append(
        "i   AlignChn ScoredChain  AlignResNum  AlignResType  AlignRespLDDT      n0chn  n0dom  n0res    d0chn     d0dom     d0res   ipTM_pae  ipSAE_d0chn ipSAE_d0dom    ipSAE \n"
    )
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue

            c1_indices = np.where(chains == c1)[0]
            c2_indices = np.where(chains == c2)[0]

            n0dom[c1][c2] = len(unique_residues_chain1[c1][c2]) + len(
                unique_residues_chain2[c1][c2]
            )
            d0dom[c1][c2] = calc_d0(n0dom[c1][c2], structure.chain_pair_type[c1][c2])

            valid_pairs_mask = pae_matrix[np.ix_(c1_indices, c2_indices)] < pae_cutoff
            ptm_matrix_d0dom = ptm_func_vec(
                pae_matrix[np.ix_(c1_indices, c2_indices)], d0dom[c1][c2]
            )
            ipsae_d0dom_byres[c1][c2][c1_indices] = np.ma.masked_where(
                ~valid_pairs_mask, ptm_matrix_d0dom
            ).mean(axis=1)

            ptm_matrix_d0res = ptm_func_vec(
                pae_matrix[np.ix_(c1_indices, c2_indices)],
                d0res_byres[c1][c2][c1_indices][:, np.newaxis],
            )
            ipsae_d0res_byres[c1][c2][c1_indices] = np.ma.masked_where(
                ~valid_pairs_mask, ptm_matrix_d0res
            ).mean(axis=1)

            # Output line generation
            for i in c1_indices:
                line = (
                    f"{i + 1:<4d}    "
                    f"{c1:4}      "
                    f"{c2:4}      "
                    f"{residues[i].resnum:4d}           "
                    f"{residues[i].res:3}        "
                    f"{plddt[i]:8.2f}         "
                    f"{int(n0chn[c1][c2]):5d}  "
                    f"{int(n0dom[c1][c2]):5d}  "
                    f"{int(n0res_byres[c1][c2][i]):5d}  "
                    f"{d0chn[c1][c2]:8.3f}  "
                    f"{d0dom[c1][c2]:8.3f}  "
                    f"{d0res_byres[c1][c2][i]:8.3f}   "
                    f"{iptm_d0chn_byres[c1][c2][i]:8.4f}    "
                    f"{ipsae_d0chn_byres[c1][c2][i]:8.4f}    "
                    f"{ipsae_d0dom_byres[c1][c2][i]:8.4f}    "
                    f"{ipsae_d0res_byres[c1][c2][i]:8.4f}\n"
                )
                by_res_lines.append(line)

    # Aggregate results (Asym and Max)
    # We need to store these to generate the summary table

    # Store results in a structured way
    results_metrics = {}

    summary_lines = []
    summary_lines.append(
        "\nChn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS       n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model\n"
    )

    pymol_lines = []
    pymol_lines.append(
        "# Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS      n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model\n"
    )

    # Helper to get max info
    def get_max_info(values_array, c1, c2):
        """Get max value and corresponding residue info from by-residue arrays."""
        vals = values_array[c1][c2]
        if np.all(vals == 0):
            return 0.0, "None", 0
        idx = np.argmax(vals)
        return vals[idx], residues[idx].residue_str, idx

    pae_str = str(int(pae_cutoff)).zfill(2)
    dist_str = str(int(dist_cutoff)).zfill(2)
    chainpairs = set()
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 >= c2:
                continue
            chainpairs.add(f"{c1}-{c2}")

    # interchain ipTM and ipSAE for each chain pair
    for pair in sorted(chainpairs):
        c_a, c_b = pair.split("-")

        # Process both directions
        for c1, c2 in [(c_a, c_b), (c_b, c_a)]:
            # Asym values
            ipsae_res_val, _, ipsae_res_idx = get_max_info(ipsae_d0res_byres, c1, c2)
            ipsae_chn_val, _, _ = get_max_info(ipsae_d0chn_byres, c1, c2)
            ipsae_dom_val, _, _ = get_max_info(ipsae_d0dom_byres, c1, c2)
            iptm_chn_val, _, _ = get_max_info(iptm_d0chn_byres, c1, c2)

            # Get n0res/d0res at max index
            n0res_val = n0res_byres[c1][c2][ipsae_res_idx]
            d0res_val = d0res_byres[c1][c2][ipsae_res_idx]

            # Counts
            res1_cnt = len(unique_residues_chain1[c1][c2])
            res2_cnt = len(unique_residues_chain2[c1][c2])
            dist1_cnt = len(dist_unique_residues_chain1[c1][c2])
            dist2_cnt = len(dist_unique_residues_chain2[c1][c2])

            # ipTM AF
            # In AF2, this is the same value for all chain pairs
            # In AF3 and Boltz, this is chain-pair specific (symmetric)
            iptm_af = pae_data.iptm_dict[c1][c2]
            if iptm_af == 0.0 and pae_data.iptm != -1.0:
                iptm_af = pae_data.iptm  # Fallback to global if per-pair not found

            outstring = (
                f"{c1}    {c2}     {pae_str:3}  {dist_str:3}  {'asym':5} "
                f"{ipsae_res_val:8.6f}    "
                f"{ipsae_chn_val:8.6f}    "
                f"{ipsae_dom_val:8.6f}    "
                f"{iptm_af:5.3f}    "
                f"{iptm_chn_val:8.6f}    "
                f"{pDockQ[c1][c2]:8.4f}   "
                f"{pDockQ2[c1][c2]:8.4f}   "
                f"{LIS[c1][c2]:8.4f}   "
                f"{int(n0res_val):5d}  "
                f"{int(n0chn[c1][c2]):5d}  "
                f"{int(n0dom[c1][c2]):5d}  "
                f"{d0res_val:6.2f}  "
                f"{d0chn[c1][c2]:6.2f}  "
                f"{d0dom[c1][c2]:6.2f}  "
                f"{res1_cnt:5d}   "
                f"{res2_cnt:5d}   "
                f"{dist1_cnt:5d}   "
                f"{dist2_cnt:5d}   "
                f"{pdb_stem}\n"
            )
            summary_lines.append(outstring)
            pymol_lines.append("# " + outstring)

            # Store in results dict
            results_metrics[f"{c1}_{c2}"] = {
                "ipsae": ipsae_res_val,
                "iptm": iptm_af,
                "pdockq": pDockQ[c1][c2],
                "pdockq2": pDockQ2[c1][c2],
                "lis": LIS[c1][c2],
            }

            # PyMOL script generation
            color1 = CHAIN_COLOR.get(c1, "magenta")
            color2 = CHAIN_COLOR.get(c2, "marine")
            chain_pair_name = f"color_{c1}_{c2}"

            # Ranges
            r1_ranges = contiguous_ranges(unique_residues_chain1[c1][c2])
            r2_ranges = contiguous_ranges(unique_residues_chain2[c1][c2])

            pymol_lines.append(
                f"alias {chain_pair_name}, color gray80, all; color {color1}, chain {c1} and resi {r1_ranges}; color {color2}, chain {c2} and resi {r2_ranges}\n\n"
            )

        # Max values (symmetric)
        # Logic: compare c1->c2 and c2->c1, take max
        # For simplicity, I'll just re-calculate max here based on computed asyms
        # But to match original output exactly, we need to output the 'max' line for c2->c1 where c1 > c2 (alphabetically)
        # The original code outputs 'max' line only once per pair?
        # "if chain1 > chain2:" block in original code implies it outputs max line once.

        c1, c2 = c_a, c_b
        if c1 < c2:  # Ensure order for comparison
            # Original code: if chain1 > chain2. So let's swap to match loop
            c1, c2 = c_b, c_a

        # Now c1 > c2
        # Calculate max values
        def get_max_of_pair(arr, k1, k2):
            v1, _, i1 = get_max_info(arr, k1, k2)
            v2, _, i2 = get_max_info(arr, k2, k1)
            if v1 >= v2:
                return v1, i1, k1, k2
            return v2, i2, k2, k1

        ipsae_res_max, idx_res, mk1_res, mk2_res = get_max_of_pair(
            ipsae_d0res_byres, c1, c2
        )
        ipsae_chn_max, _, _, _ = get_max_of_pair(ipsae_d0chn_byres, c1, c2)
        ipsae_dom_max, _, _, _ = get_max_of_pair(ipsae_d0dom_byres, c1, c2)
        iptm_chn_max, _, _, _ = get_max_of_pair(iptm_d0chn_byres, c1, c2)

        # n0/d0 for max
        n0res_max = n0res_byres[mk1_res][mk2_res][idx_res]
        d0res_max = d0res_byres[mk1_res][mk2_res][idx_res]

        # n0dom/d0dom for max (need to find which direction gave max ipsae_dom)
        v1_dom, _, _ = get_max_info(ipsae_d0dom_byres, c1, c2)
        v2_dom, _, _ = get_max_info(ipsae_d0dom_byres, c2, c1)
        if v1_dom >= v2_dom:
            n0dom_max = n0dom[c1][c2]
            d0dom_max = d0dom[c1][c2]
        else:
            n0dom_max = n0dom[c2][c1]
            d0dom_max = d0dom[c2][c1]

        # iptm af max
        iptm_af_1 = pae_data.iptm_dict[c1][c2]
        iptm_af_2 = pae_data.iptm_dict[c2][c1]
        if iptm_af_1 == 0 and pae_data.iptm != -1:
            iptm_af_1 = pae_data.iptm
        if iptm_af_2 == 0 and pae_data.iptm != -1:
            iptm_af_2 = pae_data.iptm
        iptm_af_max = max(iptm_af_1, iptm_af_2)

        pdockq2_max = max(pDockQ2[c1][c2], pDockQ2[c2][c1])
        lis_avg = (LIS[c1][c2] + LIS[c2][c1]) / 2.0

        # Residue counts (max of cross pairs)
        res1_max = max(
            len(unique_residues_chain2[c1][c2]), len(unique_residues_chain1[c2][c1])
        )
        res2_max = max(
            len(unique_residues_chain1[c1][c2]), len(unique_residues_chain2[c2][c1])
        )
        dist1_max = max(
            len(dist_unique_residues_chain2[c1][c2]),
            len(dist_unique_residues_chain1[c2][c1]),
        )
        dist2_max = max(
            len(dist_unique_residues_chain1[c1][c2]),
            len(dist_unique_residues_chain2[c2][c1]),
        )

        outstring = (
            f"{c2}    {c1}     {pae_str:3}  {dist_str:3}  {'max':5} "
            f"{ipsae_res_max:8.6f}    "
            f"{ipsae_chn_max:8.6f}    "
            f"{ipsae_dom_max:8.6f}    "
            f"{iptm_af_max:5.3f}    "
            f"{iptm_chn_max:8.6f}    "
            f"{pDockQ[c1][c2]:8.4f}   "
            f"{pdockq2_max:8.4f}   "
            f"{lis_avg:8.4f}   "
            f"{int(n0res_max):5d}  "
            f"{int(n0chn[c1][c2]):5d}  "
            f"{int(n0dom_max):5d}  "
            f"{d0res_max:6.2f}  "
            f"{d0chn[c1][c2]:6.2f}  "
            f"{d0dom_max:6.2f}  "
            f"{res1_max:5d}   "
            f"{res2_max:5d}   "
            f"{dist1_max:5d}   "
            f"{dist2_max:5d}   "
            f"{pdb_stem}\n"
        )
        summary_lines.append(outstring)
        summary_lines.append("\n")
        pymol_lines.append("# " + outstring)

    return ScoreResults(
        ipsae_scores=ipsae_d0res_byres,
        iptm_scores=iptm_d0chn_byres,
        pdockq_scores=pDockQ,
        pdockq2_scores=pDockQ2,
        lis_scores=LIS,
        metrics=results_metrics,
        by_res_data=by_res_lines,
        summary_lines=summary_lines,
        pymol_script=pymol_lines,
    )


def write_outputs(results: ScoreResults, output_prefix: str | Path) -> None:
    """Write the calculated results to output files.

    Creates three files:
    - {output_prefix}.txt: Summary table of scores.
    - {output_prefix}_byres.txt: Detailed per-residue scores.
    - {output_prefix}.pml: PyMOL script for visualization.

    Args:
        results: The ScoreResults object containing the data to write.
        output_prefix: The prefix for the output filenames (including path).
    """
    with Path(f"{output_prefix}.txt").open("w") as f:
        f.writelines(results.summary_lines)

    with Path(f"{output_prefix}_byres.txt").open("w") as f:
        f.writelines(results.by_res_data)

    with Path(f"{output_prefix}.pml").open("w") as f:
        f.writelines(results.pymol_script)


@dataclass
class CliArgs:
    """Parsed command line arguments."""

    pae_file: Path
    structure_file: Path
    pae_cutoff: float
    dist_cutoff: float
    model_type: str
    output_dir: Path | None


def parse_cli_args() -> CliArgs:
    """Parse command line arguments.

    Returns:
        A CliArgs object with the parsed arguments.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Calculate ipSAE, pDockQ, pDockQ2, and LIS scores for protein structure models."
    )
    parser.add_argument("pae_file", help="Path to PAE file (json, npz, pkl)")
    parser.add_argument("structure_file", help="Path to structure file (pdb, cif)")
    parser.add_argument("pae_cutoff", type=float, help="PAE cutoff")
    parser.add_argument("dist_cutoff", type=float, help="Distance cutoff")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save outputs. Prints results to stdout if not passed.",
    )
    parser.add_argument(
        "-t",
        "--model-type",
        help="Model type: af2, af3, boltz1, boltz2 (auto-detected if not provided).",
        default="unknown",
    )

    input_args = parser.parse_args()

    # Normalize paths and prepare typed args
    pae_path = Path(input_args.pae_file).expanduser().resolve()
    if not pae_path.exists():
        raise FileNotFoundError(f"PAE file not found: {pae_path}")
    struct_path = Path(input_args.structure_file).expanduser().resolve()
    if not struct_path.exists():
        raise FileNotFoundError(f"Structure file not found: {struct_path}")
    out_dir = (
        Path(input_args.output_dir).expanduser().resolve()
        if input_args.output_dir is not None
        else None
    )

    # Guess model type from file extensions
    if input_args.model_type != "unknown":
        model_type = input_args.model_type.lower()
        if model_type == "boltz2":
            model_type = "boltz1"  # treat boltz2 same as boltz1
        if model_type not in {"af2", "af3", "boltz1"}:
            raise ValueError(f"Invalid model type specified: {model_type}")
    else:
        model_type = "unknown"
        if struct_path.suffix == ".pdb":
            model_type = "af2"
        elif struct_path.suffix == ".cif":
            if pae_path.suffix == ".json":
                model_type = "af3"
            elif pae_path.suffix == ".npz":
                model_type = "boltz1"  # boltz2 is the same

        if model_type == "unknown":
            raise ValueError(
                f"Could not determine model type from inputs: {pae_path}, {struct_path}"
            )

    return CliArgs(
        pae_file=pae_path,
        structure_file=struct_path,
        pae_cutoff=input_args.pae_cutoff,
        dist_cutoff=input_args.dist_cutoff,
        model_type=model_type,
        output_dir=out_dir,
    )


def main() -> None:
    """Entry point for the script.

    Parses command line arguments, loads data, calculates scores, and writes outputs.
    """
    args = parse_cli_args()
    logger.debug(f"Parsed CLI args: {args}")
    logger.info(f"Detected model type: {args.model_type}")

    # Load data
    structure_data = load_structure(args.structure_file)
    pae_data = load_pae_data(args.pae_file, structure_data, args.model_type)

    # Calculate scores and dump to files
    pdb_stem = args.structure_file.stem
    results = calculate_scores(
        structure_data, pae_data, args.pae_cutoff, args.dist_cutoff, pdb_stem
    )
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        pae_str = str(int(args.pae_cutoff)).zfill(2)
        dist_str = str(int(args.dist_cutoff)).zfill(2)
        output_prefix = args.output_dir / f"{pdb_stem}_{pae_str}_{dist_str}"
        write_outputs(results, output_prefix)
        logger.info(
            f"Success! Outputs written to {output_prefix}{{.txt,_byres.txt,.pml}}"
        )
    else:
        # Print summary to stdout
        print("#" * 90 + "\n# Summary\n" + "#" * 90)
        print("".join(results.summary_lines))
        print("#" * 90 + "\n# Per-residue scores\n" + "#" * 90)
        print("".join(results.by_res_data))
        print("#" * 90 + "\n# PyMOL script\n" + "#" * 90)
        print("".join(results.pymol_script))


if __name__ == "__main__":
    main()
