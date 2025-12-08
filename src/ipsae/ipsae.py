"""Self-contained script for calculating the ipSAE score.

Changes by @y1zhou
=============================

- Included chain index fix from https://github.com/DunbrackLab/IPSAE/pull/19
- Refactored the script into functions for better modularity.
- Vectorized calculations where possible for performance improvements.
- Supported specifying model type from command line arguments.
- Supported calculating scores for specified chain groups.

If you have uv installed, you can run the script with:

    uv run ipsae.py --help

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
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy<3.0",
# ]
# ///

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
class PerResScoreResults:
    """Container for per-residue score results.

    Attributes:
        i: residue in model (from 1 to total number of residues in model)
        AlignChn: chainid of aligned residue (in PAE calculation)
        ScoredChn: chainid of scored residues (with PAE less than cutoff)
        AlignResNum: residue number of aligned residue
        AlignResType: residue type of aligned residue (three letter code)
        AlignRespLDDT: plDDT of aligned residue
        n0chn: number of residues in d0 in ipSAE_d0chn calculation
        n0dom: number of residues in d0 in ipSAE_d0dom calculation
        n0res: number of residues for d0 in ipSAE calculation
        d0chn: d0 for ipSAE_d0chn
        d0dom: d0 for ipSAE_d0dom
        d0res: d0 for ipSAE
        pTM_pae: ipTM calculated from PAE matrix and d0 = sum of chain lengths
        pSAE_d0chn: residue-specific ipSAE calculated with PAE cutoff and d0 = sum of chain lengths (n0chn)
        pSAE_d0dom: residue-specific ipSAE calculated with PAE cutoff and d0 = total number of residues in both chains with any interchain PAE<cutoff (n0dom)
        pSAE: residue-specific ipSAE value for given PAE cutoff and d0 determined by number of residues in 2nd chain with PAE<cutoff (n0res)
    """

    i: int
    AlignChn: str
    ScoredChn: str
    AlignResNum: int
    AlignResType: str
    AlignRespLDDT: float
    n0chn: int
    n0dom: int
    n0res: int
    d0chn: float
    d0dom: float
    d0res: float
    pTM_pae: float
    pSAE_d0chn: float
    pSAE_d0dom: float
    pSAE: float

    def to_formatted_line(self, end: str = "") -> str:
        """Format the per-residue score results as a fixed-width string."""
        c1, c2 = self.AlignChn, self.ScoredChn
        return (
            f"{self.i:<8d}"
            f"{c1:<10}"
            f"{c2:<10}"
            f"{self.AlignResNum:4d}           "
            f"{self.AlignResType:3}        "
            f"{self.AlignRespLDDT:8.2f}         "
            f"{self.n0chn:5d}  "
            f"{self.n0dom:5d}  "
            f"{self.n0res:5d}  "
            f"{self.d0chn:8.3f}  "
            f"{self.d0dom:8.3f}  "
            f"{self.d0res:8.3f}   "
            f"{self.pTM_pae:8.4f}    "
            f"{self.pSAE_d0chn:8.4f}    "
            f"{self.pSAE_d0dom:8.4f}    "
            f"{self.pSAE:8.4f}{end}"
        )

    @staticmethod
    def header_line() -> str:
        """Return the header line for the per-residue score output."""
        return "i   AlignChn ScoredChain  AlignResNum  AlignResType  AlignRespLDDT      n0chn  n0dom  n0res    d0chn     d0dom     d0res   ipTM_pae  ipSAE_d0chn ipSAE_d0dom    ipSAE \n"


@dataclass
class ChainPairScoreResults:
    """Container for chain-pair summary score results.

    Attributes:
        Chn1: first chain identifier
        Chn2: second chain identifier
        PAE: PAE cutoff value
        Dist: Distance cutoff for CA-CA contacts
        Type: "asym" or "max"; asym means asymmetric ipTM/ipSAE values; max is maximum of asym values
        ipSAE: ipSAE value for given PAE cutoff and d0 determined by number of residues in 2nd chain with PAE<cutoff
        ipSAE_d0chn: ipSAE calculated with PAE cutoff and d0 = sum of chain lengths
        ipSAE_d0dom: ipSAE calculated with PAE cutoff and d0 = total number of residues in both chains with any interchain PAE<cutoff
        ipTM_af: AlphaFold ipTM values. For AF2, this is for whole complex from json file. For AF3, this is symmetric pairwise value from summary json file.
        ipTM_d0chn: ipTM (no PAE cutoff) calculated from PAE matrix and d0 = sum of chain lengths
        pDockQ: score from pLDDTs from Bryant, Pozotti, and Eloffson
        pDockQ2: score based on PAE, calculated pairwise from Zhu, Shenoy, Kundrotas, Elofsson
        LIS: Local Interaction Score based on transform of PAEs from Kim, Hu, Comjean, Rodiger, Mohr, Perrimon
        n0res: number of residues for d0 in ipSAE calculation
        n0chn: number of residues in d0 in ipSAE_d0chn calculation
        n0dom: number of residues in d0 in ipSAE_d0dom calculation
        d0res: d0 for ipSAE
        d0chn: d0 for ipSAE_d0chn
        d0dom: d0 for ipSAE_d0dom
        nres1: number of residues in chain1 with PAE<cutoff with residues in chain2
        nres2: number of residues in chain2 with PAE<cutoff with residues in chain1
        dist1: number of residues in chain 1 with PAE<cutoff and dist<cutoff from chain2
        dist2: number of residues in chain 2 with PAE<cutoff and dist<cutoff from chain1
        Model: AlphaFold filename
    """

    Chn1: str
    Chn2: str
    PAE: float
    Dist: float
    Type: str
    ipSAE: float
    ipSAE_d0chn: float
    ipSAE_d0dom: float
    ipTM_af: float
    ipTM_d0chn: float
    pDockQ: float
    pDockQ2: float
    LIS: float
    n0res: int
    n0chn: int
    n0dom: int
    d0res: float
    d0chn: float
    d0dom: float
    nres1: int
    nres2: int
    dist1: int
    dist2: int
    Model: str

    def to_formatted_line(self, end: str = "") -> str:
        """Format the summary result as a fixed-width string."""
        pae_str = str(int(self.PAE)).zfill(2)
        dist_str = str(int(self.Dist)).zfill(2)

        return (
            f"{self.Chn1:<5}{self.Chn2:<5} {pae_str:3}  {dist_str:3}  {self.Type:5} "
            f"{self.ipSAE:8.6f}    "
            f"{self.ipSAE_d0chn:8.6f}    "
            f"{self.ipSAE_d0dom:8.6f}    "
            f"{self.ipTM_af:5.3f}    "
            f"{self.ipTM_d0chn:8.6f}    "
            f"{self.pDockQ:8.4f}   "
            f"{self.pDockQ2:8.4f}   "
            f"{self.LIS:8.4f}   "
            f"{self.n0res:5d}  "
            f"{self.n0chn:5d}  "
            f"{self.n0dom:5d}  "
            f"{self.d0res:6.2f}  "
            f"{self.d0chn:6.2f}  "
            f"{self.d0dom:6.2f}  "
            f"{self.nres1:5d}   "
            f"{self.nres2:5d}   "
            f"{self.dist1:5d}   "
            f"{self.dist2:5d}   "
            f"{self.Model}{end}"
        )

    @staticmethod
    def header_line() -> str:
        """Return the header line for the summary output."""
        return "Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS       n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model\n"


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
        metrics: Dictionary of pDockQ, pDockQ2, and LIS scores for each chain pair.
        by_res_scores: Lists of per-residue scores.
        chain_pair_scores: List of chain-pair summary score results.
        pymol_script: List of formatted strings for PyMOL script.
    """

    ipsae_scores: dict[str, dict[str, np.ndarray]]  # {c1: {c2: np.ndarray}}
    iptm_scores: dict[str, dict[str, np.ndarray]]  # {c1: {c2: np.ndarray}}
    pdockq_scores: dict[str, dict[str, float]]  # {c1: {c2: score}}
    pdockq2_scores: dict[str, dict[str, float]]  # {c1: {c2: score}}
    lis_scores: dict[str, dict[str, float]]  # {c1: {c2: score}}
    metrics: dict[str, dict[str, float]]  # {`<c1>_<c2>`: {metric_name: value}}

    by_res_scores: list[PerResScoreResults]
    chain_pair_scores: list[
        ChainPairScoreResults
    ]  # List of chain-pair summary score results
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


def parse_chain_groups(
    chain_groups_str: str,
    unique_chains: np.ndarray | None = None,
) -> list[tuple[list[str], list[str]]]:
    """Parse chain groups string into a list of chain group pairs.

    Format: "A/H+L,A/H,A/L" means:
        - Calculate scores between chain A and chains H+L (treated as one group)
        - Calculate scores between chain A and chain H
        - Calculate scores between chain A and chain L

    Use "..." to include all default individual chain permutations.
    For example, "A/H+L,..." with chains A, H, L will include A/H+L plus all
    individual chain pairs (A/H, A/L, H/A, H/L, L/A, L/H).

    Args:
        chain_groups_str: Comma-separated chain group pairs in format "group1/group2".
            Each group can contain multiple chains joined with "+", e.g., "H+L".
            Use "..." to include all default individual chain permutations.
        unique_chains: Array of unique chain identifiers, required when using "...".

    Returns:
        List of tuples, where each tuple contains two lists of chain identifiers.
        For example, "A/H+L,A/H" returns [(['A'], ['H', 'L']), (['A'], ['H'])].
        Duplicates are automatically removed.
    """
    result = []
    seen: set[str] = set()  # Track seen pairs to remove duplicates
    has_ellipsis = False

    pairs = chain_groups_str.split(",")
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue

        # Check for ellipsis token
        if pair == "...":
            has_ellipsis = True
            continue

        if "/" not in pair:
            raise ValueError(
                f"Invalid chain group pair format: '{pair}'. Expected 'group1/group2'."
            )
        parts = pair.split("/")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid chain group pair format: '{pair}'. Expected exactly one '/' separator."
            )
        group1_str, group2_str = parts
        group1 = sorted(g for c in group1_str.split("+") if (g := c.strip()))
        group2 = sorted(g for c in group2_str.split("+") if (g := c.strip()))
        if not group1 or not group2:
            raise ValueError(
                f"Invalid chain group pair: '{pair}'. Both groups must contain at least one chain."
            )

        # Add to result if not a duplicate (check both directions)
        pair_key_forward = f"{chain_group_name(group1)}/{chain_group_name(group2)}"
        pair_key_reverse = f"{chain_group_name(group2)}/{chain_group_name(group1)}"
        if pair_key_forward not in seen:
            seen.add(pair_key_forward)
            seen.add(pair_key_reverse)  # Mark reverse as seen too
            result.append((group1, group2))
            result.append((group2, group1))

    # Handle ellipsis - add all individual chain permutations
    # Use nested loops to match original behavior ordering:
    # For chains [A,B,C]: A->B, A->C, B->A, B->C, C->A, C->B
    if has_ellipsis:
        if unique_chains is None:
            raise ValueError(
                "Cannot use '...' without providing unique_chains. "
                "The '...' token requires knowledge of available chains."
            )

        for c1 in unique_chains:
            for c2 in unique_chains:
                if c1 == c2:
                    continue
                pair_key = f"{c1}/{c2}"
                if pair_key not in seen:
                    seen.add(pair_key)
                    result.append(([c1], [c2]))

    return result


def get_chain_group_indices(chains: np.ndarray, chain_group: list[str]) -> np.ndarray:
    """Get indices of residues belonging to any chain in the given chain group.

    Args:
        chains: Array of chain identifiers for each residue.
        chain_group: List of chain identifiers forming the group.

    Returns:
        Array of indices for residues in the chain group.
    """
    mask = np.isin(chains, chain_group)
    return np.where(mask)[0]


def chain_group_name(chain_group: list[str]) -> str:
    """Generate a display name for a chain group.

    Args:
        chain_group: List of chain identifiers.

    Returns:
        String representation of the chain group (e.g., "H+L" for ["H", "L"]).
    """
    return "+".join(chain_group)


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
    unique_chains = np.unique(chains)
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


def load_obj_from_file(file_path: Path):
    """Load an object from a file (numpy .npy/.npz/.pkl or JSON).

    Args:
        file_path: Path to the file.

    Returns:
        The loaded object (numpy array or dictionary).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix in {".npy", ".npz"}:
        return np.load(file_path)
    elif file_path.suffix == ".pkl":
        return np.load(file_path, allow_pickle=True)
    elif file_path.suffix == ".json":
        with file_path.open() as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


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
        data = load_obj_from_file(pae_path)

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
            data_plddt = load_obj_from_file(plddt_path)
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
        data_pae = load_obj_from_file(pae_path)
        pae_full = np.array(data_pae["pae"])
        pae_matrix = pae_full[np.ix_(mask_bool, mask_bool)]

        # Load ipTM scores if summary file exists
        summary_path = pae_path.with_name(
            pae_path.name.replace("pae", "confidence")
        ).with_suffix(".json")
        if summary_path.exists():
            data_summary = load_obj_from_file(summary_path)
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
        data = load_obj_from_file(pae_path)

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
            data_summary = load_obj_from_file(summary_path)
            if "chain_pair_iptm" in data_summary:
                af3_iptm = data_summary["chain_pair_iptm"]
                for i, c1 in enumerate(unique_chains):
                    for j, c2 in enumerate(unique_chains):
                        if c1 == c2:
                            continue
                        iptm_dict[c1][c2] = af3_iptm[i][j]
            else:
                logger.warning(
                    f"AF3 summary confidences file missing key 'chain_pair_iptm': {summary_path}"
                )
        elif summary_path:
            logger.warning(f"AF3 summary confidences file not found: {summary_path}")
        else:
            logger.warning(
                f"Could not determine AF3 summary confidences file path from PAE file: {pae_path}"
            )

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
    chain_pairs: list[tuple[list[str], list[str]]],
    distances: np.ndarray,
    pae_matrix: np.ndarray,
    cb_plddt: np.ndarray,
    pdockq_dist_cutoff: float = 8.0,
):
    """Calculate pDockQ and pDockQ2 scores for specified chain pairs."""
    pDockQ: dict[str, dict[str, float]] = {}
    pDockQ2: dict[str, dict[str, float]] = {}

    for group1, group2 in chain_pairs:
        g1 = chain_group_name(group1)
        g2 = chain_group_name(group2)

        # Initialize nested dict if needed
        if g1 not in pDockQ:
            pDockQ[g1] = {}
            pDockQ2[g1] = {}

        # Get indices for chain groups
        g1_indices = get_chain_group_indices(chains, group1)
        g2_indices = get_chain_group_indices(chains, group2)

        if len(g1_indices) == 0 or len(g2_indices) == 0:
            pDockQ[g1][g2] = 0.0
            pDockQ2[g1][g2] = 0.0
            continue

        # Submatrix of distances
        dists_sub = distances[np.ix_(g1_indices, g2_indices)]
        valid_mask = dists_sub <= pdockq_dist_cutoff
        npairs = np.sum(valid_mask)

        if npairs > 0:
            # Identify interface residues on g1 and g2
            # Any column in valid_mask with at least one True
            g1_interface_mask = valid_mask.any(axis=1)
            g1_interface_indices = g1_indices[g1_interface_mask]
            g2_interface_mask = valid_mask.any(axis=0)
            g2_interface_indices = g2_indices[g2_interface_mask]

            mean_plddt = cb_plddt[
                np.hstack([g1_interface_indices, g2_interface_indices])
            ].mean()
            x = mean_plddt * np.log10(npairs)
            pDockQ[g1][g2] = 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018

            # Also find sub-matrix for pDockQ2 calculation
            pae_sub = pae_matrix[np.ix_(g1_indices, g2_indices)]
            pae_valid = pae_sub[valid_mask]
            pae_ptm_sum = ptm_func_vec(pae_valid, 10.0).sum()

            mean_ptm = pae_ptm_sum / npairs
            x = mean_plddt * mean_ptm
            pDockQ2[g1][g2] = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
        else:
            pDockQ[g1][g2] = 0.0
            pDockQ2[g1][g2] = 0.0

    return pDockQ, pDockQ2


def calculate_lis(
    chains: np.ndarray,
    chain_pairs: list[tuple[list[str], list[str]]],
    pae_matrix: np.ndarray,
):
    """Calculate LIS scores for specified chain pairs."""
    LIS: dict[str, dict[str, float]] = {}

    for group1, group2 in chain_pairs:
        g1 = chain_group_name(group1)
        g2 = chain_group_name(group2)

        # Initialize nested dict if needed
        if g1 not in LIS:
            LIS[g1] = {}

        g1_indices = get_chain_group_indices(chains, group1)
        g2_indices = get_chain_group_indices(chains, group2)

        if len(g1_indices) == 0 or len(g2_indices) == 0:
            LIS[g1][g2] = 0.0
            continue

        pae_sub = pae_matrix[np.ix_(g1_indices, g2_indices)]
        valid_pae = pae_sub[pae_sub <= LIS_PAE_CUTOFF]
        if valid_pae.size > 0:
            scores = (LIS_PAE_CUTOFF - valid_pae) / LIS_PAE_CUTOFF
            LIS[g1][g2] = np.mean(scores)
        else:
            LIS[g1][g2] = 0.0

    return LIS


def aggregate_byres_scores(
    residues: list[Residue],
    pae_cutoff: int | float,
    dist_cutoff: int | float,
    chain_pairs: list[tuple[list[str], list[str]]],
    ipsae_d0res_byres: dict[str, dict[str, np.ndarray]],
    ipsae_d0chn_byres: dict[str, dict[str, np.ndarray]],
    ipsae_d0dom_byres: dict[str, dict[str, np.ndarray]],
    iptm_d0chn_byres: dict[str, dict[str, np.ndarray]],
    n0res_byres: dict[str, dict[str, np.ndarray]],
    d0res_byres: dict[str, dict[str, np.ndarray]],
    unique_residues_chain1: dict[str, dict[str, set]],
    unique_residues_chain2: dict[str, dict[str, set]],
    dist_unique_residues_chain1: dict[str, dict[str, set]],
    dist_unique_residues_chain2: dict[str, dict[str, set]],
    pae_data: PAEData,
    pDockQ: dict[str, dict[str, float]],
    pDockQ2: dict[str, dict[str, float]],
    LIS: dict[str, dict[str, float]],
    n0chn: dict[str, dict[str, int]],
    n0dom: dict[str, dict[str, int]],
    d0chn: dict[str, dict[str, float]],
    d0dom: dict[str, dict[str, float]],
    label: str,
) -> tuple[list[ChainPairScoreResults], list[str], dict[str, dict[str, float]]]:
    """Aggregate per-residue scores into chain-pair-specific scores.

    Returns:
        A tuple containing:
        - List of ChainPairScoreResults objects with chain-pair scores.
        - List of PyMOL script lines.
        - Dictionary of metrics for each chain pair.
    """
    # Store results in a structured way
    results_metrics: dict[str, dict[str, float]] = {}

    chain_pair_scores: list[ChainPairScoreResults] = []

    pymol_lines = [
        "# Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS      n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model\n"
    ]

    # Helper to get max info
    def get_max_info(values_array, c1, c2):
        """Get max value and corresponding residue info from by-residue arrays."""
        vals = values_array[c1][c2]
        if np.all(vals == 0):
            return 0.0, "None", 0
        idx = np.argmax(vals)
        return vals[idx], residues[idx].residue_str, idx

    # Build a map from group names to chain lists
    group_to_chains: dict[str, list[str]] = {}
    for group1, group2 in chain_pairs:
        g1 = chain_group_name(group1)
        g2 = chain_group_name(group2)
        group_to_chains[g1] = group1
        group_to_chains[g2] = group2

    # Identify all unique unordered pairs from chain_pairs
    unique_pair_set: set[tuple[str, str]] = set()
    for group1, group2 in chain_pairs:
        g1 = chain_group_name(group1)
        g2 = chain_group_name(group2)
        if g1 != g2:
            pair_key: tuple[str, str] = tuple(sorted((g1, g2)))
            unique_pair_set.add(pair_key)

    # Helper to get max across both directions
    def get_max_of_pair(arr, k1, k2):
        v1, _, i1 = get_max_info(arr, k1, k2)
        v2, _, i2 = get_max_info(arr, k2, k1)
        if v1 >= v2:
            return v1, i1, k1, k2
        return v2, i2, k2, k1

    # Process each unique pair: output both asym directions + max together
    for pair_key in sorted(unique_pair_set):
        g_a, g_b = pair_key  # Alphabetically sorted

        # Check if data exists for both directions
        if g_a not in ipsae_d0res_byres or g_b not in ipsae_d0res_byres.get(g_a, {}):
            continue
        if g_b not in ipsae_d0res_byres or g_a not in ipsae_d0res_byres.get(g_b, {}):
            continue

        # Process both asym directions: g_a -> g_b, then g_b -> g_a
        for g1, g2 in [(g_a, g_b), (g_b, g_a)]:
            group1 = group_to_chains[g1]
            group2 = group_to_chains[g2]

            # Asym values
            ipsae_res_val, _, ipsae_res_idx = get_max_info(ipsae_d0res_byres, g1, g2)
            ipsae_chn_val, _, _ = get_max_info(ipsae_d0chn_byres, g1, g2)
            ipsae_dom_val, _, _ = get_max_info(ipsae_d0dom_byres, g1, g2)
            iptm_chn_val, _, _ = get_max_info(iptm_d0chn_byres, g1, g2)

            # Get n0res/d0res at max index
            n0res_val = n0res_byres[g1][g2][ipsae_res_idx]
            d0res_val = d0res_byres[g1][g2][ipsae_res_idx]

            # Counts
            res1_cnt = len(unique_residues_chain1[g1][g2])
            res2_cnt = len(unique_residues_chain2[g1][g2])
            dist1_cnt = len(dist_unique_residues_chain1[g1][g2])
            dist2_cnt = len(dist_unique_residues_chain2[g1][g2])

            # ipTM AF - for chain groups, try individual chains first
            iptm_af = 0.0
            if len(group1) == 1 and len(group2) == 1:
                # Individual chains - try to get from iptm_dict
                single_c1, single_c2 = group1[0], group2[0]
                if (
                    single_c1 in pae_data.iptm_dict
                    and single_c2 in pae_data.iptm_dict.get(single_c1, {})
                ):
                    iptm_af = pae_data.iptm_dict[single_c1][single_c2]
            if iptm_af == 0.0 and pae_data.iptm != -1.0:
                iptm_af = pae_data.iptm  # Fallback to global

            summary_result = ChainPairScoreResults(
                Chn1=g1,
                Chn2=g2,
                PAE=pae_cutoff,
                Dist=dist_cutoff,
                Type="asym",
                ipSAE=float(ipsae_res_val),
                ipSAE_d0chn=float(ipsae_chn_val),
                ipSAE_d0dom=float(ipsae_dom_val),
                ipTM_af=float(iptm_af),
                ipTM_d0chn=float(iptm_chn_val),
                pDockQ=float(pDockQ[g1][g2]),
                pDockQ2=float(pDockQ2[g1][g2]),
                LIS=float(LIS[g1][g2]),
                n0res=int(n0res_val),
                n0chn=int(n0chn[g1][g2]),
                n0dom=int(n0dom[g1][g2]),
                d0res=float(d0res_val),
                d0chn=float(d0chn[g1][g2]),
                d0dom=float(d0dom[g1][g2]),
                nres1=res1_cnt,
                nres2=res2_cnt,
                dist1=dist1_cnt,
                dist2=dist2_cnt,
                Model=label,
            )
            chain_pair_scores.append(summary_result)
            pymol_lines.append("# " + summary_result.to_formatted_line(end="\n"))

            # Store in results dict
            results_metrics[f"{g1}_{g2}"] = {
                "ipsae": float(ipsae_res_val),
                "iptm": float(iptm_af),
                "pdockq": float(pDockQ[g1][g2]),
                "pdockq2": float(pDockQ2[g1][g2]),
                "lis": float(LIS[g1][g2]),
            }

            # PyMOL script generation - handle multi-chain groups
            # Use '-' instead of '+' in alias names for compatibility
            pymol_g1 = g1.replace("+", "-")
            pymol_g2 = g2.replace("+", "-")
            chain_pair_name = f"color_{pymol_g1}_{pymol_g2}"

            # Ranges
            r1_ranges = contiguous_ranges(unique_residues_chain1[g1][g2])
            r2_ranges = contiguous_ranges(unique_residues_chain2[g1][g2])

            # Build color commands for each chain in the groups
            color_cmds = ["color gray80, all"]
            for chain_id in group1:
                color = CHAIN_COLOR.get(chain_id, "magenta")
                if r1_ranges:
                    color_cmds.append(
                        f"color {color}, chain {chain_id} and resi {r1_ranges}"
                    )
            for chain_id in group2:
                color = CHAIN_COLOR.get(chain_id, "marine")
                if r2_ranges:
                    color_cmds.append(
                        f"color {color}, chain {chain_id} and resi {r2_ranges}"
                    )

            pymol_lines.append(f"alias {chain_pair_name}, {'; '.join(color_cmds)}\n\n")

        # Now add the max row for this unique pair
        # Original code uses larger chain first (c1 > c2), then outputs (c2, c1)
        # We already have g_a < g_b alphabetically sorted
        # So to match: use g_b (larger) as g1, g_a (smaller) as g2
        g1, g2 = g_b, g_a

        # Calculate max values
        ipsae_res_max, idx_res, mk1_res, mk2_res = get_max_of_pair(
            ipsae_d0res_byres, g1, g2
        )
        ipsae_chn_max, _, _, _ = get_max_of_pair(ipsae_d0chn_byres, g1, g2)
        ipsae_dom_max, _, _, _ = get_max_of_pair(ipsae_d0dom_byres, g1, g2)
        iptm_chn_max, _, _, _ = get_max_of_pair(iptm_d0chn_byres, g1, g2)

        # n0/d0 for max
        n0res_max = n0res_byres[mk1_res][mk2_res][idx_res]
        d0res_max = d0res_byres[mk1_res][mk2_res][idx_res]

        # n0dom/d0dom for max
        v1_dom, _, _ = get_max_info(ipsae_d0dom_byres, g1, g2)
        v2_dom, _, _ = get_max_info(ipsae_d0dom_byres, g2, g1)
        if v1_dom >= v2_dom:
            n0dom_max = n0dom[g1][g2]
            d0dom_max = d0dom[g1][g2]
        else:
            n0dom_max = n0dom[g2][g1]
            d0dom_max = d0dom[g2][g1]

        # iptm af max - for individual chains, use iptm_dict; for groups, use global
        iptm_af_1 = pae_data.iptm_dict.get(g1, {}).get(g2, 0.0)
        iptm_af_2 = pae_data.iptm_dict.get(g2, {}).get(g1, 0.0)
        if iptm_af_1 == 0 and pae_data.iptm != -1.0:
            iptm_af_1 = pae_data.iptm
        if iptm_af_2 == 0 and pae_data.iptm != -1.0:
            iptm_af_2 = pae_data.iptm
        iptm_af_max = max(iptm_af_1, iptm_af_2)

        pdockq2_max = max(pDockQ2[g1][g2], pDockQ2[g2][g1])
        lis_avg = (LIS[g1][g2] + LIS[g2][g1]) / 2.0

        # Residue counts (max of cross pairs)
        res1_max = max(
            len(unique_residues_chain2[g1][g2]), len(unique_residues_chain1[g2][g1])
        )
        res2_max = max(
            len(unique_residues_chain1[g1][g2]), len(unique_residues_chain2[g2][g1])
        )
        dist1_max = max(
            len(dist_unique_residues_chain2[g1][g2]),
            len(dist_unique_residues_chain1[g2][g1]),
        )
        dist2_max = max(
            len(dist_unique_residues_chain1[g1][g2]),
            len(dist_unique_residues_chain2[g2][g1]),
        )

        # Output with swapped chain names (Chn1=g2, Chn2=g1) to match original
        summary_result = ChainPairScoreResults(
            Chn1=g2,
            Chn2=g1,
            PAE=pae_cutoff,
            Dist=dist_cutoff,
            Type="max",
            ipSAE=float(ipsae_res_max),
            ipSAE_d0chn=float(ipsae_chn_max),
            ipSAE_d0dom=float(ipsae_dom_max),
            ipTM_af=float(iptm_af_max),
            ipTM_d0chn=float(iptm_chn_max),
            pDockQ=float(pDockQ[g1][g2]),
            pDockQ2=float(pdockq2_max),
            LIS=float(lis_avg),
            n0res=int(n0res_max),
            n0chn=int(n0chn[g1][g2]),
            n0dom=int(n0dom_max),
            d0res=float(d0res_max),
            d0chn=float(d0chn[g1][g2]),
            d0dom=float(d0dom_max),
            nres1=res1_max,
            nres2=res2_max,
            dist1=dist1_max,
            dist2=dist2_max,
            Model=label,
        )
        chain_pair_scores.append(summary_result)
        pymol_lines.append("# " + summary_result.to_formatted_line(end="\n") + "\n")

    return chain_pair_scores, pymol_lines, results_metrics


def calculate_scores(
    structure: StructureData,
    pae_data: PAEData,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
    label: str = "model",
    chain_groups: list[tuple[list[str], list[str]]] | None = None,
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
        label: Filename prefix (for output labeling).
        chain_groups: Optional list of chain group pairs to calculate scores for.
            If None, all chain pairs are calculated.

    Returns:
    -------
        A ScoreResults object containing all calculated scores and output strings.
    """
    chains = structure.chains
    unique_chains = structure.unique_chains
    distances = structure.distances
    pae_matrix = pae_data.pae_matrix
    cb_plddt = pae_data.cb_plddt

    # Get unique chain group names for dictionary keys
    if chain_groups is None:
        chain_pairs = parse_chain_groups("...", unique_chains)
    else:
        chain_pairs = chain_groups
    chain_group_names = list(
        {chain_group_name(g) for pair in chain_pairs for g in pair}
    )

    # Calculate pDockQ and LIS scores
    pDockQ, pDockQ2 = calculate_pdockq_scores(
        chains, chain_pairs, distances, pae_matrix, cb_plddt
    )
    LIS = calculate_lis(chains, chain_pairs, pae_matrix)

    # --- ipTM / ipSAE ---
    residues = structure.residues
    plddt = pae_data.plddt
    numres = structure.numres

    # Initialize containers using chain group names
    def init_chainpair_dict_zeros(default_val):
        return {
            c1: {c2: default_val for c2 in chain_group_names if c1 != c2}
            for c1 in chain_group_names
        }

    def init_chainpair_dict_npzeros():
        return {
            c1: {c2: np.zeros(numres) for c2 in chain_group_names if c1 != c2}
            for c1 in chain_group_names
        }

    def init_chainpair_dict_set():
        return {
            c1: {c2: set() for c2 in chain_group_names if c1 != c2}
            for c1 in chain_group_names
        }

    iptm_d0chn_byres = init_chainpair_dict_npzeros()
    ipsae_d0chn_byres = init_chainpair_dict_npzeros()
    ipsae_d0dom_byres = init_chainpair_dict_npzeros()
    ipsae_d0res_byres = init_chainpair_dict_npzeros()

    n0chn: dict[str, dict[str, int]] = init_chainpair_dict_zeros(0)
    d0chn: dict[str, dict[str, float]] = init_chainpair_dict_zeros(0.0)
    n0dom: dict[str, dict[str, int]] = init_chainpair_dict_zeros(0)
    d0dom: dict[str, dict[str, float]] = init_chainpair_dict_zeros(0.0)
    n0res_byres = init_chainpair_dict_npzeros()
    d0res_byres = init_chainpair_dict_npzeros()

    unique_residues_chain1 = init_chainpair_dict_set()
    unique_residues_chain2 = init_chainpair_dict_set()
    dist_unique_residues_chain1 = init_chainpair_dict_set()
    dist_unique_residues_chain2 = init_chainpair_dict_set()

    # Helper to determine pair type for a chain group pair
    def get_pair_type(group1: list[str], group2: list[str]) -> str:
        for c1 in group1:
            for c2 in group2:
                if (
                    c1 in structure.chain_pair_type
                    and c2 in structure.chain_pair_type.get(c1, {})
                ):
                    if structure.chain_pair_type[c1][c2] == "nucleic_acid":
                        return "nucleic_acid"
        return "protein"

    # First pass: d0chn
    # Calculate ipTM/ipSAE with and without PAE cutoff
    for group1, group2 in chain_pairs:
        g1 = chain_group_name(group1)
        g2 = chain_group_name(group2)

        g1_indices = get_chain_group_indices(chains, group1)
        g2_indices = get_chain_group_indices(chains, group2)

        if len(g1_indices) == 0 or len(g2_indices) == 0:
            continue

        pair_type = get_pair_type(group1, group2)

        n0chn[g1][g2] = len(g1_indices) + len(g2_indices)  # Total #res in group1+2
        d0chn[g1][g2] = calc_d0(n0chn[g1][g2], pair_type)

        # Precompute PTM matrix for this d0
        ptm_matrix_d0chn = ptm_func_vec(
            pae_matrix[np.ix_(g1_indices, g2_indices)], d0chn[g1][g2]
        )

        # ipTM uses all of group 2, ipSAE uses PAE cutoff
        iptm_d0chn_byres[g1][g2][g1_indices] = ptm_matrix_d0chn.mean(axis=1)

        valid_pairs_mask = pae_matrix[np.ix_(g1_indices, g2_indices)] < pae_cutoff
        ipsae_d0chn_byres[g1][g2][g1_indices] = np.ma.masked_where(
            ~valid_pairs_mask, ptm_matrix_d0chn
        ).mean(axis=1)

        # n0res and d0res by residue
        n0res_byres[g1][g2][g1_indices] = valid_pairs_mask.sum(axis=1)
        d0res_byres[g1][g2][g1_indices] = calc_d0_array(
            n0res_byres[g1][g2][g1_indices], pair_type
        )

        # Track unique residues contributing to the ipSAE for g1, g2
        g1_contrib_residues = set(
            residues[g1_indices[i]].resnum
            for i in np.where(valid_pairs_mask.any(axis=1))[0]
        )
        unique_residues_chain1[g1][g2].update(g1_contrib_residues)

        g2_contrib_residues = set(
            residues[g2_indices[j]].resnum
            for j in np.where(valid_pairs_mask.any(axis=0))[0]
        )
        unique_residues_chain2[g1][g2].update(g2_contrib_residues)

        # Track unique residues contributing to ipTM in interface
        valid_dist_mask = (valid_pairs_mask) & (
            distances[np.ix_(g1_indices, g2_indices)] < dist_cutoff
        )
        g1_dist_contrib_residues = set(
            residues[g1_indices[i]].resnum
            for i in np.where(valid_dist_mask.any(axis=1))[0]
        )
        dist_unique_residues_chain1[g1][g2].update(g1_dist_contrib_residues)

        g2_dist_contrib_residues = set(
            residues[g2_indices[j]].resnum
            for j in np.where(valid_dist_mask.any(axis=0))[0]
        )
        dist_unique_residues_chain2[g1][g2].update(g2_dist_contrib_residues)

    # Second pass: d0dom and d0res
    by_res_lines: list[PerResScoreResults] = []
    for group1, group2 in chain_pairs:
        g1 = chain_group_name(group1)
        g2 = chain_group_name(group2)

        g1_indices = get_chain_group_indices(chains, group1)
        g2_indices = get_chain_group_indices(chains, group2)

        if len(g1_indices) == 0 or len(g2_indices) == 0:
            continue

        pair_type = get_pair_type(group1, group2)

        n0dom[g1][g2] = len(unique_residues_chain1[g1][g2]) + len(
            unique_residues_chain2[g1][g2]
        )
        d0dom[g1][g2] = calc_d0(n0dom[g1][g2], pair_type)

        valid_pairs_mask = pae_matrix[np.ix_(g1_indices, g2_indices)] < pae_cutoff
        ptm_matrix_d0dom = ptm_func_vec(
            pae_matrix[np.ix_(g1_indices, g2_indices)], d0dom[g1][g2]
        )
        ipsae_d0dom_byres[g1][g2][g1_indices] = np.ma.masked_where(
            ~valid_pairs_mask, ptm_matrix_d0dom
        ).mean(axis=1)

        ptm_matrix_d0res = ptm_func_vec(
            pae_matrix[np.ix_(g1_indices, g2_indices)],
            d0res_byres[g1][g2][g1_indices][:, np.newaxis],
        )
        ipsae_d0res_byres[g1][g2][g1_indices] = np.ma.masked_where(
            ~valid_pairs_mask, ptm_matrix_d0res
        ).mean(axis=1)

        # Output line generation
        for i in g1_indices:
            by_res_lines.append(
                PerResScoreResults(
                    i=int(i + 1),
                    AlignChn=str(residues[i].chainid),
                    ScoredChn=str(g2),
                    AlignResNum=residues[i].resnum,
                    AlignResType=residues[i].res,
                    AlignRespLDDT=float(plddt[i]),
                    n0chn=int(n0chn[g1][g2]),
                    n0dom=int(n0dom[g1][g2]),
                    n0res=int(n0res_byres[g1][g2][i]),
                    d0chn=d0chn[g1][g2],
                    d0dom=d0dom[g1][g2],
                    d0res=float(d0res_byres[g1][g2][i]),
                    pTM_pae=float(iptm_d0chn_byres[g1][g2][i]),
                    pSAE_d0chn=float(ipsae_d0chn_byres[g1][g2][i]),
                    pSAE_d0dom=float(ipsae_d0dom_byres[g1][g2][i]),
                    pSAE=float(ipsae_d0res_byres[g1][g2][i]),
                )
            )

    # Aggregate results (Asym and Max)
    # We need to store these to generate the summary table

    # Store results in a structured way
    chain_pair_scores, pymol_lines, results_metrics = aggregate_byres_scores(
        residues,
        pae_cutoff,
        dist_cutoff,
        chain_pairs,
        ipsae_d0res_byres,
        ipsae_d0chn_byres,
        ipsae_d0dom_byres,
        iptm_d0chn_byres,
        n0res_byres,
        d0res_byres,
        unique_residues_chain1,
        unique_residues_chain2,
        dist_unique_residues_chain1,
        dist_unique_residues_chain2,
        pae_data,
        pDockQ,
        pDockQ2,
        LIS,
        n0chn,
        n0dom,
        d0chn,
        d0dom,
        label,
    )

    return ScoreResults(
        ipsae_scores=ipsae_d0res_byres,
        iptm_scores=iptm_d0chn_byres,
        pdockq_scores=pDockQ,
        pdockq2_scores=pDockQ2,
        lis_scores=LIS,
        metrics=results_metrics,
        by_res_scores=by_res_lines,
        chain_pair_scores=chain_pair_scores,
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
    # Append to file if it exists, since we may be processing multiple models
    # or comparing different input parameters
    chain_pair_scores_file = Path(f"{output_prefix}.txt")
    if chain_pair_scores_file.exists():
        existing_chain_pair_lines = set(
            chain_pair_scores_file.read_text().strip().splitlines()
        )
    else:
        existing_chain_pair_lines = set()
        chain_pair_scores_file.write_text("\n" + ChainPairScoreResults.header_line())
    with chain_pair_scores_file.open("a") as f:
        for summary in results.chain_pair_scores:
            line_str = summary.to_formatted_line()
            if line_str not in existing_chain_pair_lines:
                f.write(f"{line_str}\n")
                # Add blank line after max rows to separate chain pair groups
                if summary.Type == "max":
                    f.write("\n")

    # For per-residue scores and PyMOL scripts, overwrite each time
    with Path(f"{output_prefix}_byres.txt").open("w") as f:
        f.write(PerResScoreResults.header_line())
        f.writelines(
            res_line.to_formatted_line(end="\n") for res_line in results.by_res_scores
        )

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
    chain_groups: str | None


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
    parser.add_argument(
        "-g",
        "--chain-groups",
        help=(
            "Calculate scores between specified chain groups instead of all pairs. "
            "Format: 'group1/group2,group1/group3,...' where groups can contain "
            "multiple chains joined with '+'. Use '...' to include all default "
            "individual chain permutations. Example: 'A/H+L,...' calculates "
            "scores between chain A and chains H+L, plus all individual chain pairs."
        ),
        default=None,
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
        chain_groups=input_args.chain_groups,
    )


def ipsae(
    pae_file: Path,
    structure_file: Path,
    pae_cutoff: float,
    dist_cutoff: float,
    model_type: str,
    chain_groups: str | None = None,
) -> ScoreResults:
    """Calculate ipSAE, pDockQ, pDockQ2, and LIS scores for protein structure models.

    Args:
        pae_file: Path to the PAE file (json, npz, pkl).
        structure_file: Path to the structure file (pdb, cif).
        pae_cutoff: Cutoff for PAE to consider a residue pair "good".
        dist_cutoff: Distance cutoff for contact definition.
        model_type: Type of the model: af2, af3, boltz1.
        chain_groups: Optional string to parse chain groups from. If provided,
            this takes precedence over chain_groups. Use "..." to include all
            default individual chain permutations. Default is None, which behaves
            the same as passing "...".

    Returns:
        A ScoreResults object containing all calculated scores and output strings.
        The main attributes are chain_pair_scores, by_res_scores, and pymol_script.
    """
    # Load data
    structure_data = load_structure(structure_file)
    pae_data = load_pae_data(pae_file, structure_data, model_type)

    # Parse chain groups if string is provided
    if chain_groups is None:
        chain_groups = "..."
    parsed_chain_groups = parse_chain_groups(chain_groups, structure_data.unique_chains)

    # Calculate scores and dump to files
    pdb_stem = structure_file.stem
    results = calculate_scores(
        structure_data, pae_data, pae_cutoff, dist_cutoff, pdb_stem, parsed_chain_groups
    )
    return results


def main():
    """Entry point for the script.

    Parses command line arguments, loads data, calculates scores, and writes outputs.
    """
    args = parse_cli_args()
    logger.debug(f"Parsed CLI args: {args}")
    logger.info(f"Detected model type: {args.model_type}")
    if args.chain_groups:
        logger.info(f"Chain groups: {args.chain_groups}")
    scores = ipsae(
        pae_file=args.pae_file,
        structure_file=args.structure_file,
        pae_cutoff=args.pae_cutoff,
        dist_cutoff=args.dist_cutoff,
        model_type=args.model_type,
        chain_groups=args.chain_groups,
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        pae_str = str(int(args.pae_cutoff)).zfill(2)
        dist_str = str(int(args.dist_cutoff)).zfill(2)
        pdb_stem = args.structure_file.stem
        output_prefix = args.output_dir / f"{pdb_stem}_{pae_str}_{dist_str}"
        write_outputs(scores, output_prefix)
        logger.info(
            f"Success! Outputs written to {output_prefix}{{.txt,_byres.txt,.pml}}"
        )
    else:
        # Print summary to stdout
        print("#" * 90 + "\n# Per-residue scores\n" + "#" * 90)
        print(PerResScoreResults.header_line())
        print("\n".join(x.to_formatted_line() for x in scores.by_res_scores))

        print("\n\n" + "#" * 90 + "\n# Summary\n" + "#" * 90)
        print("\n" + ChainPairScoreResults.header_line(), end="")
        for summary in scores.chain_pair_scores:
            line_end = "\n" if summary.Type == "max" else ""
            print(summary.to_formatted_line(end="\n"), end=line_end)

        print("\n\n" + "#" * 90 + "\n# PyMOL script\n" + "#" * 90)
        print("".join(scores.pymol_script))


if __name__ == "__main__":
    main()
