# parsers.py
# Structure-file (PDB / mmCIF) parsing for the ipsae package.
#
# Derived from ipsae.py by Roland Dunbrack, Fox Chase Cancer Center.
# https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2
# MIT license: script can be modified and redistributed for non-commercial and
# commercial use, as long as this information is reproduced.

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .utils import init_chainpairdict_zeros

# Residue tokens recognized as polymer residues (proteins + nucleic acids).
# For AF3/Boltz: used to build the mask that identifies one CA/C1' token per
# residue in the pLDDT vector and PAE matrix. Ligand atom tokens and
# non-CA-atom tokens in PTM residues (those not in RESIDUE_SET) are skipped.
RESIDUE_SET = {"ALA", "ARG", "ASN", "ASP", "CYS",
               "GLN", "GLU", "GLY", "HIS", "ILE",
               "LEU", "LYS", "MET", "PHE", "PRO",
               "SER", "THR", "TRP", "TYR", "VAL",
               "DA", "DC", "DT", "DG", "A", "C", "U", "G"}

NUC_RESIDUE_SET = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
REQUIRED_CIF_FIELDS = ('id', 'label_atom_id', 'label_comp_id', 'label_seq_id', 'Cartn_x', 'Cartn_y', 'Cartn_z')


def parse_pdb_atom_line(line):
    """Parse an ATOM/HETATM line from a legacy-PDB format file (fixed columns).

    Returns None for Boltz "LIG" ligand residues.
    """
    atom_num = line[6:11].strip()
    atom_name = line[12:16].strip()
    residue_name = line[17:20].strip()
    if residue_name == "LIG":
        return None  # ligands in Boltz PDB-format files
    chain_id = line[21].strip()
    residue_seq_num = line[22:26].strip()
    x = line[30:38].strip()
    y = line[38:46].strip()
    z = line[46:54].strip()

    # Convert string numbers to integers or floats as appropriate
    atom_num = int(atom_num)
    residue_seq_num = int(residue_seq_num)
    x = float(x)
    y = float(y)
    z = float(z)

    return {
        'atom_num': atom_num,
        'atom_name': atom_name,
        'residue_name': residue_name,
        'chain_id': chain_id,
        'residue_seq_num': residue_seq_num,
        'x': x,
        'y': y,
        'z': z
    }


def parse_cif_atom_line(line, fielddict):
    """Parse an ATOM/HETATM line from an AF3 or Boltz1/2 mmCIF file.

    ``fielddict`` maps ``_atom_site`` field names to their column positions, so
    any mmCIF field order is handled. Ligands do not have residue numbers but
    modified residues do; returns None for ligand atoms.
    """
    linelist = line.split()
    atom_num = linelist[fielddict['id']]
    atom_name = linelist[fielddict['label_atom_id']]
    residue_name = linelist[fielddict['label_comp_id']]
    if "auth_asym_id" in fielddict:
        chain_id = linelist[fielddict['auth_asym_id']]
    else:
        chain_id = linelist[fielddict['label_asym_id']]

    residue_seq_num = linelist[fielddict['label_seq_id']]
    x = linelist[fielddict['Cartn_x']]
    y = linelist[fielddict['Cartn_y']]
    z = linelist[fielddict['Cartn_z']]

    if residue_seq_num == ".":
        return None   # ligand atom

    # Convert string numbers to integers or floats as appropriate
    atom_num = int(atom_num)
    residue_seq_num = int(residue_seq_num)
    x = float(x)
    y = float(y)
    z = float(z)

    return {
        'atom_num': atom_num,
        'atom_name': atom_name,
        'residue_name': residue_name,
        'chain_id': chain_id,
        'residue_seq_num': residue_seq_num,
        'x': x,
        'y': y,
        'z': z
    }


def validate_cif_atom_site_fields(fielddict, structure_path):
    """Validate required mmCIF ``_atom_site`` fields before parsing atom rows."""
    missing = [field for field in REQUIRED_CIF_FIELDS if field not in fielddict]
    if missing:
        missing_string = ", ".join(missing)
        raise ValueError(
            f"mmCIF file is missing required _atom_site fields ({missing_string}): {structure_path}")
    if "auth_asym_id" not in fielddict and "label_asym_id" not in fielddict:
        raise ValueError(
            f"mmCIF file must contain _atom_site.auth_asym_id or _atom_site.label_asym_id: {structure_path}")


def classify_chains(chains, residue_types):
    """Classify each chain as 'protein' or 'nucleic_acid' from its residue types."""
    chain_types = {}

    # Get unique chains in order of first appearance and iterate over them
    _, first_idx = np.unique(chains, return_index=True)
    unique_chains = chains[np.sort(first_idx)]

    for chain in unique_chains:
        # Find indices where the current chain is located
        indices = np.where(chains == chain)[0]
        # Get the residues for these indices
        chain_residues = residue_types[indices]
        # Count nucleic acid residues
        nuc_count = sum(residue in NUC_RESIDUE_SET for residue in chain_residues)

        # Determine if the chain is a nucleic acid or protein
        chain_types[chain] = 'nucleic_acid' if nuc_count > 0 else 'protein'

    return chain_types


@dataclass
class Structure:
    """Parsed structural model: residues, chains, coordinates, and distances."""

    file_format: str                 # 'pdb' or 'cif'
    residues: List[dict]             # one entry per CA (or C1') token residue
    cb_residues: List[dict]          # one entry per CB (or C3'; CA for GLY) residue
    chains: np.ndarray               # chain id per residue
    unique_chains: np.ndarray        # chain ids in order of first appearance
    token_array: np.ndarray          # AF3/Boltz token mask (1 = residue token)
    residue_types: np.ndarray        # residue name per residue
    numres: int
    CA_atom_num: np.ndarray          # 0-based atom index of each CA atom
    CB_atom_num: np.ndarray          # 0-based atom index of each CB atom
    coordinates: np.ndarray          # CB coordinates used for contact distances
    distances: np.ndarray            # numres x numres CB-CB distance matrix
    chain_types: Dict[str, str]      # chain -> 'protein' | 'nucleic_acid'
    chain_pair_type: Dict[str, Dict[str, str]]  # 'nucleic_acid' if either chain is NA

    @property
    def ntokens(self):
        return int(np.sum(self.token_array))


def load_structure(structure_path, file_format=None):
    """Load residues from an AlphaFold/Boltz PDB or mmCIF file into a Structure.

    Reads CA (C1' for nucleic acids) and CB (C3'; CA for GLY) atoms, converts
    them to numpy arrays, and computes the CB-CB distance matrix. Also builds
    the AF3/Boltz token mask used to subset per-token pLDDT/PAE data.
    """
    if file_format is None:
        if ".pdb" in structure_path:
            file_format = 'pdb'
        elif ".cif" in structure_path:
            file_format = 'cif'
        else:
            raise ValueError(f"Cannot determine structure file format (expected .pdb or .cif): {structure_path}")
    cif = (file_format == 'cif')

    residues = []
    cb_residues = []
    chains = []
    atomsitefield_num = 0
    atomsitefield_dict = {}  # order of atom_site fields in mmCIF files; handles any mmCIF field order
    atomsite_validated = False

    # For af3 and boltz: mask identifying CA atom tokens in plddt vector and pae matrix;
    # skip ligand atom tokens and non-CA-atom tokens in PTMs (those not in RESIDUE_SET)
    token_mask = []

    with open(structure_path, 'r') as PDB:
        for line in PDB:

            if line.startswith("_atom_site."):
                line = line.strip()
                (atomsite, fieldname) = line.split(".")
                atomsitefield_dict[fieldname] = atomsitefield_num
                atomsitefield_num += 1
                continue

            if line.startswith("ATOM") or line.startswith("HETATM"):
                if cif:
                    if not atomsite_validated:
                        validate_cif_atom_site_fields(atomsitefield_dict, structure_path)
                        atomsite_validated = True
                    atom = parse_cif_atom_line(line, atomsitefield_dict)
                else:
                    atom = parse_pdb_atom_line(line)
                if atom is None:  # ligand atom
                    token_mask.append(0)
                    continue

                if atom['atom_name'] == "CA" or "C1" in atom['atom_name']:
                    token_mask.append(1)
                    residues.append({
                        'atom_num': atom['atom_num'],
                        'coor': np.array([atom['x'], atom['y'], atom['z']]),
                        'res': atom['residue_name'],
                        'chainid': atom['chain_id'],
                        'resnum': atom['residue_seq_num'],
                        'residue': f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}"
                    })
                    chains.append(atom['chain_id'])

                if atom['atom_name'] == "CB" or "C3" in atom['atom_name'] or (atom['residue_name'] == "GLY" and atom['atom_name'] == "CA"):
                    cb_residues.append({
                        'atom_num': atom['atom_num'],
                        'coor': np.array([atom['x'], atom['y'], atom['z']]),
                        'res': atom['residue_name'],
                        'chainid': atom['chain_id'],
                        'resnum': atom['residue_seq_num'],
                        'residue': f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}"
                    })

                # add nucleic acids and non-CA atoms in PTM residues to tokens (as 0),
                # whether labeled as "HETATM" (af3) or as "ATOM" (boltz)
                if atom['atom_name'] != "CA" and "C1" not in atom['atom_name'] and atom['residue_name'] not in RESIDUE_SET:
                    token_mask.append(0)

    if cif and atomsitefield_dict and not atomsite_validated:
        validate_cif_atom_site_fields(atomsitefield_dict, structure_path)

    numres = len(residues)
    if numres == 0:
        raise ValueError(f"No polymer residues found in structure file: {structure_path}")
    if len(cb_residues) != numres:
        raise ValueError(
            f"Mismatch between CA residues ({numres}) and CB residues ({len(cb_residues)}) "
            f"in {structure_path}; cannot compute the residue-residue distance matrix.")

    # Convert structure information to numpy arrays
    CA_atom_num = np.array([res['atom_num'] - 1 for res in residues])     # for AF3 atom indexing from 0
    CB_atom_num = np.array([res['atom_num'] - 1 for res in cb_residues])  # for AF3 atom indexing from 0
    coordinates = np.array([res['coor'] for res in cb_residues])
    chains = np.array(chains)

    _, first_idx = np.unique(chains, return_index=True)
    unique_chains = chains[np.sort(first_idx)]
    token_array = np.array(token_mask)
    residue_types = np.array([res['res'] for res in residues])

    # chain types (nucleic acid (NA) or protein) and chain_pair_types
    # ('nucleic_acid' if either chain is NA) for d0 calculation
    chain_types = classify_chains(chains, residue_types)
    chain_pair_type = init_chainpairdict_zeros(unique_chains)
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            if chain_types[chain1] == 'nucleic_acid' or chain_types[chain2] == 'nucleic_acid':
                chain_pair_type[chain1][chain2] = 'nucleic_acid'
            else:
                chain_pair_type[chain1][chain2] = 'protein'

    # Calculate distance matrix using NumPy broadcasting
    distances = np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2).sum(axis=2))

    return Structure(
        file_format=file_format,
        residues=residues,
        cb_residues=cb_residues,
        chains=chains,
        unique_chains=unique_chains,
        token_array=token_array,
        residue_types=residue_types,
        numres=numres,
        CA_atom_num=CA_atom_num,
        CB_atom_num=CB_atom_num,
        coordinates=coordinates,
        distances=distances,
        chain_types=chain_types,
        chain_pair_type=chain_pair_type,
    )
