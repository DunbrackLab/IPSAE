#!/usr/bin/env python
# ipsae.py
# script for calculating the ipSAE score for scoring pairwise protein-protein interactions in AlphaFold2, AlphaFold3, ESMfold2, and Boltz models
# https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2

# Also calculates:
#    pDockQ: Bryant, Pozotti, and Eloffson. https://www.nature.com/articles/s41467-022-28865-w
#    pDockQ2: Zhu, Shenoy, Kundrotas, Elofsson. https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714
#    LIS: Kim, Hu, Comjean, Rodiger, Mohr, Perrimon. https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1

# Roland Dunbrack
# Fox Chase Cancer Center
# version 4
# January 3, 2026: Fixed Boltz2 issues (PDB and mmCIF format; chainIDs)
# MIT license: script can be modified and redistributed for non-commercial and commercial use, as long as this information is reproduced.

# includes support for Boltz and ESMfold2 structures and structures with nucleic acids

# This script is a backwards-compatible wrapper around the installable "ipsae"
# package in src/ipsae. Install the package with:
#      pip install .
# after which the same interface is available as the "ipsae" console command
# or "python -m ipsae". Running this script directly still works and requires
# only numpy:
#      pip install numpy

# Usage:

#  python ipsae.py <path_to_af2_pae_file>        <path_to_af2_pdb_file>     <pae_cutoff> <dist_cutoff>
#  python ipsae.py <path_to_af3_pae_file>        <path_to_af3_cif_file>     <pae_cutoff> <dist_cutoff>
#  python ipsae.py <path_to_esmfold2_pae_file>    <path_to_esmfold2_cif_file> <pae_cutoff> <dist_cutoff>
#  python ipsae.py <path_to_boltz_pae_npz_file>  <path_to_boltz_cif_file>   <pae_cutoff> <dist_cutoff>
#  python ipsae.py <path_to_boltz_pae_npz_file>  <path_to_boltz_pdb_file>   <pae_cutoff> <dist_cutoff>
#
# All output files will be in same path/folder as cif or pdb file

import importlib
import os
import sys

# Make the bundled package importable when this script is run from a source
# checkout without installation. Inserted first so that the src/ipsae package
# takes precedence over this script for "import ipsae".
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

if __name__ == "ipsae":
    # This file was imported as "import ipsae" (repository root on sys.path).
    # Hand over to the real package in src/ipsae so that the public API and
    # submodules (ipsae.api, ipsae.cli, ...) are available.
    del sys.modules["ipsae"]
    sys.modules["ipsae"] = importlib.import_module("ipsae")
else:
    from ipsae.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
