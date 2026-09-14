# scoring.py
# Core scoring logic for pairwise protein-protein interactions:
# ipSAE, ipTM (from PAE), pDockQ, pDockQ2, and LIS.
#
# Derived from ipsae.py by Roland Dunbrack, Fox Chase Cancer Center.
# ipSAE:   https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2
# pDockQ:  Bryant, Pozotti, and Elofsson. https://www.nature.com/articles/s41467-022-28865-w
# pDockQ2: Zhu, Shenoy, Kundrotas, Elofsson. https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714
# LIS:     Kim, Hu, Comjean, Rodiger, Mohr, Perrimon. https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1
# MIT license: script can be modified and redistributed for non-commercial and
# commercial use, as long as this information is reproduced.

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .utils import (
    init_chainpairdict_npzeros,
    init_chainpairdict_set,
    init_chainpairdict_zeros,
)

# CB-CB (CA for GLY) distance cutoff defining interface contacts for pDockQ/pDockQ2
PDOCKQ_DISTANCE_CUTOFF = 8.0

# PAE cutoff used by the Local Interaction Score (LIS)
LIS_PAE_CUTOFF = 12.0

# d0 used for nucleic-acid-containing chain pairs (approximately 21 base pairs)
D0_NUCLEIC_ACID = 2.0


def ptm_func(x, d0):
    """TM-score/pTM kernel: 1 / (1 + (x/d0)^2). Works on scalars and numpy arrays."""
    return 1.0 / (1 + (x / d0) ** 2.0)


# Backwards-compatible alias for the old np.vectorize'd version; ptm_func is
# already vectorized because it only uses elementwise numpy operations.
ptm_func_vec = ptm_func


def calc_d0(L, pair_type):
    """d0 from Yang and Skolnick, PROTEINS 57:702-710 (2004); minimum value = 1.0
    (2.0 for nucleic acid pairs)."""
    L = float(L)
    min_value = 1.0
    if pair_type == 'nucleic_acid':
        min_value = 2.0
    if L > 27:
        d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    else:
        d0 = 1.0
    return max(min_value, d0)


def calc_d0_array(L, pair_type):
    """Array version of calc_d0; accepts any array-like of residue counts.

    Note: this reproduces the original script exactly, including a small
    inconsistency with the scalar calc_d0 at L == 27 (array: 1.0385,
    scalar: 1.0). The clamp below maps every L <= 26 to d0 = 1.0, but the
    formula is still applied at L == 27, whereas the scalar version returns
    the minimum for L <= 27. Kept as-is so that by-residue output files
    remain byte-identical to the original ipsae.py.
    """
    # Convert L to a NumPy array if it isn't already one (enables flexibility in input types)
    L = np.array(L, dtype=float)
    L = np.maximum(26, L)
    min_value = 1.0

    if pair_type == 'nucleic_acid':
        min_value = 2.0

    # Calculate d0 using the vectorized operation
    return np.maximum(min_value, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


@dataclass
class ScoreResults:
    """All per-chain-pair and per-residue scores computed from one model.

    Nomenclature:
      iptm_d0chn  = ipTM  from PAEs with no PAE cutoff; d0 from n0chn = len(chain1) + len(chain2)
      ipsae_d0chn = ipSAE from PAEs with PAE cutoff;    d0 from n0chn = len(chain1) + len(chain2)
      ipsae_d0dom = ipSAE from PAEs with PAE cutoff;    d0 from number of residues in chain1 and
                    chain2 with interchain PAE < cutoff
      ipsae_d0res = ipSAE from PAEs with PAE cutoff;    d0 from number of residues in chain2 with
                    interchain PAE < cutoff given each aligned residue in chain1

    For each score there is (for example):
      ipsae_d0res_byres   = by-residue array
      ipsae_d0res_asym    = asymmetric pair value (A->B is different from B->A)
      ipsae_d0res_max     = maximum of A->B and B->A values
      ipsae_d0res_asymres = identity of residue that provides each asym maximum
      ipsae_d0res_maxres  = identity of residue that provides each maximum over both chains

    n0chn = number of residues in chain pair = len(chain1) + len(chain2)
    n0dom = number of residues in chain pair with good PAE values (< cutoff)
    n0res = number of residues in chain2 with good PAE values for each residue of chain1
    """

    pae_cutoff: float
    dist_cutoff: float

    iptm_d0chn_byres: Dict = field(repr=False, default=None)
    ipsae_d0chn_byres: Dict = field(repr=False, default=None)
    ipsae_d0dom_byres: Dict = field(repr=False, default=None)
    ipsae_d0res_byres: Dict = field(repr=False, default=None)

    iptm_d0chn_asym: Dict = None
    ipsae_d0chn_asym: Dict = None
    ipsae_d0dom_asym: Dict = None
    ipsae_d0res_asym: Dict = None

    iptm_d0chn_max: Dict = None
    ipsae_d0chn_max: Dict = None
    ipsae_d0dom_max: Dict = None
    ipsae_d0res_max: Dict = None

    iptm_d0chn_asymres: Dict = None
    ipsae_d0chn_asymres: Dict = None
    ipsae_d0dom_asymres: Dict = None
    ipsae_d0res_asymres: Dict = None

    iptm_d0chn_maxres: Dict = None
    ipsae_d0chn_maxres: Dict = None
    ipsae_d0dom_maxres: Dict = None
    ipsae_d0res_maxres: Dict = None

    n0chn: Dict = None
    n0dom: Dict = None
    n0dom_max: Dict = None
    n0res: Dict = None
    n0res_max: Dict = None
    n0res_byres: Dict = field(repr=False, default=None)

    d0chn: Dict = None
    d0dom: Dict = None
    d0dom_max: Dict = None
    d0res: Dict = None
    d0res_max: Dict = None
    d0res_byres: Dict = field(repr=False, default=None)

    valid_pair_counts: Dict = None
    dist_valid_pair_counts: Dict = None
    unique_residues_chain1: Dict = None
    unique_residues_chain2: Dict = None
    dist_unique_residues_chain1: Dict = None
    dist_unique_residues_chain2: Dict = None

    pDockQ: Dict = None
    pDockQ2: Dict = None
    LIS: Dict = None


def compute_pdockq(structure, confidence):
    """pDockQ for every ordered chain pair, plus the interface residue sets."""
    numres = structure.numres
    chains = structure.chains
    unique_chains = structure.unique_chains
    distances = structure.distances
    cb_plddt = confidence.cb_plddt

    pDockQ_unique_residues = init_chainpairdict_set(unique_chains)
    pDockQ = init_chainpairdict_zeros(unique_chains)

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            npairs = 0
            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid_pairs = (chains == chain2) & (distances[i] <= PDOCKQ_DISTANCE_CUTOFF)
                npairs += np.sum(valid_pairs)
                if valid_pairs.any():
                    pDockQ_unique_residues[chain1][chain2].add(i)
                    chain2residues = np.where(valid_pairs)[0]

                    for residue in chain2residues:
                        pDockQ_unique_residues[chain1][chain2].add(residue)

            if npairs > 0:
                mean_plddt = cb_plddt[list(pDockQ_unique_residues[chain1][chain2])].mean()
                x = mean_plddt * math.log10(npairs)
                pDockQ[chain1][chain2] = 0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018
            else:
                pDockQ[chain1][chain2] = 0.0

    return pDockQ, pDockQ_unique_residues


def compute_pdockq2(structure, confidence, pDockQ_unique_residues):
    """pDockQ2 for every ordered chain pair."""
    numres = structure.numres
    chains = structure.chains
    unique_chains = structure.unique_chains
    distances = structure.distances
    pae_matrix = confidence.pae_matrix
    cb_plddt = confidence.cb_plddt

    pDockQ2 = init_chainpairdict_zeros(unique_chains)

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            npairs = 0
            pae_ptm_sum = 0.0
            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid_pairs = (chains == chain2) & (distances[i] <= PDOCKQ_DISTANCE_CUTOFF)
                if valid_pairs.any():
                    npairs += np.sum(valid_pairs)
                    pae_list = pae_matrix[i][valid_pairs]
                    pae_list_ptm = ptm_func(pae_list, 10.0)
                    pae_ptm_sum += pae_list_ptm.sum()

            if npairs > 0:
                mean_plddt = cb_plddt[list(pDockQ_unique_residues[chain1][chain2])].mean()
                mean_ptm = pae_ptm_sum / npairs
                x = mean_plddt * mean_ptm
                pDockQ2[chain1][chain2] = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
            else:
                pDockQ2[chain1][chain2] = 0.0

    return pDockQ2


def compute_lis(structure, confidence):
    """Local Interaction Score (LIS) for every ordered chain pair."""
    chains = structure.chains
    unique_chains = structure.unique_chains
    pae_matrix = confidence.pae_matrix

    LIS = init_chainpairdict_zeros(unique_chains)

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            mask = (chains[:, None] == chain1) & (chains[None, :] == chain2)  # Select residues for (chain1, chain2)
            selected_pae = pae_matrix[mask]  # Get PAE values for this pair

            if selected_pae.size > 0:  # Ensure we have values
                valid_pae = selected_pae[selected_pae < LIS_PAE_CUTOFF]  # Apply the threshold
                if valid_pae.size > 0:
                    scores = (LIS_PAE_CUTOFF - valid_pae) / LIS_PAE_CUTOFF  # Compute scores
                    avg_score = np.mean(scores)  # Average score for (chain1, chain2)
                    LIS[chain1][chain2] = avg_score
                else:
                    LIS[chain1][chain2] = 0.0  # No valid values
            else:
                LIS[chain1][chain2] = 0.0

    return LIS


def compute_scores(structure, confidence, pae_cutoff=10.0, dist_cutoff=10.0):
    """Compute all interaction scores for every chain pair of a model.

    Returns a :class:`ScoreResults` with by-residue, asymmetric (A->B and
    B->A), and pairwise-maximum values of ipSAE/ipTM, plus pDockQ, pDockQ2,
    and LIS.
    """
    numres = structure.numres
    chains = structure.chains
    unique_chains = structure.unique_chains
    distances = structure.distances
    residues = structure.residues
    chain_pair_type = structure.chain_pair_type
    pae_matrix = confidence.pae_matrix

    if pae_matrix.shape != (numres, numres):
        raise ValueError(
            f"PAE matrix shape {pae_matrix.shape} does not match the number of scored residues "
            f"({numres}); the PAE file and structure file are probably from different models.")

    results = ScoreResults(pae_cutoff=pae_cutoff, dist_cutoff=dist_cutoff)

    # pDockQ, pDockQ2, and LIS
    results.pDockQ, pDockQ_unique_residues = compute_pdockq(structure, confidence)
    results.pDockQ2 = compute_pdockq2(structure, confidence, pDockQ_unique_residues)
    results.LIS = compute_lis(structure, confidence)

    # Create dictionaries of appropriate size: top keys are chain1 and chain2 where chain1 != chain2
    iptm_d0chn_byres = init_chainpairdict_npzeros(unique_chains, numres)
    ipsae_d0chn_byres = init_chainpairdict_npzeros(unique_chains, numres)
    ipsae_d0dom_byres = init_chainpairdict_npzeros(unique_chains, numres)
    ipsae_d0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

    iptm_d0chn_asym = init_chainpairdict_zeros(unique_chains)
    ipsae_d0chn_asym = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_asym = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_asym = init_chainpairdict_zeros(unique_chains)

    iptm_d0chn_max = init_chainpairdict_zeros(unique_chains)
    ipsae_d0chn_max = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_max = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_max = init_chainpairdict_zeros(unique_chains)

    iptm_d0chn_asymres = init_chainpairdict_zeros(unique_chains)
    ipsae_d0chn_asymres = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_asymres = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_asymres = init_chainpairdict_zeros(unique_chains)

    iptm_d0chn_maxres = init_chainpairdict_zeros(unique_chains)
    ipsae_d0chn_maxres = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_maxres = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_maxres = init_chainpairdict_zeros(unique_chains)

    n0chn = init_chainpairdict_zeros(unique_chains)
    n0dom = init_chainpairdict_zeros(unique_chains)
    n0dom_max = init_chainpairdict_zeros(unique_chains)
    n0res = init_chainpairdict_zeros(unique_chains)
    n0res_max = init_chainpairdict_zeros(unique_chains)
    n0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

    d0chn = init_chainpairdict_zeros(unique_chains)
    d0dom = init_chainpairdict_zeros(unique_chains)
    d0dom_max = init_chainpairdict_zeros(unique_chains)
    d0res = init_chainpairdict_zeros(unique_chains)
    d0res_max = init_chainpairdict_zeros(unique_chains)
    d0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

    valid_pair_counts = init_chainpairdict_zeros(unique_chains)
    dist_valid_pair_counts = init_chainpairdict_zeros(unique_chains)
    unique_residues_chain1 = init_chainpairdict_set(unique_chains)
    unique_residues_chain2 = init_chainpairdict_set(unique_chains)
    dist_unique_residues_chain1 = init_chainpairdict_set(unique_chains)
    dist_unique_residues_chain2 = init_chainpairdict_set(unique_chains)

    # Calculate ipTM/ipSAE with and without PAE cutoff (d0 = d0chn)
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            # total number of residues in chain1 and chain2
            n0chn[chain1][chain2] = np.sum(chains == chain1) + np.sum(chains == chain2)
            d0chn[chain1][chain2] = calc_d0(n0chn[chain1][chain2], chain_pair_type[chain1][chain2])
            ptm_matrix_d0chn = ptm_func(pae_matrix, d0chn[chain1][chain2])

            valid_pairs_iptm = (chains == chain2)
            valid_pairs_matrix = np.outer(chains == chain1, chains == chain2) & (pae_matrix < pae_cutoff)

            for i in range(numres):

                if chains[i] != chain1:
                    continue

                valid_pairs_ipsae = valid_pairs_matrix[i]  # row for residue i of chain1
                iptm_d0chn_byres[chain1][chain2][i] = ptm_matrix_d0chn[i, valid_pairs_iptm].mean() if valid_pairs_iptm.any() else 0.0
                ipsae_d0chn_byres[chain1][chain2][i] = ptm_matrix_d0chn[i, valid_pairs_ipsae].mean() if valid_pairs_ipsae.any() else 0.0

                # Track unique residues contributing to the ipSAE for chain1,chain2
                valid_pair_counts[chain1][chain2] += np.sum(valid_pairs_ipsae)
                if valid_pairs_ipsae.any():
                    iresnum = residues[i]['resnum']
                    unique_residues_chain1[chain1][chain2].add(iresnum)
                    for j in np.where(valid_pairs_ipsae)[0]:
                        jresnum = residues[j]['resnum']
                        unique_residues_chain2[chain1][chain2].add(jresnum)

                # Track unique residues contributing to ipTM in interface
                valid_pairs = (chains == chain2) & (pae_matrix[i] < pae_cutoff) & (distances[i] < dist_cutoff)
                dist_valid_pair_counts[chain1][chain2] += np.sum(valid_pairs)

                if valid_pairs.any():
                    iresnum = residues[i]['resnum']
                    dist_unique_residues_chain1[chain1][chain2].add(iresnum)
                    for j in np.where(valid_pairs)[0]:
                        jresnum = residues[j]['resnum']
                        dist_unique_residues_chain2[chain1][chain2].add(jresnum)

    # Calculate ipSAE with PAE cutoff and d0 = d0dom (from interface size) or d0res (per aligned residue)
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            residues_1 = len(unique_residues_chain1[chain1][chain2])
            residues_2 = len(unique_residues_chain2[chain1][chain2])
            n0dom[chain1][chain2] = residues_1 + residues_2
            d0dom[chain1][chain2] = calc_d0(n0dom[chain1][chain2], chain_pair_type[chain1][chain2])

            ptm_matrix_d0dom = ptm_func(pae_matrix, d0dom[chain1][chain2])

            valid_pairs_matrix = np.outer(chains == chain1, chains == chain2) & (pae_matrix < pae_cutoff)

            n0res_byres_all = np.sum(valid_pairs_matrix, axis=1)
            d0res_byres_all = calc_d0_array(n0res_byres_all, chain_pair_type[chain1][chain2])

            n0res_byres[chain1][chain2] = n0res_byres_all
            d0res_byres[chain1][chain2] = d0res_byres_all

            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid_pairs = valid_pairs_matrix[i]
                ipsae_d0dom_byres[chain1][chain2][i] = ptm_matrix_d0dom[i, valid_pairs].mean() if valid_pairs.any() else 0.0

                ptm_row_d0res = ptm_func(pae_matrix[i], d0res_byres[chain1][chain2][i])
                ipsae_d0res_byres[chain1][chain2][i] = ptm_row_d0res[valid_pairs].mean() if valid_pairs.any() else 0.0

    # Compute interchain asymmetric ipTM and ipSAE values for each ordered chain pair.
    # This is done for *all* ordered pairs before computing pairwise maxima below, so
    # that maxima are correct regardless of the chain order in the structure file.
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            interchain_values = iptm_d0chn_byres[chain1][chain2]
            max_index = np.argmax(interchain_values)
            iptm_d0chn_asym[chain1][chain2] = interchain_values[max_index]
            iptm_d0chn_asymres[chain1][chain2] = residues[max_index]['residue']

            interchain_values = ipsae_d0chn_byres[chain1][chain2]
            max_index = np.argmax(interchain_values)
            ipsae_d0chn_asym[chain1][chain2] = interchain_values[max_index]
            ipsae_d0chn_asymres[chain1][chain2] = residues[max_index]['residue']

            interchain_values = ipsae_d0dom_byres[chain1][chain2]
            max_index = np.argmax(interchain_values)
            ipsae_d0dom_asym[chain1][chain2] = interchain_values[max_index]
            ipsae_d0dom_asymres[chain1][chain2] = residues[max_index]['residue']

            interchain_values = ipsae_d0res_byres[chain1][chain2]
            max_index = np.argmax(interchain_values)
            ipsae_d0res_asym[chain1][chain2] = interchain_values[max_index]
            ipsae_d0res_asymres[chain1][chain2] = residues[max_index]['residue']
            n0res[chain1][chain2] = n0res_byres[chain1][chain2][max_index]
            d0res[chain1][chain2] = d0res_byres[chain1][chain2][max_index]

    # Pick maximum value over the two asymmetric values for each chain pair for each ipTM/ipSAE type
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            if not chain1 > chain2:
                continue

            maxvalue = max(iptm_d0chn_asym[chain1][chain2], iptm_d0chn_asym[chain2][chain1])
            if maxvalue == iptm_d0chn_asym[chain1][chain2]:
                maxres = iptm_d0chn_asymres[chain1][chain2]
            else:
                maxres = iptm_d0chn_asymres[chain2][chain1]
            iptm_d0chn_max[chain1][chain2] = maxvalue
            iptm_d0chn_maxres[chain1][chain2] = maxres
            iptm_d0chn_max[chain2][chain1] = maxvalue
            iptm_d0chn_maxres[chain2][chain1] = maxres

            maxvalue = max(ipsae_d0chn_asym[chain1][chain2], ipsae_d0chn_asym[chain2][chain1])
            if maxvalue == ipsae_d0chn_asym[chain1][chain2]:
                maxres = ipsae_d0chn_asymres[chain1][chain2]
            else:
                maxres = ipsae_d0chn_asymres[chain2][chain1]
            ipsae_d0chn_max[chain1][chain2] = maxvalue
            ipsae_d0chn_maxres[chain1][chain2] = maxres
            ipsae_d0chn_max[chain2][chain1] = maxvalue
            ipsae_d0chn_maxres[chain2][chain1] = maxres

            maxvalue = max(ipsae_d0dom_asym[chain1][chain2], ipsae_d0dom_asym[chain2][chain1])
            if maxvalue == ipsae_d0dom_asym[chain1][chain2]:
                maxres = ipsae_d0dom_asymres[chain1][chain2]
                maxn0 = n0dom[chain1][chain2]
                maxd0 = d0dom[chain1][chain2]
            else:
                maxres = ipsae_d0dom_asymres[chain2][chain1]
                maxn0 = n0dom[chain2][chain1]
                maxd0 = d0dom[chain2][chain1]
            ipsae_d0dom_max[chain1][chain2] = maxvalue
            ipsae_d0dom_maxres[chain1][chain2] = maxres
            ipsae_d0dom_max[chain2][chain1] = maxvalue
            ipsae_d0dom_maxres[chain2][chain1] = maxres
            n0dom_max[chain1][chain2] = maxn0
            n0dom_max[chain2][chain1] = maxn0
            d0dom_max[chain1][chain2] = maxd0
            d0dom_max[chain2][chain1] = maxd0

            maxvalue = max(ipsae_d0res_asym[chain1][chain2], ipsae_d0res_asym[chain2][chain1])
            if maxvalue == ipsae_d0res_asym[chain1][chain2]:
                maxres = ipsae_d0res_asymres[chain1][chain2]
                maxn0 = n0res[chain1][chain2]
                maxd0 = d0res[chain1][chain2]
            else:
                maxres = ipsae_d0res_asymres[chain2][chain1]
                maxn0 = n0res[chain2][chain1]
                maxd0 = d0res[chain2][chain1]
            ipsae_d0res_max[chain1][chain2] = maxvalue
            ipsae_d0res_maxres[chain1][chain2] = maxres
            ipsae_d0res_max[chain2][chain1] = maxvalue
            ipsae_d0res_maxres[chain2][chain1] = maxres
            n0res_max[chain1][chain2] = maxn0
            n0res_max[chain2][chain1] = maxn0
            d0res_max[chain1][chain2] = maxd0
            d0res_max[chain2][chain1] = maxd0

    results.iptm_d0chn_byres = iptm_d0chn_byres
    results.ipsae_d0chn_byres = ipsae_d0chn_byres
    results.ipsae_d0dom_byres = ipsae_d0dom_byres
    results.ipsae_d0res_byres = ipsae_d0res_byres

    results.iptm_d0chn_asym = iptm_d0chn_asym
    results.ipsae_d0chn_asym = ipsae_d0chn_asym
    results.ipsae_d0dom_asym = ipsae_d0dom_asym
    results.ipsae_d0res_asym = ipsae_d0res_asym

    results.iptm_d0chn_max = iptm_d0chn_max
    results.ipsae_d0chn_max = ipsae_d0chn_max
    results.ipsae_d0dom_max = ipsae_d0dom_max
    results.ipsae_d0res_max = ipsae_d0res_max

    results.iptm_d0chn_asymres = iptm_d0chn_asymres
    results.ipsae_d0chn_asymres = ipsae_d0chn_asymres
    results.ipsae_d0dom_asymres = ipsae_d0dom_asymres
    results.ipsae_d0res_asymres = ipsae_d0res_asymres

    results.iptm_d0chn_maxres = iptm_d0chn_maxres
    results.ipsae_d0chn_maxres = ipsae_d0chn_maxres
    results.ipsae_d0dom_maxres = ipsae_d0dom_maxres
    results.ipsae_d0res_maxres = ipsae_d0res_maxres

    results.n0chn = n0chn
    results.n0dom = n0dom
    results.n0dom_max = n0dom_max
    results.n0res = n0res
    results.n0res_max = n0res_max
    results.n0res_byres = n0res_byres

    results.d0chn = d0chn
    results.d0dom = d0dom
    results.d0dom_max = d0dom_max
    results.d0res = d0res
    results.d0res_max = d0res_max
    results.d0res_byres = d0res_byres

    results.valid_pair_counts = valid_pair_counts
    results.dist_valid_pair_counts = dist_valid_pair_counts
    results.unique_residues_chain1 = unique_residues_chain1
    results.unique_residues_chain2 = unique_residues_chain2
    results.dist_unique_residues_chain1 = dist_unique_residues_chain1
    results.dist_unique_residues_chain2 = dist_unique_residues_chain2

    return results
