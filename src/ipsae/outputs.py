# outputs.py
# Chain-pair record building and output-file writers (.txt, _byres.txt, .pml, .csv)
# for the ipsae package. File formats match the original ipsae.py script exactly.
#
# Derived from ipsae.py by Roland Dunbrack, Fox Chase Cancer Center.
# https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2
# MIT license: script can be modified and redistributed for non-commercial and
# commercial use, as long as this information is reproduced.

import csv

from .utils import contiguous_ranges

CHAIN_COLORS = {'A': 'magenta',   'B': 'marine',   'C': 'lime',        'D': 'orange',
                'E': 'yellow',    'F': 'cyan',     'G': 'lightorange', 'H': 'pink',
                'I': 'deepteal',  'J': 'forest',   'K': 'lightblue',   'L': 'slate',
                'M': 'violet',    'N': 'arsenic',  'O': 'iodine',      'P': 'silver',
                'Q': 'red',       'R': 'sulfur',   'S': 'purple',      'T': 'olive',
                'U': 'palegreen', 'V': 'green',    'W': 'blue',        'X': 'palecyan',
                'Y': 'limon',     'Z': 'chocolate'}

SCORES_HEADER = ("Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn"
                 "     pDockQ     pDockQ2    LIS       n0res  n0chn  n0dom   d0res   d0chn   d0dom"
                 "  nres1   nres2   dist1   dist2  Model\n")

PML_HEADER = ("# Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn"
              "     pDockQ     pDockQ2    LIS      n0res  n0chn  n0dom   d0res   d0chn   d0dom"
              "  nres1   nres2   dist1   dist2  Model\n")

BYRES_HEADER = ("i   AlignChn ScoredChain  AlignResNum  AlignResType  AlignRespLDDT      n0chn  n0dom"
                "  n0res    d0chn     d0dom     d0res   ipTM_pae  ipSAE_d0chn ipSAE_d0dom    ipSAE \n")

CSV_COLUMNS = ["Chn1", "Chn2", "PAE", "Dist", "Type", "ipSAE", "ipSAE_d0chn", "ipSAE_d0dom",
               "ipTM_af", "ipTM_d0chn", "pDockQ", "pDockQ2", "LIS", "n0res", "n0chn", "n0dom",
               "d0res", "d0chn", "d0dom", "nres1", "nres2", "dist1", "dist2", "Model"]


def build_chain_pair_groups(structure, confidence, results, pae_string, dist_string, model_name):
    """Build the chain-pair score records for every unordered chain pair.

    Returns a list of groups, one per unordered chain pair (sorted), each a
    list of three record dicts in output order: asym (A->B), asym (B->A), max.
    Record keys match the columns of the chain-pair score file; keys starting
    with '_' carry the PyMOL alias information.
    """
    unique_chains = structure.unique_chains

    chainpairs = set()
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 >= chain2:
                continue
            chainpairs.add(chain1 + "-" + chain2)

    groups = []
    for pairname in sorted(chainpairs):
        (chain_a, chain_b) = pairname.split("-")
        group = []
        for (chain1, chain2) in ((chain_a, chain_b), (chain_b, chain_a)):

            if chain1 in CHAIN_COLORS:
                color1 = CHAIN_COLORS[chain1]
            else:
                color1 = 'magenta'

            if chain2 in CHAIN_COLORS:
                color2 = CHAIN_COLORS[chain2]
            else:
                color2 = 'marine'

            residues_1 = len(results.unique_residues_chain1[chain1][chain2])
            residues_2 = len(results.unique_residues_chain2[chain1][chain2])
            dist_residues_1 = len(results.dist_unique_residues_chain1[chain1][chain2])
            dist_residues_2 = len(results.dist_unique_residues_chain2[chain1][chain2])
            iptm_af = confidence.get_iptm_af(chain1, chain2)

            group.append({
                "Chn1": chain1,
                "Chn2": chain2,
                "PAE": pae_string,
                "Dist": dist_string,
                "Type": "asym",
                "ipSAE": results.ipsae_d0res_asym[chain1][chain2],
                "ipSAE_d0chn": results.ipsae_d0chn_asym[chain1][chain2],
                "ipSAE_d0dom": results.ipsae_d0dom_asym[chain1][chain2],
                "ipTM_af": iptm_af,
                "ipTM_d0chn": results.iptm_d0chn_asym[chain1][chain2],
                "pDockQ": results.pDockQ[chain1][chain2],
                "pDockQ2": results.pDockQ2[chain1][chain2],
                "LIS": results.LIS[chain1][chain2],
                "n0res": int(results.n0res[chain1][chain2]),
                "n0chn": int(results.n0chn[chain1][chain2]),
                "n0dom": int(results.n0dom[chain1][chain2]),
                "d0res": results.d0res[chain1][chain2],
                "d0chn": results.d0chn[chain1][chain2],
                "d0dom": results.d0dom[chain1][chain2],
                "nres1": residues_1,
                "nres2": residues_2,
                "dist1": dist_residues_1,
                "dist2": dist_residues_2,
                "Model": model_name,
                "_color1": color1,
                "_color2": color2,
                "_resranges1": contiguous_ranges(results.unique_residues_chain1[chain1][chain2]),
                "_resranges2": contiguous_ranges(results.unique_residues_chain2[chain1][chain2]),
            })

            if chain1 > chain2:
                residues_1 = max(len(results.unique_residues_chain2[chain1][chain2]),
                                 len(results.unique_residues_chain1[chain2][chain1]))
                residues_2 = max(len(results.unique_residues_chain1[chain1][chain2]),
                                 len(results.unique_residues_chain2[chain2][chain1]))
                dist_residues_1 = max(len(results.dist_unique_residues_chain2[chain1][chain2]),
                                      len(results.dist_unique_residues_chain1[chain2][chain1]))
                dist_residues_2 = max(len(results.dist_unique_residues_chain1[chain1][chain2]),
                                      len(results.dist_unique_residues_chain2[chain2][chain1]))

                iptm_af_value = iptm_af
                pDockQ2_value = max(results.pDockQ2[chain1][chain2], results.pDockQ2[chain2][chain1])
                if confidence.model_type == 'boltz':
                    iptm_af_value = max(confidence.get_iptm_af(chain1, chain2),
                                        confidence.get_iptm_af(chain2, chain1))

                LIS_score = (results.LIS[chain1][chain2] + results.LIS[chain2][chain1]) / 2.0

                group.append({
                    "Chn1": chain2,
                    "Chn2": chain1,
                    "PAE": pae_string,
                    "Dist": dist_string,
                    "Type": "max",
                    "ipSAE": results.ipsae_d0res_max[chain1][chain2],
                    "ipSAE_d0chn": results.ipsae_d0chn_max[chain1][chain2],
                    "ipSAE_d0dom": results.ipsae_d0dom_max[chain1][chain2],
                    "ipTM_af": iptm_af_value,
                    "ipTM_d0chn": results.iptm_d0chn_max[chain1][chain2],
                    "pDockQ": results.pDockQ[chain1][chain2],
                    "pDockQ2": pDockQ2_value,
                    "LIS": LIS_score,
                    "n0res": int(results.n0res_max[chain1][chain2]),
                    "n0chn": int(results.n0chn[chain1][chain2]),
                    "n0dom": int(results.n0dom_max[chain1][chain2]),
                    "d0res": results.d0res_max[chain1][chain2],
                    "d0chn": results.d0chn[chain1][chain2],
                    "d0dom": results.d0dom_max[chain1][chain2],
                    "nres1": residues_1,
                    "nres2": residues_2,
                    "dist1": dist_residues_1,
                    "dist2": dist_residues_2,
                    "Model": model_name,
                })
        groups.append(group)
    return groups


def build_residue_records(structure, confidence, results):
    """Build the by-residue score records in the order of the _byres.txt file."""
    numres = structure.numres
    chains = structure.chains
    unique_chains = structure.unique_chains
    residues = structure.residues
    plddt = confidence.plddt

    records = []
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            for i in range(numres):
                if chains[i] != chain1:
                    continue
                records.append({
                    "i": i + 1,
                    "AlignChn": str(chain1),
                    "ScoredChain": str(chain2),
                    "AlignResNum": residues[i]['resnum'],
                    "AlignResType": residues[i]['res'],
                    "AlignRespLDDT": float(plddt[i]),
                    "n0chn": int(results.n0chn[chain1][chain2]),
                    "n0dom": int(results.n0dom[chain1][chain2]),
                    "n0res": int(results.n0res_byres[chain1][chain2][i]),
                    "d0chn": float(results.d0chn[chain1][chain2]),
                    "d0dom": float(results.d0dom[chain1][chain2]),
                    "d0res": float(results.d0res_byres[chain1][chain2][i]),
                    "ipTM_pae": float(results.iptm_d0chn_byres[chain1][chain2][i]),
                    "ipSAE_d0chn": float(results.ipsae_d0chn_byres[chain1][chain2][i]),
                    "ipSAE_d0dom": float(results.ipsae_d0dom_byres[chain1][chain2][i]),
                    "ipSAE": float(results.ipsae_d0res_byres[chain1][chain2][i]),
                })
    return records


def _format_pair_line(rec):
    """Format one chain-pair record exactly as in the original chain-pair score file."""
    return (f'{rec["Chn1"]}    {rec["Chn2"]}     {rec["PAE"]:3}  {rec["Dist"]:3}  {rec["Type"]:5} '
            f'{rec["ipSAE"]:8.6f}    '
            f'{rec["ipSAE_d0chn"]:8.6f}    '
            f'{rec["ipSAE_d0dom"]:8.6f}    '
            f'{rec["ipTM_af"]:5.3f}    '
            f'{rec["ipTM_d0chn"]:8.6f}    '
            f'{rec["pDockQ"]:8.4f}   '
            f'{rec["pDockQ2"]:8.4f}   '
            f'{rec["LIS"]:8.4f}   '
            f'{rec["n0res"]:5d}  '
            f'{rec["n0chn"]:5d}  '
            f'{rec["n0dom"]:5d}  '
            f'{rec["d0res"]:6.2f}  '
            f'{rec["d0chn"]:6.2f}  '
            f'{rec["d0dom"]:6.2f}  '
            f'{rec["nres1"]:5d}   '
            f'{rec["nres2"]:5d}   '
            f'{rec["dist1"]:5d}   '
            f'{rec["dist2"]:5d}   '
            f'{rec["Model"]}\n')


def write_pair_scores_file(groups, scores_path):
    """Write the chain-pair score file (.txt)."""
    with open(scores_path, 'w') as OUT:
        OUT.write("\n" + SCORES_HEADER)
        for group in groups:
            for rec in group:
                OUT.write(_format_pair_line(rec))
            OUT.write("\n")


def _pml_alias_lines(rec):
    """PyMOL alias command coloring the interface residues of one asym record."""
    chain1 = rec["Chn1"]
    chain2 = rec["Chn2"]
    chain_pair = f'color_{chain1}_{chain2}'
    chain1_residues = f'chain  {chain1} and resi {rec["_resranges1"]}'
    chain2_residues = f'chain  {chain2} and resi {rec["_resranges2"]}'
    return (f'alias {chain_pair}, color gray80, all; '
            f'color {rec["_color1"]}, {chain1_residues}; '
            f'color {rec["_color2"]}, {chain2_residues}\n\n')


def write_pml_file(groups, pml_path):
    """Write the PyMOL script (.pml) with commented score lines and coloring aliases."""
    with open(pml_path, 'w') as PML:
        PML.write(PML_HEADER)
        for group in groups:
            asym_recs = [rec for rec in group if rec["Type"] == "asym"]
            max_recs = [rec for rec in group if rec["Type"] == "max"]
            # Original line order per chain pair: asym A->B comment, A->B alias,
            # asym B->A comment, max comment, B->A alias.
            PML.write("# " + _format_pair_line(asym_recs[0]))
            PML.write(_pml_alias_lines(asym_recs[0]))
            for rec in asym_recs[1:]:
                PML.write("# " + _format_pair_line(rec))
            for rec in max_recs:
                PML.write("# " + _format_pair_line(rec))
            for rec in asym_recs[1:]:
                PML.write(_pml_alias_lines(rec))


def write_byres_file(structure, confidence, results, byres_path):
    """Write the by-residue score file (_byres.txt)."""
    numres = structure.numres
    chains = structure.chains
    unique_chains = structure.unique_chains
    residues = structure.residues
    plddt = confidence.plddt

    with open(byres_path, 'w') as OUT2:
        OUT2.write(BYRES_HEADER)
        for chain1 in unique_chains:
            for chain2 in unique_chains:
                if chain1 == chain2:
                    continue
                for i in range(numres):
                    if chains[i] != chain1:
                        continue
                    outstring = f'{i+1:<4d}    ' + (
                        f'{chain1:4}      '
                        f'{chain2:4}      '
                        f'{residues[i]["resnum"]:4d}           '
                        f'{residues[i]["res"]:3}        '
                        f'{plddt[i]:8.2f}         '
                        f'{int(results.n0chn[chain1][chain2]):5d}  '
                        f'{int(results.n0dom[chain1][chain2]):5d}  '
                        f'{int(results.n0res_byres[chain1][chain2][i]):5d}  '
                        f'{results.d0chn[chain1][chain2]:8.3f}  '
                        f'{results.d0dom[chain1][chain2]:8.3f}  '
                        f'{results.d0res_byres[chain1][chain2][i]:8.3f}   '
                        f'{results.iptm_d0chn_byres[chain1][chain2][i]:8.4f}    '
                        f'{results.ipsae_d0chn_byres[chain1][chain2][i]:8.4f}    '
                        f'{results.ipsae_d0dom_byres[chain1][chain2][i]:8.4f}    '
                        f'{results.ipsae_d0res_byres[chain1][chain2][i]:8.4f}\n'
                    )
                    OUT2.write(outstring)


def write_csv_file(groups, csv_path):
    """Write the chain-pair scores as a CSV file with the same values as the .txt file."""
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for group in groups:
            for rec in group:
                writer.writerow([
                    rec["Chn1"], rec["Chn2"], int(rec["PAE"]), int(rec["Dist"]), rec["Type"],
                    f'{rec["ipSAE"]:.6f}', f'{rec["ipSAE_d0chn"]:.6f}', f'{rec["ipSAE_d0dom"]:.6f}',
                    f'{rec["ipTM_af"]:.3f}', f'{rec["ipTM_d0chn"]:.6f}',
                    f'{rec["pDockQ"]:.4f}', f'{rec["pDockQ2"]:.4f}', f'{rec["LIS"]:.4f}',
                    rec["n0res"], rec["n0chn"], rec["n0dom"],
                    f'{rec["d0res"]:.2f}', f'{rec["d0chn"]:.2f}', f'{rec["d0dom"]:.2f}',
                    rec["nres1"], rec["nres2"], rec["dist1"], rec["dist2"], rec["Model"],
                ])
