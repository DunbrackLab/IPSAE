# ipSAE
Scoring function for interprotein interactions in AlphaFold, ESMfold2, and Boltz models.

Calculates the **ipSAE** score (https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2) as well as **ipTM**, **pDockQ**, **pDockQ2**, and **LIS** for every chain pair in a predicted structure.

# Installation

ipSAE is an installable Python package (requires Python >= 3.8 and numpy):

    pip install .

or directly from GitHub:

    pip install git+https://github.com/DunbrackLab/IPSAE.git

This installs the `ipsae` console command and the `ipsae` Python library.

Alternatively, the standalone script still works without installation — simply download `ipsae.py` and run it with Python (only numpy is required: `pip install numpy`).

# Command-line usage

All of the following are equivalent (`ipsae` console command, `python -m ipsae`, or the legacy script):

AlphaFold2:

     ipsae <path_to_af2_json_file> <path_to_af2_pdb_file> <pae_cutoff> <dist_cutoff>
     ipsae RAF1_KSR1_scores_rank_001_alphafold2_multimer_v3_model_4_seed_003.json RAF1_KSR1_unrelaxed_rank_001_alphafold2_multimer_v3_model_4_seed_003.pdb 15 15

AlphaFold3:

     ipsae <path_to_af3_json_file> <path_to_af3_cif_file> <pae_cutoff> <dist_cutoff>
     ipsae fold_aurka_tpx2_full_data_0.json fold_aurka_tpx2_model_0.cif 10 10

ESMfold2:

     ipsae <path_to_esmfold2_pae_json_file> <path_to_esmfold2_cif_file> <pae_cutoff> <dist_cutoff>
     ipsae esmfold2_complex_pae.json esmfold2_complex.cif 10 10

Boltz1/Boltz2:

     ipsae <path_to_boltz_pae_npz_file> <path_to_boltz_cif_file> <pae_cutoff> <dist_cutoff>
     ipsae pae_AURKA_TPX2_model_0.npz  AURKA_TPX2_model_0.cif 10 10

The legacy invocation is unchanged:

     python ipsae.py <path_to_af2_json_file> <path_to_af2_pdb_file> <pae_cutoff> <dist_cutoff>

Options:

* `--model <af2|af3|boltz2|esmfold2>` — explicitly set the prediction source (recommended for `.cif + .json` inputs)
* `--csv` — additionally write the chain-pair scores as a machine-readable CSV file
* `--version` — print the version and exit
* `-h, --help` — print usage information

Gzipped PAE files (e.g. `scores.json.gz` as produced by ColabFold) are read transparently. All output files are written to the same folder as the CIF/PDB file.
For `.cif + .json` inputs, use `--model` to avoid ambiguity between AF3 and ESMfold2. If omitted, ipSAE falls back to filename/schema heuristics. ESMfold2 pairwise chain ipTM values are not expected and default to 0. ESMfold2 PAE JSON can be either a keyed object (`predicted_aligned_error`/`pae`) or a raw top-level PAE matrix array.

# Python API

```python
from ipsae import score_interactions

## Automatic model detection:
result = score_interactions("fold_aurka_tpx2_full_data_0.json",
                            "fold_aurka_tpx2_model_0.cif",
                            pae_cutoff=10, dist_cutoff=10)

## Explicit model source ('af2', 'af3', 'boltz2', or 'esmfold2'):
result = score_interactions("esmfold2_complex_pae.json",
                            "esmfold2_complex.cif",
                            pae_cutoff=10, dist_cutoff=10,
                            model="esmfold2")

## chain-pair records (asym and max rows, same values as the .txt output)
for record in result.chain_pairs:
    print(record["Chn1"], record["Chn2"], record["Type"], record["ipSAE"])

## direct score lookup: metric is any output column (ipSAE, ipSAE_d0chn,
## ipSAE_d0dom, ipTM_af, ipTM_d0chn, pDockQ, pDockQ2, LIS, ...)
print(result.get_score("A", "B", "ipSAE"))                    # max of both directions
print(result.get_score("A", "B", "ipSAE", score_type="asym")) # A-aligned direction

## by-residue records (same values as the _byres.txt output)
print(result.residues[0])

## write the standard output files and/or a CSV
result.write_outputs()
result.to_csv()
```

# What's new in version 4.1

* Installable package (`pip install .`) with an `ipsae` console command, `python -m ipsae`, and a programmatic API; the original `python ipsae.py` interface is fully preserved and produces byte-identical output files.
* Optional CSV output (`--csv` or `result.to_csv()`).
* Transparent support for gzipped PAE JSON files.
* ~10x faster on large complexes (vectorized pTM transform).
* Bug fixes in the scoring logic:
  * the "max" rows are now correct when chains appear in non-alphabetical order in the structure file (previously one direction could be missed);
  * Boltz confidence files without a `pair_chains_iptm` entry no longer crash the script;
  * companion files (Boltz pLDDT/confidence, AF3 summary confidences) are located correctly when directory names contain "pae" or "confidences";
  * AlphaFold2 PAE files missing the `pae` key and PAE matrices that do not match the structure size now produce clear error messages.
* Regression test suite (`pip install -e ".[test]" && pytest`) verifying byte-identical outputs against the original script.

# Output chain-chain score file

AlphaFold2 complex of RAF1 (chain A), KSR1 (chain B), and MEK1 (chain C). RAF1 and KSR1 kinase domains make pseudosymmetric dimer, similar to BRAF homodimer. In this model, MEK1 kinase domain interacts directly with RAF1 kinase domain (AF3 places MEK1 on KSR1; both complexes exist in PDB). 

    Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS       n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model
    A    B     15   15   asym  0.647603    0.868775    0.753413    0.570    0.378501      0.2342     0.0142     0.4128     290   1571    553    6.26   12.57    8.29    261     292      92      93   RAF1_KSR1_MEK1_9f755_rank001_iptm0.57_ptm0.54_nrx_4_000
    B    A     15   15   asym  0.641387    0.867977    0.749378    0.570    0.482265      0.2342     0.0143     0.3782     285   1571    543    6.21   12.57    8.22    255     288      95      94   RAF1_KSR1_MEK1_9f755_rank001_iptm0.57_ptm0.54_nrx_4_000
    A    B     15   15   max   0.647603    0.868775    0.753413    0.570    0.482265      0.2342     0.0143     0.3955     290   1571    553    6.26   12.57    8.29    288     292      94      95   RAF1_KSR1_MEK1_9f755_rank001_iptm0.57_ptm0.54_nrx_4_000
    
    A    C     15   15   asym  0.666048    0.810562    0.750777    0.570    0.765550      0.3016     0.1158     0.3642     364   1041    635    6.93   10.71    8.77    270     365     114     105   RAF1_KSR1_MEK1_9f755_rank001_iptm0.57_ptm0.54_nrx_4_000
    C    A     15   15   asym  0.626076    0.818914    0.749619    0.570    0.438184      0.3016     0.0912     0.3470     284   1041    597    6.20   10.71    8.55    313     284      96     112   RAF1_KSR1_MEK1_9f755_rank001_iptm0.57_ptm0.54_nrx_4_000
    A    C     15   15   max   0.666048    0.818914    0.750777    0.570    0.765550      0.3016     0.1158     0.3556     364   1041    635    6.93   10.71    8.77    284     365     114     105   RAF1_KSR1_MEK1_9f755_rank001_iptm0.57_ptm0.54_nrx_4_000
    
    B    C     15   15   asym  0.323178    0.576574    0.413315    0.570    0.542858      0.0000     0.0000     0.1329     352   1316    563    6.83   11.74    8.35    207     356       0       0   RAF1_KSR1_MEK1_9f755_rank001_iptm0.57_ptm0.54_nrx_4_000
    C    B     15   15   asym  0.323743    0.619089    0.457644    0.570    0.287348      0.0000     0.0000     0.1335     288   1316    566    6.24   11.74    8.37    277     289       0       0   RAF1_KSR1_MEK1_9f755_rank001_iptm0.57_ptm0.54_nrx_4_000
    B    C     15   15   max   0.323743    0.619089    0.457644    0.570    0.542858      0.0000     0.0000     0.1332     288   1316    566    6.24   11.74    8.37    289     356       0       0   RAF1_KSR1_MEK1_9f755_rank001_iptm0.57_ptm0.54_nrx_4_000



**Chn1**=first chain

**Chn2**=second chain

**PAE**=PAE cutoff

**Dist**=Distance cutoff for CA-CA contacts

**Type**="asym" or "max"; asym means asymmetric ipTM/ipSAE values; max is maximum of asym values

**ipSAE**=ipSAE value for given PAE cutoff and d0 determined by number of residues in 2nd chain with PAE<cutoff 

**ipSAE_d0chn**=ipSAE calculated with PAE cutoff and d0 = sum of chain lengths

**ipSAE_d0dom**=ipSAE calculated with PAE cutoff and d0 = total number of residues in both chains with any interchain PAE<cutoff

**ipTM_af**=AlphaFold ipTM values. For AF2, this is for whole complex from json file. For AF3, this is symmetric pairwise value from summary json file.   

**ipTM_d0chn**=ipTM (no PAE cutoff) calculated from PAE matrix and d0 = sum of chain lengths 

**pDockQ**=score from pLDDTs from Bryant, Pozotti, and Eloffson. https://www.nature.com/articles/s41467-022-28865-w

**pDockQ2**=score based on PAE, calculated pairwise: from Zhu, Shenoy, Kundrotas, Elofsson. https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714

**LIS**=Local Interaction Score based on transform of PAEs from Kim, Hu, Comjean, Rodiger, Mohr, Perrimon. https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1

**n0res**=number of residues for d0 in ipSAE calculation

**n0chn**=number of residues in d0 in ipSAE_d0chn calculation

**n0dom**=number of residues in d0 in ipSAE_d0dom calculation

**d0res**=d0 for ipSAE

**d0chn**=d0 for ipSAE_d0chn

**d0dom**=d0 for ipSAE_d0dom

**nres1**=number of residues in chain1 with PAE<cutoff with residues in chain2

**nres2**=number of residues in chain2 with PAE<cutoff with residues in chain1

**dist1**=number of residues in chain 1 with PAE<cutoff and dist<cutoff from chain2

**dist2**=number of residues in chain 2 with PAE<cutoff and dist<cutoff from chain1

**pdb**=AlphaFold filename

# Output by-residue score file

    i   AlignChn ScoredChain  AlignResNum  AlignResType  AlignRespLDDT      n0chn  n0dom  n0res    d0chn     d0dom     d0res    ptm_pae   psae_d0chn  psae_d0dom  psae_d0res 
    1       A         B            1           MET           16.32          1571    541      0    12.569     8.210     1.001     0.1362      0.0000      0.0000      0.0000
    2       A         B            2           GLU           22.12          1571    541      0    12.569     8.210     1.001     0.1362      0.0000      0.0000      0.0000
    3       A         B            3           HIS           24.73          1571    541      0    12.569     8.210     1.001     0.1362      0.0000      0.0000      0.0000
    4       A         B            4           ILE           25.78          1571    541      0    12.569     8.210     1.001     0.1362      0.0000      0.0000      0.0000
    5       A         B            5           GLN           25.27          1571    541      0    12.569     8.210     1.001     0.1362      0.0000      0.0000      0.0000
    6       A         B            6           GLY           28.12          1571    541      0    12.569     8.210     1.001     0.1363      0.0000      0.0000      0.0000


**i**=residue in model (from 1 to total number of residues in model)

**AlignChn**=chainid of aligned residue (in PAE calculation)

**ScoredChn**= chainid of scored residues (with PAE less than cutoff)

**AlignResNum**=residue number of aligned residue

**AlignResType**=residue type of aligned residue (three letter code)

**AlignRespLDDT**=plDDT of aligned residue

**n0chn**=number of residues in d0 in ipSAE_d0chn calculation

**n0dom**=number of residues in d0 in ipSAE_d0dom calculation

**n0res**=number of residues for d0 in ipSAE calculation

**d0chn**=d0 for ipSAE_d0chn

**d0dom**=d0 for ipSAE_d0dom

**d0res**=d0 for ipSAE

**pTM_pae**=ipTM calculated from PAE matrix and d0 = sum of chain lengths 

**pSAE_d0chn**=residue-specific ipSAE calculated with PAE cutoff and d0 = sum of chain lengths (n0chn)

**pSAE_d0dom**=residue-specific ipSAE calculated with PAE cutoff and d0 = total number of residues in both chains with any interchain PAE<cutoff (n0dom)

**pSAE**=residue-specific ipSAE value for given PAE cutoff and d0 determined by number of residues in 2nd chain with PAE<cutoff (n0res)
