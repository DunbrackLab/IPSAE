"""ipsae: scoring pairwise protein-protein interactions in AlphaFold2, AlphaFold3, ESMfold2, and Boltz models.

Calculates the ipSAE score (Dunbrack, https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2)
as well as:
    pDockQ:  Bryant, Pozotti, and Elofsson. https://www.nature.com/articles/s41467-022-28865-w
    pDockQ2: Zhu, Shenoy, Kundrotas, Elofsson. https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714
    LIS:     Kim, Hu, Comjean, Rodiger, Mohr, Perrimon. https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1

Original script by Roland Dunbrack, Fox Chase Cancer Center.
MIT license: can be modified and redistributed for non-commercial and
commercial use, as long as this information is reproduced.

Includes support for ESMfold2 models, Boltz structures, and structures with nucleic acids.

Basic usage::

    import ipsae

    result = ipsae.score_interactions("model_scores.json", "model.pdb", pae_cutoff=10, dist_cutoff=10)
    for record in result.chain_pairs:          # one dict per output row
        print(record["Chn1"], record["Chn2"], record["Type"], record["ipSAE"])
    result.get_score("A", "B", metric="ipSAE") # single max score for a chain pair
    result.write_outputs()                     # classic .txt, _byres.txt, and .pml files
    result.to_csv()                            # chain-pair scores as CSV
"""

__version__ = "4.1.1"

from .api import IPSAEResult, detect_file_format, detect_model_type, normalize_model_type, score_interactions
from .confidence import ConfidenceData, load_confidence
from .parsers import Structure, load_structure
from .scoring import (
    ScoreResults,
    calc_d0,
    calc_d0_array,
    compute_scores,
    ptm_func,
)

__all__ = [
    "__version__",
    "score_interactions",
    "IPSAEResult",
    "detect_model_type",
    "detect_file_format",
    "normalize_model_type",
    "load_structure",
    "Structure",
    "load_confidence",
    "ConfidenceData",
    "compute_scores",
    "ScoreResults",
    "ptm_func",
    "calc_d0",
    "calc_d0_array",
]
