# Tests for the ipsae package.
#
# The regression tests verify that the packaged code reproduces the output of
# the original monolithic ipsae.py script byte-for-byte on the bundled
# AlphaFold2 and AlphaFold3 examples (fixtures in tests/expected were
# generated with the original script).

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import ipsae
from ipsae.api import cutoff_string, detect_file_format, detect_model_type, normalize_model_type
from ipsae.cli import main as cli_main
from ipsae.scoring import calc_d0, calc_d0_array, ptm_func
from ipsae.utils import contiguous_ranges

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "RAF1_KSR1_MEK1_af3"
EXAMPLES_DIR = REPO / "examples"
EXPECTED = Path(__file__).resolve().parent / "expected"

AF3_JSON = EXAMPLE / "fold_aurka_0_tpx2_0_full_data_0.json"
AF3_CIF = EXAMPLE / "fold_aurka_0_tpx2_0_model_0.cif"
AF2_JSON_GZ = EXAMPLE / "RAF1_KSR1_MEK1_9f755_scores_alphafold2_multimer_v3_model_1_seed_000.json.gz"
AF2_PDB = EXAMPLE / "RAF1_KSR1_MEK1_9f755_unrelaxed_alphafold2_multimer_v3_model_1_seed_000.pdb"


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_ptm_func_scalar_and_array():
    assert ptm_func(0.0, 5.0) == 1.0
    assert ptm_func(5.0, 5.0) == pytest.approx(0.5)
    values = ptm_func(np.array([0.0, 5.0, 10.0]), 5.0)
    assert values == pytest.approx([1.0, 0.5, 0.2])


def test_calc_d0_minimum_values():
    # short chains are clamped to the minimum d0
    assert calc_d0(1, 'protein') == 1.0
    assert calc_d0(27, 'protein') == 1.0
    assert calc_d0(1, 'nucleic_acid') == 2.0
    # Yang & Skolnick formula above the cutoff
    assert calc_d0(100, 'protein') == pytest.approx(1.24 * (100 - 15) ** (1 / 3) - 1.8)


def test_calc_d0_array_matches_scalar_for_large_L():
    lengths = [50, 100, 500, 1571]
    array_values = calc_d0_array(lengths, 'protein')
    for length, value in zip(lengths, array_values):
        assert value == pytest.approx(calc_d0(length, 'protein'))


def test_calc_d0_array_minimum():
    values = calc_d0_array([0, 1, 26], 'protein')
    assert np.all(values >= 1.0)
    values_na = calc_d0_array([0, 1, 26], 'nucleic_acid')
    assert np.all(values_na >= 2.0)


def test_contiguous_ranges():
    assert contiguous_ranges(set()) is None
    assert contiguous_ranges({5}) == "5"
    assert contiguous_ranges({1, 2, 3, 7, 9, 10}) == "1-3+7+9-10"


def test_cutoff_string_padding():
    assert cutoff_string(5.0) == "05"
    assert cutoff_string(10.0) == "10"
    assert cutoff_string(15.0) == "15"


def test_detect_model_type():
    assert detect_model_type("scores.json", "model.pdb") == ("af2", "pdb")
    assert detect_model_type("scores.json.gz", "model.pdb") == ("af2", "pdb")
    assert detect_model_type("scores.pkl", "model.pdb") == ("af2", "pdb")
    assert detect_model_type("full_data_0.json", "model.cif") == ("af3", "cif")
    assert detect_model_type("esmfold2_complex_pae.json", "model.cif") == ("esmfold2", "cif")
    assert detect_model_type("full_data_0.json.gz", "model.cif") == ("af3", "cif")
    assert detect_model_type("pae_model.npz", "model.cif") == ("boltz", "cif")
    assert detect_model_type("pae_model.npz", "model.pdb") == ("boltz", "pdb")
    with pytest.raises(FileNotFoundError):
        detect_model_type("scores.json", "model.cif")
    with pytest.raises(ValueError):
        detect_model_type("scores.txt", "model.pdb")
    with pytest.raises(ValueError):
        detect_model_type("scores.json", "model.xyz")


def test_detect_model_type_esmfold2_pae_key_dict(tmp_path):
    pae_path = tmp_path / "esmfold2_scores.json"
    pae_path.write_text(json.dumps({"pae": [[0.2, 0.3], [0.4, 0.2]]}))
    assert detect_model_type(str(pae_path), "model.cif") == ("esmfold2", "cif")


def test_detect_model_type_unknown_cif_json_schema_raises(tmp_path):
    pae_path = tmp_path / "scores.json"
    pae_path.write_text(json.dumps({"foo": 1}))
    with pytest.raises(ValueError):
        detect_model_type(str(pae_path), "model.cif")


def test_detect_model_type_af3_marker_wins_over_pae(tmp_path):
    pae_path = tmp_path / "scores.json"
    pae_path.write_text(json.dumps({"atom_chain_ids": ["A", "A"], "pae": [[0.2, 0.3], [0.4, 0.2]]}))
    assert detect_model_type(str(pae_path), "model.cif") == ("af3", "cif")


# ---------------------------------------------------------------------------
# Regression tests against the original script's outputs
# ---------------------------------------------------------------------------

def run_cli(tmp_path, monkeypatch, pae_name, structure_name, pae, dist, extra=()):
    monkeypatch.chdir(tmp_path)
    exit_code = cli_main([pae_name, structure_name, pae, dist, *extra])
    assert exit_code == 0


def test_af3_outputs_match_original(tmp_path, monkeypatch):
    shutil.copy(AF3_JSON, tmp_path)
    shutil.copy(AF3_CIF, tmp_path)
    run_cli(tmp_path, monkeypatch, AF3_JSON.name, AF3_CIF.name, "10", "10")

    stem = "fold_aurka_0_tpx2_0_model_0_10_10"
    for suffix in (".txt", "_byres.txt", ".pml"):
        produced = (tmp_path / (stem + suffix)).read_text()
        expected = (EXPECTED / (stem + suffix)).read_text()
        assert produced == expected, f"{stem + suffix} differs from the original script output"


def test_af2_outputs_match_original(tmp_path, monkeypatch):
    with gzip.open(AF2_JSON_GZ, "rb") as handle:
        (tmp_path / AF2_JSON_GZ.name[:-3]).write_bytes(handle.read())
    shutil.copy(AF2_PDB, tmp_path)
    run_cli(tmp_path, monkeypatch, AF2_JSON_GZ.name[:-3], AF2_PDB.name, "15", "15")

    stem = "RAF1_KSR1_MEK1_9f755_unrelaxed_alphafold2_multimer_v3_model_1_seed_000_15_15"
    for suffix in (".txt", ".pml"):
        produced = (tmp_path / (stem + suffix)).read_text()
        expected = (EXPECTED / (stem + suffix)).read_text()
        assert produced == expected, f"{stem + suffix} differs from the original script output"

    produced = (tmp_path / (stem + "_byres.txt")).read_text()
    with gzip.open(EXPECTED / (stem + "_byres.txt.gz"), "rt") as handle:
        expected = handle.read()
    assert produced == expected, "by-residue file differs from the original script output"


def test_af2_gzipped_json_gives_same_scores(tmp_path, monkeypatch):
    shutil.copy(AF2_JSON_GZ, tmp_path)
    shutil.copy(AF2_PDB, tmp_path)
    run_cli(tmp_path, monkeypatch, AF2_JSON_GZ.name, AF2_PDB.name, "15", "15")

    stem = "RAF1_KSR1_MEK1_9f755_unrelaxed_alphafold2_multimer_v3_model_1_seed_000_15_15"
    produced = (tmp_path / (stem + ".txt")).read_text()
    expected = (EXPECTED / (stem + ".txt")).read_text()
    assert produced == expected


def test_csv_output(tmp_path, monkeypatch):
    shutil.copy(AF3_JSON, tmp_path)
    shutil.copy(AF3_CIF, tmp_path)
    run_cli(tmp_path, monkeypatch, AF3_JSON.name, AF3_CIF.name, "10", "10", extra=("--csv",))

    csv_path = tmp_path / "fold_aurka_0_tpx2_0_model_0_10_10.csv"
    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].startswith("Chn1,Chn2,PAE,Dist,Type,ipSAE,")
    assert len(lines) == 4  # header + asym A->B + asym B->A + max
    assert lines[1].startswith("A,B,10,10,asym,0.448952,")
    assert lines[3].startswith("A,B,10,10,max,0.866498,")


def test_api_scores(tmp_path):
    result = ipsae.score_interactions(str(AF3_JSON), str(AF3_CIF), 10, 10)

    assert len(result.chain_pairs) == 3
    assert len(result.residues) == result.structure.numres  # two chains: each aligned once

    asym_ab = result.get_score("A", "B", "ipSAE", "asym")
    asym_ba = result.get_score("B", "A", "ipSAE", "asym")
    max_ab = result.get_score("A", "B", "ipSAE", "max")
    assert max_ab == max(asym_ab, asym_ba)
    assert asym_ab == pytest.approx(0.448952, abs=1e-6)
    assert asym_ba == pytest.approx(0.866498, abs=1e-6)

    with pytest.raises(KeyError):
        result.get_score("A", "B", "not_a_metric")

    paths = result.write_outputs(output_stem=str(tmp_path / "af3_scores"))
    assert Path(paths["scores"]).exists()
    assert Path(paths["byres"]).exists()
    assert Path(paths["pml"]).exists()
    csv_path = result.to_csv(str(tmp_path / "af3_scores.csv"))
    assert Path(csv_path).exists()


# ---------------------------------------------------------------------------
# Synthetic-model tests for the fixed scoring bugs
# ---------------------------------------------------------------------------

def write_gly_pdb(path, chain_specs, spacing=2.0):
    """Write a synthetic PDB with GLY CA atoms; chain_specs = [(chain, nres), ...]."""
    lines = []
    serial = 0
    for offset, (chain, nres) in enumerate(chain_specs):
        for resnum in range(1, nres + 1):
            serial += 1
            x, y, z = float(resnum), offset * spacing, 0.0
            lines.append(
                f"ATOM  {serial:5d}  CA  GLY {chain}{resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
    path.write_text("".join(lines) + "END\n")


def write_gly_cif(path, chain_specs, spacing=2.0):
    """Write a synthetic mmCIF with one GLY CA atom per residue."""
    lines = [
        "data_model\n",
        "loop_\n",
        "_atom_site.group_PDB\n",
        "_atom_site.id\n",
        "_atom_site.type_symbol\n",
        "_atom_site.label_atom_id\n",
        "_atom_site.label_comp_id\n",
        "_atom_site.label_asym_id\n",
        "_atom_site.label_seq_id\n",
        "_atom_site.Cartn_x\n",
        "_atom_site.Cartn_y\n",
        "_atom_site.Cartn_z\n",
    ]
    serial = 0
    for offset, (chain, nres) in enumerate(chain_specs):
        for resnum in range(1, nres + 1):
            serial += 1
            x, y, z = float(resnum), offset * spacing, 0.0
            lines.append(f"ATOM {serial} C CA GLY {chain} {resnum} {x:.3f} {y:.3f} {z:.3f}\n")
    lines.append("#\n")
    path.write_text("".join(lines))


def test_max_scores_with_non_alphabetical_chain_order(tmp_path):
    """Chain B before chain A in the file: the pair maximum must still be the
    maximum over both asymmetric directions (bug in the original script)."""
    nres = 8
    pdb_path = tmp_path / "model.pdb"
    write_gly_pdb(pdb_path, [("B", nres), ("A", nres)])

    # Residues 0-7 = chain B, residues 8-15 = chain A.
    # A-aligned scores of B (rows 8-15, cols 0-7) get much better PAE than
    # B-aligned scores of A, so the pair maximum comes from the A->B direction.
    numres = 2 * nres
    pae = np.full((numres, numres), 3.0)
    pae[:nres, nres:] = 8.0   # B aligned -> A scored (worse)
    pae[nres:, :nres] = 2.0   # A aligned -> B scored (better)
    data = {
        "plddt": [90.0] * numres,
        "pae": pae.tolist(),
        "iptm": 0.5,
        "ptm": 0.5,
    }
    json_path = tmp_path / "scores.json"
    json_path.write_text(json.dumps(data))

    result = ipsae.score_interactions(str(json_path), str(pdb_path), 10, 10)

    assert list(result.structure.unique_chains) == ["B", "A"]

    scores = result.scores
    for max_dict, asym_dict in (
        (scores.ipsae_d0res_max, scores.ipsae_d0res_asym),
        (scores.ipsae_d0chn_max, scores.ipsae_d0chn_asym),
        (scores.ipsae_d0dom_max, scores.ipsae_d0dom_asym),
        (scores.iptm_d0chn_max, scores.iptm_d0chn_asym),
    ):
        expected_max = max(asym_dict["A"]["B"], asym_dict["B"]["A"])
        assert max_dict["A"]["B"] == expected_max
        assert max_dict["B"]["A"] == expected_max

    # The better direction is A->B, which the original script dropped
    assert scores.ipsae_d0res_asym["A"]["B"] > scores.ipsae_d0res_asym["B"]["A"]
    assert result.get_score("A", "B", "ipSAE", "max") == scores.ipsae_d0res_asym["A"]["B"]


def make_boltz_inputs(tmp_path, confidence_payload):
    nres = 8
    pdb_path = tmp_path / "model_0.pdb"
    write_gly_pdb(pdb_path, [("A", nres), ("B", nres)])

    numres = 2 * nres
    pae = np.full((numres, numres), 4.0)
    np.savez(tmp_path / "pae_model_0.npz", pae=pae)
    np.savez(tmp_path / "plddt_model_0.npz", plddt=np.full(numres, 0.9))  # normalized
    (tmp_path / "confidence_model_0.json").write_text(json.dumps(confidence_payload))
    return tmp_path / "pae_model_0.npz", pdb_path


def test_boltz_missing_pair_chains_iptm_does_not_crash(tmp_path, capsys):
    """The original script crashed with a KeyError when the Boltz confidence
    file had no 'pair_chains_iptm' key; it must fall back to ipTM = 0."""
    pae_path, pdb_path = make_boltz_inputs(tmp_path, {"confidence_score": 0.8})
    result = ipsae.score_interactions(str(pae_path), str(pdb_path), 10, 10)
    assert "pair_chains_iptm" in capsys.readouterr().out
    assert result.get_score("A", "B", "ipTM_af", "asym") == 0
    # normalized Boltz pLDDT values are rescaled to 0-100
    assert result.confidence.plddt == pytest.approx(np.full(16, 90.0))


def test_boltz_pair_chains_iptm_used(tmp_path):
    payload = {"pair_chains_iptm": {"0": {"0": 0.0, "1": 0.77}, "1": {"0": 0.66, "1": 0.0}}}
    pae_path, pdb_path = make_boltz_inputs(tmp_path, payload)
    result = ipsae.score_interactions(str(pae_path), str(pdb_path), 10, 10)
    assert result.get_score("A", "B", "ipTM_af", "asym") == pytest.approx(0.77)
    assert result.get_score("B", "A", "ipTM_af", "asym") == pytest.approx(0.66)
    # the "max" row reports the larger of the two Boltz ipTM values
    assert result.get_score("A", "B", "ipTM_af", "max") == pytest.approx(0.77)


def test_esmfold2_cif_json_scoring_and_detection(tmp_path):
    nres = 8
    cif_path = tmp_path / "esmfold2_model.cif"
    write_gly_cif(cif_path, [("A", nres), ("B", nres)])

    numres = 2 * nres
    pae = np.full((numres, numres), 4.0)
    pae[nres:, :nres] = 2.0
    pae[:nres, nres:] = 7.0
    json_path = tmp_path / "esmfold2_complex_pae.json"
    json_path.write_text(json.dumps({
        "predicted_aligned_error": pae.tolist(),
        "plddt": [0.9] * numres,
    }))

    assert detect_model_type(str(json_path), str(cif_path)) == ("esmfold2", "cif")
    result = ipsae.score_interactions(str(json_path), str(cif_path), 10, 10)

    assert result.model_type == "esmfold2"
    assert result.get_score("A", "B", "ipTM_af", "asym") == 0
    assert result.get_score("B", "A", "ipSAE", "asym") > result.get_score("A", "B", "ipSAE", "asym")
    assert result.confidence.plddt == pytest.approx(np.full(numres, 90.0))


def test_esmfold2_raw_matrix_json_supported(tmp_path):
    nres = 6
    cif_path = tmp_path / "model.cif"
    write_gly_cif(cif_path, [("A", nres), ("B", nres)])

    numres = 2 * nres
    pae = np.full((numres, numres), 3.0)
    pae[nres:, :nres] = 1.5
    json_path = tmp_path / "complex_pae.json"
    json_path.write_text(json.dumps(pae.tolist()))

    assert detect_model_type(str(json_path), str(cif_path)) == ("esmfold2", "cif")
    result = ipsae.score_interactions(str(json_path), str(cif_path), 10, 10)
    assert result.model_type == "esmfold2"
    assert result.confidence.pae_matrix.shape == (numres, numres)
    assert result.confidence.plddt == pytest.approx(np.zeros(numres))


def test_esmfold2_raw_matrix_json_gz_supported(tmp_path):
    nres = 5
    cif_path = tmp_path / "model.cif"
    write_gly_cif(cif_path, [("A", nres), ("B", nres)])

    numres = 2 * nres
    pae = np.full((numres, numres), 4.0)
    pae[nres:, :nres] = 2.0
    pae[:nres, nres:] = 7.0
    json_gz_path = tmp_path / "complex_pae.json.gz"
    with gzip.open(json_gz_path, "wt") as handle:
        json.dump(pae.tolist(), handle)

    assert detect_model_type(str(json_gz_path), str(cif_path)) == ("esmfold2", "cif")
    result = ipsae.score_interactions(str(json_gz_path), str(cif_path), 10, 10)
    assert result.model_type == "esmfold2"
    assert result.confidence.pae_matrix.shape == (numres, numres)
    assert result.confidence.plddt == pytest.approx(np.zeros(numres))
    assert result.get_score("A", "B", "ipTM_af", "asym") == 0
    assert result.get_score("B", "A", "ipSAE", "asym") > result.get_score("A", "B", "ipSAE", "asym")


def test_cli_model_flag_esmfold2(tmp_path, monkeypatch):
    nres = 6
    cif_path = tmp_path / "model.cif"
    write_gly_cif(cif_path, [("A", nres), ("B", nres)])
    pae = np.full((2 * nres, 2 * nres), 3.0)
    pae_path = tmp_path / "raw_pae.json"
    pae_path.write_text(json.dumps(pae.tolist()))

    monkeypatch.chdir(tmp_path)
    exit_code = cli_main([pae_path.name, cif_path.name, "10", "10", "--model", "esmfold2"])
    assert exit_code == 0
    assert (tmp_path / "model_10_10.txt").exists()


def test_boltz2_model_alias_in_api(tmp_path):
    payload = {"pair_chains_iptm": {"0": {"0": 0.0, "1": 0.77}, "1": {"0": 0.66, "1": 0.0}}}
    pae_path, pdb_path = make_boltz_inputs(tmp_path, payload)
    result = ipsae.score_interactions(str(pae_path), str(pdb_path), 10, 10, model_type="boltz2")
    assert result.model_type == "boltz"
    assert result.get_score("A", "B", "ipTM_af", "asym") == pytest.approx(0.77)


def test_esmfold2_invalid_top_level_json_type_raises(tmp_path):
    nres = 4
    cif_path = tmp_path / "model.cif"
    write_gly_cif(cif_path, [("A", nres), ("B", nres)])
    pae_path = tmp_path / "bad.json"
    pae_path.write_text(json.dumps("not-a-matrix"))
    with pytest.raises(ValueError):
        ipsae.score_interactions(str(pae_path), str(cif_path), 10, 10, model_type="esmfold2")


def test_normalize_model_type():
    assert normalize_model_type("af2") == "af2"
    assert normalize_model_type("alphafold2") == "af2"
    assert normalize_model_type("af3") == "af3"
    assert normalize_model_type("alphafold3") == "af3"
    assert normalize_model_type("boltz") == "boltz"
    assert normalize_model_type("boltz1") == "boltz"
    assert normalize_model_type("boltz2") == "boltz"
    assert normalize_model_type("esmfold") == "esmfold2"
    assert normalize_model_type("esmfold2") == "esmfold2"
    with pytest.raises(ValueError, match="Unknown model type"):
        normalize_model_type("unknown_model")


def test_detect_file_format():
    assert detect_file_format("model.pdb") == "pdb"
    assert detect_file_format("MODEL.PDB") == "pdb"
    assert detect_file_format("model.cif") == "cif"
    assert detect_file_format("MODEL.CIF") == "cif"
    with pytest.raises(ValueError, match="Wrong structure file type"):
        detect_file_format("model.txt")


def test_api_model_keyword_argument(tmp_path):
    nres = 6
    cif_path = tmp_path / "model.cif"
    write_gly_cif(cif_path, [("A", nres), ("B", nres)])
    pae = np.full((2 * nres, 2 * nres), 3.0)
    pae_path = tmp_path / "raw_pae.json"
    pae_path.write_text(json.dumps(pae.tolist()))

    result = ipsae.score_interactions(str(pae_path), str(cif_path), 10, 10, model="esmfold2")
    assert result.model_type == "esmfold2"
    assert result.model == "esmfold2"

    # Conflicting model and model_type raises
    with pytest.raises(ValueError, match="Conflicting model arguments"):
        ipsae.score_interactions(str(pae_path), str(cif_path), 10, 10, model="esmfold2", model_type="af3")


def test_esmfold2_hemoglobin_example_score():
    esm_dir = EXAMPLES_DIR / "hemoglobin_esmfold2"
    pae_file = esm_dir / "fold-pae.json"
    cif_file = esm_dir / "structure-1.cif"
    expected_txt = esm_dir / "structure-1_10_15.txt"

    result = ipsae.score_interactions(str(pae_file), str(cif_file), 10, 15, model="esmfold2")
    assert result.model == "esmfold2"
    assert result.model_type == "esmfold2"

    # Compare key scores against example file
    expected_lines = expected_txt.read_text().splitlines()
    assert len(expected_lines) > 2


def test_boltz2_example_score():
    boltz_dir = EXAMPLES_DIR / "BDBV_scFv_peleke_boltz2"
    pae_file = boltz_dir / "pae.npz"
    cif_file = boltz_dir / "predicted_structure.cif"

    result = ipsae.score_interactions(str(pae_file), str(cif_file), 10, 10, model="boltz2")
    assert result.model == "boltz"
    assert result.model_type == "boltz"

