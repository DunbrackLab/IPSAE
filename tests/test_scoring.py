"""Integration tests for scoring output against sample files."""

from pathlib import Path
import subprocess
import tempfile
import shutil
import sys

import pytest

# Path to example files
EXAMPLE_DIR = Path(__file__).parent.parent / "Example"


def find_output_files(output_dir: Path, pae_cutoff: int, dist_cutoff: int):
    """Find output files in the output directory matching the cutoff pattern."""
    pattern = f"*_{pae_cutoff}_{dist_cutoff}"
    txt_files = list(output_dir.glob(f"{pattern}.txt"))
    byres_files = list(output_dir.glob(f"{pattern}_byres.txt"))
    pml_files = list(output_dir.glob(f"{pattern}.pml"))
    return {
        "txt": txt_files[0] if txt_files else None,
        "byres": byres_files[0] if byres_files else None,
        "pml": pml_files[0] if pml_files else None,
    }


def normalize_whitespace(s: str) -> str:
    """Normalize whitespace in a string for comparison."""
    return " ".join(s.split())


def run_ipsae_cli(pae_file, structure_file, pae_cutoff, dist_cutoff, output_dir):
    """Run ipsae CLI and return result."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ipsae.ipsae",
            str(pae_file),
            str(structure_file),
            str(pae_cutoff),
            str(dist_cutoff),
            "-o",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    return result


@pytest.mark.skipif(
    not (
        EXAMPLE_DIR / "5b8c_scores_alphafold2_multimer_v3_model_4_seed_000.json"
    ).exists(),
    reason="Example files not available",
)
class TestScoringOutputAF2:
    """Integration tests comparing output against AF2 sample files."""

    @pytest.fixture(scope="class")
    def af2_outputs(self, tmp_path_factory):
        """Run ipsae once and return output files for all tests in this class."""
        # Input files
        pae_file = (
            EXAMPLE_DIR / "5b8c_scores_alphafold2_multimer_v3_model_4_seed_000.json"
        )
        structure_file = (
            EXAMPLE_DIR / "5b8c_unrelaxed_alphafold2_multimer_v3_model_4_seed_000.pdb"
        )
        pae_cutoff = 15
        dist_cutoff = 15

        # Expected files
        expected_txt = (
            EXAMPLE_DIR
            / "5b8c_unrelaxed_alphafold2_multimer_v3_model_4_seed_000_15_15.txt"
        )
        expected_byres = (
            EXAMPLE_DIR
            / "5b8c_unrelaxed_alphafold2_multimer_v3_model_4_seed_000_15_15_byres.txt"
        )
        expected_pml = (
            EXAMPLE_DIR
            / "5b8c_unrelaxed_alphafold2_multimer_v3_model_4_seed_000_15_15.pml"
        )

        # Run ipsae CLI once
        output_dir = tmp_path_factory.mktemp("af2_output")
        result = run_ipsae_cli(
            pae_file, structure_file, pae_cutoff, dist_cutoff, output_dir
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Find output files
        outputs = find_output_files(output_dir, pae_cutoff, dist_cutoff)

        return {
            "output_txt": outputs["txt"],
            "output_byres": outputs["byres"],
            "output_pml": outputs["pml"],
            "expected_txt": expected_txt,
            "expected_byres": expected_byres,
            "expected_pml": expected_pml,
        }

    def test_txt_output_matches_af2(self, af2_outputs):
        """Test that .txt output matches expected sample file line-by-line."""
        output_txt = af2_outputs["output_txt"]
        assert output_txt is not None, "Output .txt file not found"

        with open(output_txt) as f:
            actual_lines = f.readlines()
        with open(af2_outputs["expected_txt"]) as f:
            expected_lines = f.readlines()

        assert len(actual_lines) == len(expected_lines), (
            f"Line count mismatch: {len(actual_lines)} vs {len(expected_lines)}"
        )

        for i, (actual, expected) in enumerate(zip(actual_lines, expected_lines), 1):
            assert actual == expected, (
                f"Line {i} differs:\nActual: {actual!r}\nExpected: {expected!r}"
            )

    def test_byres_output_matches_af2(self, af2_outputs):
        """Test that _byres.txt output matches expected sample file line-by-line."""
        output_byres = af2_outputs["output_byres"]
        assert output_byres is not None, "Output _byres.txt file not found"

        with open(output_byres) as f:
            actual_lines = f.readlines()
        with open(af2_outputs["expected_byres"]) as f:
            expected_lines = f.readlines()

        assert len(actual_lines) == len(expected_lines), (
            f"Line count mismatch: {len(actual_lines)} vs {len(expected_lines)}"
        )

        for i, (actual, expected) in enumerate(zip(actual_lines, expected_lines), 1):
            assert actual == expected, (
                f"Line {i} differs:\nActual: {actual!r}\nExpected: {expected!r}"
            )

    def test_pml_output_content_matches_af2(self, af2_outputs):
        """Test that .pml output has same content (rows may be reordered)."""
        output_pml = af2_outputs["output_pml"]
        assert output_pml is not None, "Output .pml file not found"

        with open(output_pml) as f:
            actual_lines = sorted(
                normalize_whitespace(line) for line in f if line.strip()
            )
        with open(af2_outputs["expected_pml"]) as f:
            expected_lines = sorted(
                normalize_whitespace(line) for line in f if line.strip()
            )

        assert actual_lines == expected_lines, "PML content mismatch (sorted comparison)"


@pytest.mark.skipif(
    not (EXAMPLE_DIR / "fold_5b8c_full_data_0.json").exists(),
    reason="Example files not available",
)
class TestScoringOutputAF3:
    """Integration tests comparing output against AF3 sample files."""

    @pytest.fixture(scope="class")
    def af3_outputs(self, tmp_path_factory):
        """Run ipsae once and return output files for all tests in this class."""
        # Input files
        pae_file = EXAMPLE_DIR / "fold_5b8c_full_data_0.json"
        structure_file = EXAMPLE_DIR / "fold_5b8c_model_0.cif"
        pae_cutoff = 10
        dist_cutoff = 10

        # Expected files
        expected_txt = EXAMPLE_DIR / "fold_5b8c_model_0_10_10.txt"
        expected_byres = EXAMPLE_DIR / "fold_5b8c_model_0_10_10_byres.txt"
        expected_pml = EXAMPLE_DIR / "fold_5b8c_model_0_10_10.pml"

        # Run ipsae CLI once
        output_dir = tmp_path_factory.mktemp("af3_output")
        result = run_ipsae_cli(
            pae_file, structure_file, pae_cutoff, dist_cutoff, output_dir
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Find output files
        outputs = find_output_files(output_dir, pae_cutoff, dist_cutoff)

        return {
            "output_txt": outputs["txt"],
            "output_byres": outputs["byres"],
            "output_pml": outputs["pml"],
            "expected_txt": expected_txt,
            "expected_byres": expected_byres,
            "expected_pml": expected_pml,
        }

    def test_txt_output_matches_af3(self, af3_outputs):
        """Test that .txt output matches expected sample file line-by-line."""
        output_txt = af3_outputs["output_txt"]
        assert output_txt is not None, "Output .txt file not found"

        with open(output_txt) as f:
            actual_lines = f.readlines()
        with open(af3_outputs["expected_txt"]) as f:
            expected_lines = f.readlines()

        assert len(actual_lines) == len(expected_lines), (
            f"Line count mismatch: {len(actual_lines)} vs {len(expected_lines)}"
        )

        for i, (actual, expected) in enumerate(zip(actual_lines, expected_lines), 1):
            assert actual == expected, (
                f"Line {i} differs:\nActual: {actual!r}\nExpected: {expected!r}"
            )

    def test_byres_output_matches_af3(self, af3_outputs):
        """Test that _byres.txt output matches expected sample file line-by-line."""
        output_byres = af3_outputs["output_byres"]
        assert output_byres is not None, "Output _byres.txt file not found"

        with open(output_byres) as f:
            actual_lines = f.readlines()
        with open(af3_outputs["expected_byres"]) as f:
            expected_lines = f.readlines()

        assert len(actual_lines) == len(expected_lines), (
            f"Line count mismatch: {len(actual_lines)} vs {len(expected_lines)}"
        )

        for i, (actual, expected) in enumerate(zip(actual_lines, expected_lines), 1):
            assert actual == expected, (
                f"Line {i} differs:\nActual: {actual!r}\nExpected: {expected!r}"
            )

    def test_pml_output_content_matches_af3(self, af3_outputs):
        """Test that .pml output has same content (rows may be reordered)."""
        output_pml = af3_outputs["output_pml"]
        assert output_pml is not None, "Output .pml file not found"

        with open(output_pml) as f:
            actual_lines = sorted(
                normalize_whitespace(line) for line in f if line.strip()
            )
        with open(af3_outputs["expected_pml"]) as f:
            expected_lines = sorted(
                normalize_whitespace(line) for line in f if line.strip()
            )

        assert actual_lines == expected_lines, "PML content mismatch (sorted comparison)"
