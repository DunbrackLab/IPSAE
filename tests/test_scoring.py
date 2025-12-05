"""Integration tests for scoring output against sample files."""

from pathlib import Path
import subprocess
import tempfile
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


class TestScoringOutputAF2:
    """Integration tests comparing output against AF2 sample files."""

    @pytest.fixture
    def af2_files(self):
        """Return paths to AF2 example files."""
        pae_file = (
            EXAMPLE_DIR / "5b8c_scores_alphafold2_multimer_v3_model_4_seed_000.json"
        )
        structure_file = (
            EXAMPLE_DIR / "5b8c_unrelaxed_alphafold2_multimer_v3_model_4_seed_000.pdb"
        )
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
        return {
            "pae": pae_file,
            "structure": structure_file,
            "expected_txt": expected_txt,
            "expected_byres": expected_byres,
            "expected_pml": expected_pml,
            "pae_cutoff": 15,
            "dist_cutoff": 15,
        }

    @pytest.mark.skipif(
        not (
            EXAMPLE_DIR / "5b8c_scores_alphafold2_multimer_v3_model_4_seed_000.json"
        ).exists(),
        reason="Example files not available",
    )
    def test_txt_output_matches_af2(self, af2_files):
        """Test that .txt output matches expected sample file line-by-line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run ipsae CLI
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ipsae.ipsae",
                    str(af2_files["pae"]),
                    str(af2_files["structure"]),
                    str(af2_files["pae_cutoff"]),
                    str(af2_files["dist_cutoff"]),
                    "-o",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Find output files
            outputs = find_output_files(
                output_dir, af2_files["pae_cutoff"], af2_files["dist_cutoff"]
            )
            output_txt = outputs["txt"]
            assert output_txt is not None, f"Output .txt file not found in {output_dir}"

            with open(output_txt) as f:
                actual_lines = f.readlines()
            with open(af2_files["expected_txt"]) as f:
                expected_lines = f.readlines()

            assert len(actual_lines) == len(expected_lines), (
                f"Line count mismatch: {len(actual_lines)} vs {len(expected_lines)}"
            )

            for i, (actual, expected) in enumerate(
                zip(actual_lines, expected_lines), 1
            ):
                assert actual == expected, f"Line {i} differs:\nActual: {actual!r}\nExpected: {expected!r}"

    @pytest.mark.skipif(
        not (
            EXAMPLE_DIR / "5b8c_scores_alphafold2_multimer_v3_model_4_seed_000.json"
        ).exists(),
        reason="Example files not available",
    )
    def test_byres_output_matches_af2(self, af2_files):
        """Test that _byres.txt output matches expected sample file line-by-line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run ipsae CLI
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ipsae.ipsae",
                    str(af2_files["pae"]),
                    str(af2_files["structure"]),
                    str(af2_files["pae_cutoff"]),
                    str(af2_files["dist_cutoff"]),
                    "-o",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Find output files
            outputs = find_output_files(
                output_dir, af2_files["pae_cutoff"], af2_files["dist_cutoff"]
            )
            output_byres = outputs["byres"]
            assert output_byres is not None, f"Output _byres.txt file not found in {output_dir}"

            with open(output_byres) as f:
                actual_lines = f.readlines()
            with open(af2_files["expected_byres"]) as f:
                expected_lines = f.readlines()

            assert len(actual_lines) == len(expected_lines), (
                f"Line count mismatch: {len(actual_lines)} vs {len(expected_lines)}"
            )

            for i, (actual, expected) in enumerate(
                zip(actual_lines, expected_lines), 1
            ):
                assert actual == expected, f"Line {i} differs:\nActual: {actual!r}\nExpected: {expected!r}"

    @pytest.mark.skipif(
        not (
            EXAMPLE_DIR / "5b8c_scores_alphafold2_multimer_v3_model_4_seed_000.json"
        ).exists(),
        reason="Example files not available",
    )
    def test_pml_output_content_matches_af2(self, af2_files):
        """Test that .pml output has same content (rows may be reordered)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run ipsae CLI
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ipsae.ipsae",
                    str(af2_files["pae"]),
                    str(af2_files["structure"]),
                    str(af2_files["pae_cutoff"]),
                    str(af2_files["dist_cutoff"]),
                    "-o",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Find output files
            outputs = find_output_files(
                output_dir, af2_files["pae_cutoff"], af2_files["dist_cutoff"]
            )
            output_pml = outputs["pml"]
            assert output_pml is not None, f"Output .pml file not found in {output_dir}"

            with open(output_pml) as f:
                actual_lines = sorted(
                    normalize_whitespace(line) for line in f if line.strip()
                )
            with open(af2_files["expected_pml"]) as f:
                expected_lines = sorted(
                    normalize_whitespace(line) for line in f if line.strip()
                )

            assert actual_lines == expected_lines, (
                "PML content mismatch (sorted comparison)"
            )


class TestScoringOutputAF3:
    """Integration tests comparing output against AF3 sample files."""

    @pytest.fixture
    def af3_files(self):
        """Return paths to AF3 example files."""
        pae_file = EXAMPLE_DIR / "fold_5b8c_full_data_0.json"
        structure_file = EXAMPLE_DIR / "fold_5b8c_model_0.cif"
        expected_txt = EXAMPLE_DIR / "fold_5b8c_model_0_10_10.txt"
        expected_byres = EXAMPLE_DIR / "fold_5b8c_model_0_10_10_byres.txt"
        expected_pml = EXAMPLE_DIR / "fold_5b8c_model_0_10_10.pml"
        return {
            "pae": pae_file,
            "structure": structure_file,
            "expected_txt": expected_txt,
            "expected_byres": expected_byres,
            "expected_pml": expected_pml,
            "pae_cutoff": 10,
            "dist_cutoff": 10,
        }

    @pytest.mark.skipif(
        not (EXAMPLE_DIR / "fold_5b8c_full_data_0.json").exists(),
        reason="Example files not available",
    )
    def test_txt_output_matches_af3(self, af3_files):
        """Test that .txt output matches expected sample file line-by-line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run ipsae CLI
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ipsae.ipsae",
                    str(af3_files["pae"]),
                    str(af3_files["structure"]),
                    str(af3_files["pae_cutoff"]),
                    str(af3_files["dist_cutoff"]),
                    "-o",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Find output files
            outputs = find_output_files(
                output_dir, af3_files["pae_cutoff"], af3_files["dist_cutoff"]
            )
            output_txt = outputs["txt"]
            assert output_txt is not None, f"Output .txt file not found in {output_dir}"

            with open(output_txt) as f:
                actual_lines = f.readlines()
            with open(af3_files["expected_txt"]) as f:
                expected_lines = f.readlines()

            assert len(actual_lines) == len(expected_lines), (
                f"Line count mismatch: {len(actual_lines)} vs {len(expected_lines)}"
            )

            for i, (actual, expected) in enumerate(
                zip(actual_lines, expected_lines), 1
            ):
                assert actual == expected, f"Line {i} differs:\nActual: {actual!r}\nExpected: {expected!r}"

    @pytest.mark.skipif(
        not (EXAMPLE_DIR / "fold_5b8c_full_data_0.json").exists(),
        reason="Example files not available",
    )
    def test_byres_output_matches_af3(self, af3_files):
        """Test that _byres.txt output matches expected sample file line-by-line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run ipsae CLI
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ipsae.ipsae",
                    str(af3_files["pae"]),
                    str(af3_files["structure"]),
                    str(af3_files["pae_cutoff"]),
                    str(af3_files["dist_cutoff"]),
                    "-o",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Find output files
            outputs = find_output_files(
                output_dir, af3_files["pae_cutoff"], af3_files["dist_cutoff"]
            )
            output_byres = outputs["byres"]
            assert output_byres is not None, f"Output _byres.txt file not found in {output_dir}"

            with open(output_byres) as f:
                actual_lines = f.readlines()
            with open(af3_files["expected_byres"]) as f:
                expected_lines = f.readlines()

            assert len(actual_lines) == len(expected_lines), (
                f"Line count mismatch: {len(actual_lines)} vs {len(expected_lines)}"
            )

            for i, (actual, expected) in enumerate(
                zip(actual_lines, expected_lines), 1
            ):
                assert actual == expected, f"Line {i} differs:\nActual: {actual!r}\nExpected: {expected!r}"

    @pytest.mark.skipif(
        not (EXAMPLE_DIR / "fold_5b8c_full_data_0.json").exists(),
        reason="Example files not available",
    )
    def test_pml_output_content_matches_af3(self, af3_files):
        """Test that .pml output has same content (rows may be reordered)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run ipsae CLI
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ipsae.ipsae",
                    str(af3_files["pae"]),
                    str(af3_files["structure"]),
                    str(af3_files["pae_cutoff"]),
                    str(af3_files["dist_cutoff"]),
                    "-o",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Find output files
            outputs = find_output_files(
                output_dir, af3_files["pae_cutoff"], af3_files["dist_cutoff"]
            )
            output_pml = outputs["pml"]
            assert output_pml is not None, f"Output .pml file not found in {output_dir}"

            with open(output_pml) as f:
                actual_lines = sorted(
                    normalize_whitespace(line) for line in f if line.strip()
                )
            with open(af3_files["expected_pml"]) as f:
                expected_lines = sorted(
                    normalize_whitespace(line) for line in f if line.strip()
                )

            assert actual_lines == expected_lines, (
                "PML content mismatch (sorted comparison)"
            )
