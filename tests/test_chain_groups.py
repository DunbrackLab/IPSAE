"""Tests for chain group parsing and scoring functionality."""

from pathlib import Path

import numpy as np
import pytest

from ipsae.ipsae import (
    chain_group_name,
    get_chain_group_indices,
    ipsae,
    parse_chain_groups,
)

# Path to example files
EXAMPLE_DIR = Path(__file__).parent.parent / "Example"


class TestParseChainGroups:
    """Tests for parse_chain_groups function."""

    def test_single_pair_single_chains(self):
        """Test parsing a single pair of single chains (includes both directions)."""
        result = parse_chain_groups("A/B")
        # Both directions are generated: A->B and B->A
        assert len(result) == 2
        assert (["A"], ["B"]) in result
        assert (["B"], ["A"]) in result

    def test_single_pair_multi_chain_group(self):
        """Test parsing a pair where one group has multiple chains."""
        result = parse_chain_groups("A/H+L")
        # Both directions are generated
        assert len(result) == 2
        assert (["A"], ["H", "L"]) in result
        assert (["H", "L"], ["A"]) in result

    def test_multiple_pairs(self):
        """Test parsing multiple pairs (each generates both directions)."""
        result = parse_chain_groups("A/H+L,A/H,A/L")
        # 3 pairs x 2 directions = 6 total
        assert len(result) == 6
        assert (["A"], ["H", "L"]) in result
        assert (["H", "L"], ["A"]) in result
        assert (["A"], ["H"]) in result
        assert (["H"], ["A"]) in result
        assert (["A"], ["L"]) in result
        assert (["L"], ["A"]) in result

    def test_both_groups_multi_chain(self):
        """Test parsing where both groups have multiple chains."""
        result = parse_chain_groups("A+B/C+D")
        # Both directions are generated
        assert len(result) == 2
        assert (["A", "B"], ["C", "D"]) in result
        assert (["C", "D"], ["A", "B"]) in result

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        result = parse_chain_groups(" A / B + C , D / E ")
        # 2 pairs x 2 directions = 4
        assert len(result) == 4
        assert (["A"], ["B", "C"]) in result
        assert (["B", "C"], ["A"]) in result
        assert (["D"], ["E"]) in result
        assert (["E"], ["D"]) in result

    def test_empty_string(self):
        """Test that empty string returns empty list."""
        result = parse_chain_groups("")
        assert result == []

    def test_invalid_format_no_slash(self):
        """Test that missing slash raises ValueError."""
        with pytest.raises(ValueError, match="Expected 'group1/group2'"):
            parse_chain_groups("ABC")

    def test_invalid_format_multiple_slashes(self):
        """Test that multiple slashes in a pair raises ValueError."""
        with pytest.raises(ValueError, match="Expected exactly one '/' separator"):
            parse_chain_groups("A/B/C")

    def test_invalid_format_empty_group(self):
        """Test that empty group raises ValueError."""
        with pytest.raises(
            ValueError, match="Both groups must contain at least one chain"
        ):
            parse_chain_groups("A/")

    def test_ellipsis_token(self):
        """Test that ... token adds all individual chain permutations."""
        unique_chains = np.array(["A", "B", "C"])
        result = parse_chain_groups("A/B+C,...", unique_chains)
        # A/B+C generates 2 directions, plus all individual pairs from ...
        # Individual pairs: A->B, A->C, B->A, B->C, C->A, C->B = 6
        # Total should be 8 (2 for A/B+C + 6 for individual)
        assert (["A"], ["B", "C"]) in result
        assert (["B", "C"], ["A"]) in result
        # Check that individual pairs are added
        assert (["A"], ["B"]) in result
        assert (["B"], ["A"]) in result
        assert (["A"], ["C"]) in result
        assert (["C"], ["A"]) in result

    def test_ellipsis_requires_unique_chains(self):
        """Test that ... token requires unique_chains parameter."""
        with pytest.raises(ValueError, match="Cannot use '...' without"):
            parse_chain_groups("A/B,...")

    def test_duplicate_removal(self):
        """Test that duplicate pairs are removed."""
        result = parse_chain_groups("A/B,A/B,B/A")
        # A/B appears twice, generates A->B and B->A
        # B/A generates B->A and A->B
        # But duplicates are removed, so:
        # First A/B generates: A->B, B->A (both added)
        # Second A/B: A->B already seen, B->A already seen (neither added)
        # B/A: B->A already seen, A->B already seen (neither added)
        assert len(result) == 2
        assert (["A"], ["B"]) in result
        assert (["B"], ["A"]) in result


class TestGetChainGroupIndices:
    """Tests for get_chain_group_indices function."""

    def test_single_chain(self):
        """Test getting indices for a single chain."""
        chains = np.array(["A", "A", "B", "B", "C"])
        result = get_chain_group_indices(chains, ["A"])
        np.testing.assert_array_equal(result, np.array([0, 1]))

    def test_multiple_chains(self):
        """Test getting indices for multiple chains as one group."""
        chains = np.array(["A", "A", "B", "B", "C"])
        result = get_chain_group_indices(chains, ["A", "C"])
        np.testing.assert_array_equal(result, np.array([0, 1, 4]))

    def test_chain_not_present(self):
        """Test that non-existent chain returns empty array."""
        chains = np.array(["A", "A", "B", "B"])
        result = get_chain_group_indices(chains, ["X"])
        assert len(result) == 0

    def test_all_chains(self):
        """Test getting indices for all chains."""
        chains = np.array(["A", "B", "C"])
        result = get_chain_group_indices(chains, ["A", "B", "C"])
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))


class TestChainGroupName:
    """Tests for chain_group_name function."""

    def test_single_chain(self):
        """Test name for single chain."""
        assert chain_group_name(["A"]) == "A"

    def test_multiple_chains(self):
        """Test name for multiple chains."""
        assert chain_group_name(["H", "L"]) == "H+L"

    def test_three_chains(self):
        """Test name for three chains."""
        assert chain_group_name(["A", "B", "C"]) == "A+B+C"


class TestChainGroupScoring:
    """Integration tests for chain group scoring."""

    @pytest.mark.skipif(
        not (
            EXAMPLE_DIR / "5b8c_scores_alphafold2_multimer_v3_model_4_seed_000.json"
        ).exists(),
        reason="Example files not available",
    )
    def test_chain_group_scoring_af2(self):
        """Test chain group scoring with AF2 example files."""
        pae_file = (
            EXAMPLE_DIR / "5b8c_scores_alphafold2_multimer_v3_model_4_seed_000.json"
        )
        structure_file = (
            EXAMPLE_DIR / "5b8c_unrelaxed_alphafold2_multimer_v3_model_4_seed_000.pdb"
        )

        # Test with chain groups (pass string, not parsed list)
        results = ipsae(
            pae_file=pae_file,
            structure_file=structure_file,
            pae_cutoff=15.0,
            dist_cutoff=15.0,
            model_type="af2",
            chain_groups="A/B+C,A/B,A/C",
        )

        # Each chain group pair generates both directions + max
        # A/B+C => A->B+C, B+C->A, max for A/B+C
        # A/B => A->B, B->A, max for A/B
        # A/C => A->C, C->A, max for A/C
        # Total = 6 asym + 3 max = 9
        assert len(results.chain_pair_scores) == 9

        # Check chain group names (asym only)
        chain_pair_names = [
            (r.Chn1, r.Chn2) for r in results.chain_pair_scores if r.Type == "asym"
        ]
        assert ("A", "B+C") in chain_pair_names
        assert ("B+C", "A") in chain_pair_names
        assert ("A", "B") in chain_pair_names
        assert ("B", "A") in chain_pair_names
        assert ("A", "C") in chain_pair_names
        assert ("C", "A") in chain_pair_names

        # Check that scores are reasonable (between 0 and 1)
        for result in results.chain_pair_scores:
            assert 0 <= result.ipSAE <= 1
            assert 0 <= result.ipSAE_d0chn <= 1
            assert 0 <= result.pDockQ <= 1
            assert 0 <= result.LIS <= 1

    @pytest.mark.skipif(
        not (
            EXAMPLE_DIR / "5b8c_scores_alphafold2_multimer_v3_model_4_seed_000.json"
        ).exists(),
        reason="Example files not available",
    )
    def test_without_chain_groups_default_behavior(self):
        """Test that default behavior (no chain groups) still works."""
        pae_file = (
            EXAMPLE_DIR / "5b8c_scores_alphafold2_multimer_v3_model_4_seed_000.json"
        )
        structure_file = (
            EXAMPLE_DIR / "5b8c_unrelaxed_alphafold2_multimer_v3_model_4_seed_000.pdb"
        )

        results = ipsae(
            pae_file=pae_file,
            structure_file=structure_file,
            pae_cutoff=15.0,
            dist_cutoff=15.0,
            model_type="af2",
            chain_groups=None,  # Explicitly None
        )

        # Default behavior: 3 chain pairs (A-B, A-C, B-C) with asym and max = 9 results
        # asym: A->B, B->A, A->C, C->A, B->C, C->B = 6
        # max: A-B, A-C, B-C = 3
        # Total = 9
        assert len(results.chain_pair_scores) == 9

    @pytest.mark.skipif(
        not (
            EXAMPLE_DIR / "5b8c_scores_alphafold2_multimer_v3_model_4_seed_000.json"
        ).exists(),
        reason="Example files not available",
    )
    def test_chain_group_combined_residues(self):
        """Test that combined chain groups have correct residue counts."""
        pae_file = (
            EXAMPLE_DIR / "5b8c_scores_alphafold2_multimer_v3_model_4_seed_000.json"
        )
        structure_file = (
            EXAMPLE_DIR / "5b8c_unrelaxed_alphafold2_multimer_v3_model_4_seed_000.pdb"
        )

        # Pass string, not parsed list
        results = ipsae(
            pae_file=pae_file,
            structure_file=structure_file,
            pae_cutoff=15.0,
            dist_cutoff=15.0,
            model_type="af2",
            chain_groups="A/B+C",
        )

        # A/B+C generates 3 results: A->B+C, B+C->A, max
        assert len(results.chain_pair_scores) == 3

        # Find the A->B+C result
        a_to_bc = [
            r for r in results.chain_pair_scores if r.Chn1 == "A" and r.Chn2 == "B+C"
        ][0]
        # Chain A has 139 residues, B has 120, C has 111
        # Combined B+C should be 231 residues
        # n0chn should be 139 + 231 = 370
        assert a_to_bc.n0chn == 370
