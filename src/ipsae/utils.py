# utils.py
# Shared helpers for the ipsae package.
#
# Derived from ipsae.py by Roland Dunbrack, Fox Chase Cancer Center.
# https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2
# MIT license: script can be modified and redistributed for non-commercial and
# commercial use, as long as this information is reproduced.

import numpy as np


def init_chainpairdict_zeros(chainlist):
    """Initialize a nested chain-pair dictionary with all values set to 0."""
    return {chain1: {chain2: 0 for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}


def init_chainpairdict_npzeros(chainlist, arraysize):
    """Initialize a nested chain-pair dictionary with NumPy arrays of zeros."""
    return {chain1: {chain2: np.zeros(arraysize) for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}


def init_chainpairdict_set(chainlist):
    """Initialize a nested chain-pair dictionary with empty sets."""
    return {chain1: {chain2: set() for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}


def contiguous_ranges(numbers):
    """Format a set of residue numbers as PyMOL-style contiguous ranges (e.g. '1-4+7+9-12')."""
    if not numbers:  # Check if the set is empty
        return None

    sorted_numbers = sorted(numbers)  # Sort the numbers
    start = sorted_numbers[0]
    end = start
    ranges = []  # List to store ranges

    def format_range(start, end):
        if start == end:
            return f"{start}"
        else:
            return f"{start}-{end}"

    for number in sorted_numbers[1:]:
        if number == end + 1:
            end = number
        else:
            ranges.append(format_range(start, end))
            start = end = number

    # Append the last range after the loop
    ranges.append(format_range(start, end))

    # Join all ranges with a plus sign
    return '+'.join(ranges)
