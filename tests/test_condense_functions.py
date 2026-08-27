"""
Regression tests for condense_by_location and condense_by_temporal in utils.py.

These tests document the intentional difference between the two functions:
- condense_by_location: groups ALL occurrences of the same location (order-insensitive)
- condense_by_temporal: groups only CONSECUTIVE occurrences (order-sensitive)

This difference is critical: on non-monotone input (A,B,A) they produce different results.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.utils import condense_by_location, condense_by_temporal

A = (3, 5)
B = (4, 8)


class TestCondenseByLocation:
    def test_all_same_location_merged(self):
        seq = [(A, 'N'), (B, 'E'), (A, 'S')]
        result = condense_by_location(seq)
        result_dict = {loc: turns for loc, turns in result}
        assert set(result_dict[A]) == {'N', 'S'}
        assert result_dict[B] == ['E']

    def test_single_element(self):
        result = condense_by_location([(A, 'N')])
        assert result == [(A, ['N'])]

    def test_empty(self):
        assert condense_by_location([]) == []

    def test_monotone_input(self):
        seq = [(A, 'N'), (A, 'E'), (B, 'S')]
        result = condense_by_location(seq)
        result_dict = {loc: turns for loc, turns in result}
        assert sorted(result_dict[A]) == ['E', 'N']
        assert result_dict[B] == ['S']


class TestCondenseByTemporal:
    def test_consecutive_grouping_only(self):
        seq = [(A, 'N'), (B, 'E'), (A, 'S')]
        result = condense_by_temporal(seq)
        # A appears twice because it's not consecutive
        locs = [loc for loc, _ in result]
        assert locs == [A, B, A]

    def test_consecutive_merged(self):
        seq = [(A, 'N'), (A, 'E'), (B, 'S')]
        result = condense_by_temporal(seq)
        locs = [loc for loc, _ in result]
        assert locs == [A, B]
        turns_A = [turns for loc, turns in result if loc == A][0]
        assert set(turns_A) == {'N', 'E'}

    def test_single_element(self):
        result = condense_by_temporal([(A, 'N')])
        assert result == [(A, ['N'])]

    def test_empty(self):
        assert condense_by_temporal([]) == []


class TestCondensesDiffer:
    def test_non_monotone_input_produces_different_results(self):
        """Documents the deliberate behavioural difference on non-monotone input."""
        seq = [(A, 'N'), (B, 'E'), (A, 'S')]
        by_loc = condense_by_location(seq)
        by_tmp = condense_by_temporal(seq)
        # condense_by_location returns 2 groups; condense_by_temporal returns 3
        assert len(by_loc) == 2
        assert len(by_tmp) == 3
