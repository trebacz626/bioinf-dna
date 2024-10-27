import pytest
import numpy as np
from pathlib import Path
from dna.smith_waterman import make_dp_matrix_smith_waterman, smith_waterman
from dna.utils import read_substitution_matrix_from_csv


@pytest.fixture
def setup_data_wiki():
    s1 = "TGTTACGG"
    s2 = "GGTTGACTA"
    gap_penalty = -2
    substitution_matrix = {
        "A": {"A": 3, "C": -3, "G": -3, "T": -3},
        "C": {"A": -3, "C": 3, "G": -3, "T": -3},
        "G": {"A": -3, "C": -3, "G": 3, "T": -3},
        "T": {"A": -3, "C": -3, "G": -3, "T": 3},
    }
    return s1, s2, gap_penalty, substitution_matrix


def test_make_dp_matrix_smith_waterman_wiki(setup_data_wiki):
    s1, s2, gap_penalty, substitution_matrix = setup_data_wiki
    expected_matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 1, 0, 0, 0, 3, 3],
        [0, 0, 3, 1, 0, 0, 0, 3, 6],
        [0, 3, 1, 6, 4, 2, 0, 1, 4],
        [0, 3, 1, 4, 9, 7, 5, 3, 2],
        [0, 1, 6, 4, 7, 6, 4, 8, 6],
        [0, 0, 4, 3, 5, 10, 8, 6, 5],
        [0, 0, 2, 1, 3, 8, 13, 11, 9],
        [0, 3, 1, 5, 4, 6, 11, 10, 8],
        [0, 1, 0, 3, 2, 7, 9, 8, 7],
    ]

    obtained_matrix = make_dp_matrix_smith_waterman(
        s1, s2, substitution_matrix, gap_penalty
    )

    assert np.array_equal(obtained_matrix, expected_matrix)


def test_smith_waterman_wiki(setup_data_wiki):
    s1, s2, gap_penalty, substitution_matrix = setup_data_wiki
    expected_alignments = set([("GTT-AC", "GTTGAC")])
    results = smith_waterman(s1, s2, 10, substitution_matrix, gap_penalty)
    results = [result[:2] for result in results]
    obtained_alignments = set(results)
    print(obtained_alignments)
    assert obtained_alignments == expected_alignments


@pytest.fixture
def setup_data_assigment():
    # from https://www.biogem.org/downloads/notes/kau/Local%20Alignment.pdf
    s1 = "GAATTCAGTTA"
    s2 = "GGATCGA"
    gap_penalty = -4
    substitution_matrix = {
        "A": {"A": 5, "C": -3, "G": -3, "T": -3},
        "C": {"A": -3, "C": 5, "G": -3, "T": -3},
        "G": {"A": -3, "C": -3, "G": 5, "T": -3},
        "T": {"A": -3, "C": -3, "G": -3, "T": 5},
    }
    return s1, s2, gap_penalty, substitution_matrix


def test_make_dp_matrix_smith_waterman_assignment(setup_data_assigment):
    s1, s2, gap_penalty, substitution_matrix = setup_data_assigment
    expected_matrix = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 1, 0, 0, 0, 0, 0, 5, 1, 0, 0],
            [0, 5, 2, 0, 0, 0, 0, 0, 5, 2, 0, 0],
            [0, 1, 10, 7, 3, 0, 0, 5, 1, 2, 0, 5],
            [0, 0, 6, 7, 12, 8, 4, 1, 2, 6, 7, 3],
            [0, 0, 2, 3, 8, 9, 13, 9, 5, 2, 3, 4],
            [0, 5, 1, 0, 4, 5, 9, 10, 14, 10, 6, 2],
            [0, 1, 10, 6, 2, 1, 5, 14, 10, 11, 7, 11],
        ]
    )

    obtained_matrix = make_dp_matrix_smith_waterman(
        s1, s2, substitution_matrix, gap_penalty
    )
    print(obtained_matrix == expected_matrix)
    print(obtained_matrix)

    assert np.array_equal(obtained_matrix, expected_matrix)


def test_smith_waterman_assignment(setup_data_assigment):
    s1, s2, gap_penalty, substitution_matrix = setup_data_assigment
    expected_alignments = set(
        [
            ("GAATTC-A", "GGA-TCGA"),
            ("GAATTC-A", "GGAT-CGA"),
            ("GAATTCAG", "GGA-TC-G"),
            ("GAATTCAG", "GGAT-C-G"),
        ]
    )
    results = smith_waterman(s1, s2, 10, substitution_matrix, gap_penalty)
    results = [result[:2] for result in results]
    obtained_alignments = set(results)
    assert obtained_alignments == expected_alignments
