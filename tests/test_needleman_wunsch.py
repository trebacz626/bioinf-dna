import pytest
import numpy as np
from pathlib import Path
from dna.needleman_wunsch import make_dp_matrix_needlman_wunsch, needlman_wunsch
from dna.utils import read_substitution_matrix_from_csv


@pytest.fixture
def setup_data_wiki():
    s1 = "GCATGCG"
    s2 = "GATTACA"
    gap_penalty = -1
    substitution_matrix = {
        "A": {"A": 1, "C": -1, "G": -1, "T": -1},
        "C": {"A": -1, "C": 1, "G": -1, "T": -1},
        "G": {"A": -1, "C": -1, "G": 1, "T": -1},
        "T": {"A": -1, "C": -1, "G": -1, "T": 1},
    }
    return s1, s2, gap_penalty, substitution_matrix


def test_make_dp_matrix_needlman_wunsch_wiki(setup_data_wiki):
    s1, s2, gap_penalty, substitution_matrix = setup_data_wiki
    expected_matrix = [
        [0, -1, -2, -3, -4, -5, -6, -7],
        [-1, 1, 0, -1, -2, -3, -4, -5],
        [-2, 0, 0, 1, 0, -1, -2, -3],
        [-3, -1, -1, 0, 2, 1, 0, -1],
        [-4, -2, -2, -1, 1, 1, 0, -1],
        [-5, -3, -3, -1, 0, 0, 0, -1],
        [-6, -4, -2, -2, -1, -1, 1, 0],
        [-7, -5, -3, -1, -2, -2, 0, 0],
    ]

    obtained_matrix = make_dp_matrix_needlman_wunsch(
        s1, s2, substitution_matrix, gap_penalty
    )

    assert np.array_equal(obtained_matrix, expected_matrix)


def test_needlman_wunsch_wiki(setup_data_wiki):
    s1, s2, gap_penalty, substitution_matrix = setup_data_wiki
    expected_alignments = set(
        [("GCA-TGCG", "G-ATTACA"), ("GCAT-GCG", "G-ATTACA"), ("GCATG-CG", "G-ATTACA")]
    )
    results = needlman_wunsch(s1, s2, 10, substitution_matrix, gap_penalty)
    results = [result[:2] for result in results]
    obtained_alignments = set(results)
    assert obtained_alignments == expected_alignments


@pytest.fixture
def setup_data_assigment():
    s1 = "TATA"
    s2 = "ATAT"
    gap_penalty = -1
    substitution_matrix = read_substitution_matrix_from_csv(
        Path("data/substitution_matrix_assignment.csv")
    )
    return s1, s2, gap_penalty, substitution_matrix


def test_make_dp_matrix_needlman_wunsch_assignment(setup_data_assigment):
    s1, s2, gap_penalty, substitution_matrix = setup_data_assigment
    expected_matrix = [
        [
            0,
            -1,
            -2,
            -3,
            -4,
        ],
        [
            -1,
            -1,
            4,
            3,
            2,
        ],
        [
            -2,
            4,
            3,
            9,
            8,
        ],
        [
            -3,
            3,
            9,
            8,
            14,
        ],
        [
            -4,
            2,
            8,
            14,
            13,
        ],
    ]

    obtained_matrix = make_dp_matrix_needlman_wunsch(
        s1, s2, substitution_matrix, gap_penalty
    )
    print(obtained_matrix)

    assert np.array_equal(obtained_matrix, expected_matrix)


def test_needlman_wunsch_assignment(setup_data_assigment):
    s1, s2, gap_penalty, substitution_matrix = setup_data_assigment
    expected_alignments = set([("-TATA", "ATAT-"), ("TATA-", "-ATAT")])
    results = needlman_wunsch(s1, s2, 10, substitution_matrix, gap_penalty)
    results = [result[:2] for result in results]
    obtained_alignments = set(results)
    assert obtained_alignments == expected_alignments
