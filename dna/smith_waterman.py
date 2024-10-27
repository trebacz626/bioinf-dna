from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from dna.utils import (
    read_substitution_matrix_from_csv,
    construct_alignments,
    write_results_to_file,
)


def make_dp_matrix_smith_waterman(
    s1: str, s2: str, substitution_matrix: dict[str, dict[str, int]], gap_penalty: int
) -> np.array:
    results = np.zeros((len(s2) + 1, len(s1) + 1))

    results[1:, 0] = 0
    results[0, 1:] = 0

    for i in range(1, len(s2) + 1):
        for j in range(1, len(s1) + 1):
            best_candidate_score = (
                results[i - 1][j - 1] + substitution_matrix[s2[i - 1]][s1[j - 1]]
            )
            #'↖'

            if results[i - 1][j] + gap_penalty > best_candidate_score:
                best_candidate_score = results[i - 1][j] + gap_penalty
                #'↑'

            if results[i][j - 1] + gap_penalty > best_candidate_score:
                best_candidate_score = results[i][j - 1] + gap_penalty
                #'←'

            results[i][j] = max(best_candidate_score, 0)
    return results


def smith_waterman(
    s1: str, s2: str, n: int, substitution_matrix: dict, gap_penalty: int
):
    dp_matrix = make_dp_matrix_smith_waterman(s1, s2, substitution_matrix, gap_penalty)

    results: list[tuple[str]] = []

    start_positions = np.where(dp_matrix == dp_matrix.max())

    def end_condition(i: int, j: int, dp_matrix: np.array):
        return dp_matrix[i][j] == 0

    results = construct_alignments(
        s1,
        s2,
        n,
        dp_matrix,
        substitution_matrix,
        gap_penalty,
        start_positions,
        end_condition,
    )
    return results


def main(
    s1: str,
    s2: str,
    n: int,
    substitution_matrix_path: Path,
    gap_penalty: int,
    output_filename: Path,
):
    substitution_matrix = read_substitution_matrix_from_csv(substitution_matrix_path)
    results = smith_waterman(s1, s2, n, substitution_matrix, gap_penalty)

    write_results_to_file(results, output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Needleman-Wunsch algorithm for sequence alignment"
    )
    parser.add_argument("--s1", type=str, help="First sequence")
    parser.add_argument("--s2", type=str, help="Second sequence")
    parser.add_argument(
        "--n", type=int, help="Number of alignments to find", default=10
    )
    parser.add_argument(
        "--substitution_matrix_path",
        type=Path,
        help="Path to the substitution matrix CSV file",
        default="data/substitution_matrix_assignment.csv",
    )
    parser.add_argument("--gap_penalty", type=int, help="Gap penalty", default=-2)
    parser.add_argument(
        "--output_filename", type=Path, help="Output filename", default="output.txt"
    )

    args = parser.parse_args()

    main(
        args.s1,
        args.s2,
        args.n,
        Path(args.substitution_matrix_path),
        args.gap_penalty,
        Path(args.output_filename),
    )
