from pathlib import Path
import numpy as np
import pandas as pd


def read_substitution_matrix_from_csv(file_path: Path) -> dict:
    """
    Reads a substitution matrix from a CSV file and returns it as a dictionary.
    Args:
        file_path (Path): The path to the CSV file containing the substitution matrix.
    Returns:
        dict: A dictionary representation of the substitution matrix with integer values.
    """

    df = pd.read_csv(file_path, index_col=0)
    df = df.astype(int)
    return df.to_dict()


def construct_alignments(
    s1: str,
    s2: str,
    n: int,
    dp_matrix: np.array,
    substitution_matrix: dict,
    gap_penalty: int,
    start_positions: list[tuple[int], tuple[int]],
    end_condition: callable,
) -> list[tuple[str]]:
    """
    Constructs alignments between two sequences based on a dynamic programming matrix.
    Args:
        s1 (str): The first sequence.
        s2 (str): The second sequence.
        n (int): The maximum number of alignments to construct.
        dp_matrix (np.array): The dynamic programming matrix used for alignment scoring.
        substitution_matrix (dict): A dictionary representing the substitution matrix for scoring matches/mismatches.
        gap_penalty (int): The penalty score for introducing gaps in the alignment.
        start_positions (list[tuple[int], tuple[int]]): A list of tuples representing the starting positions in the dp_matrix for alignment construction.
        end_condition (callable): A function that determines when to stop the alignment construction based on the current positions in the dp_matrix.
    Returns:
        list[tuple[str]]: A list of tuples, each containing two aligned sequences and their corresponding alignment score.
    """

    results: list[tuple[str]] = []

    def dfs(i: int, j: int, new_s1: str, new_s2: str, score: int):
        assert len(new_s1) == len(new_s2)
        if len(results) == n:
            return
        if end_condition(i, j, dp_matrix):
            results.append((new_s1, new_s2, score))
            return
        if i > 0 and j > 0:
            if (
                dp_matrix[i][j]
                == dp_matrix[i - 1][j - 1] + substitution_matrix[s2[i - 1]][s1[j - 1]]
            ):
                dfs(i - 1, j - 1, s1[j - 1] + new_s1, s2[i - 1] + new_s2, score)
        if i > 0:
            if dp_matrix[i][j] == dp_matrix[i - 1][j] + gap_penalty:
                dfs(i - 1, j, "-" + new_s1, s2[i - 1] + new_s2, score)
        if j > 0:
            if dp_matrix[i][j] == dp_matrix[i][j - 1] + gap_penalty:
                dfs(i, j - 1, s1[j - 1] + new_s1, "-" + new_s2, score)

    for start_i, start_j in zip(start_positions[0], start_positions[1]):
        dfs(start_i, start_j, "", "", dp_matrix[start_i][start_j])

    return results


def write_results_to_file(results: list[tuple[int]], output_filename: Path):
    """
    Write sequence alignment results to a text file.

    Args:
        results: A list of tuples, where each tuple contains two aligned sequences
                (strings) and their alignment score (integer).
        output_filename: A Path object specifying where to write the output file.

    The function writes each alignment to the file in the following format:
        Global alignment no. X:
        [First aligned sequence]
        [Second aligned sequence]
        Score: [alignment score]
    """

    with output_filename.open("w") as f:
        for i, (new_s1, new_s2, score) in enumerate(results):
            f.write(f"Global alignment no. {i+1}:\n")
            f.write(new_s1 + "\n")
            f.write(new_s2 + "\n")
            f.write(f"Score: {score}\n\n")
