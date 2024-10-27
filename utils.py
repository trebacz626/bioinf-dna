from pathlib import Path
import numpy as np
import pandas as pd


def read_substitution_matrix_from_csv(file_path: Path) -> dict:
    df = pd.read_csv(file_path, index_col=0)
    df = df.astype(int)
    return df.to_dict()



def construct_alignments(s1: str, s2: str, n: int, dp_matrix: np.array, substitution_matrix: dict, gap_penalty: int, start_positions: list[tuple[int]], end_condition: callable) -> list[tuple[str]]:
    results: list[tuple[str]] = []    
    def dfs(i: int, j: int, new_s1: str, new_s2: str, score: int):
        assert len(new_s1) == len(new_s2)
        if len(results) == n:
            return
        if end_condition(i, j, dp_matrix):
            results.append((new_s1, new_s2, score))
            return
        if i > 0 and j > 0:
            if dp_matrix[i][j] == dp_matrix[i-1][j-1] + substitution_matrix[s2[i-1]][s1[j-1]]:
                dfs(i-1, j-1, s1[j-1]+new_s1, s2[i-1]+new_s2, score)
        if i > 0:
            if dp_matrix[i][j] == dp_matrix[i-1][j] + gap_penalty:
                dfs(i-1, j, "-"+new_s1, s2[i-1]+new_s2, score)
        if j > 0:
            if dp_matrix[i][j] == dp_matrix[i][j-1] + gap_penalty:
                dfs(i, j-1, s1[j-1]+new_s1, "-"+new_s2, score)
    
    for start_i, start_j in zip(start_positions[0], start_positions[1]):
        dfs(start_i, start_j, "", "", dp_matrix[start_i][start_j])
    
    return results
