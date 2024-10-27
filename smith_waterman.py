from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from utils import read_substitution_matrix_from_csv


def make_dp_matrix_smith_waterman(s1: str, s2: str, substitution_matrix: dict[str,dict[str,int]], gap_penalty: int) -> np.array:
    results = np.zeros((len(s2)+1, len(s1)+1))
    
    results[1:, 0] = 0
    results[0, 1:] = 0
        
    for i in range(1, len(s2)+1):
        for j in range(1, len(s1)+1):
            best_candidate_score = results[i-1][j-1] + substitution_matrix[s2[i-1]][s1[j-1]]
            #'↖'
            
            if results[i-1][j] + gap_penalty > best_candidate_score:
                best_candidate_score = results[i-1][j] + gap_penalty
                #'↑'
                
            if results[i][j-1] + gap_penalty > best_candidate_score:
                best_candidate_score = results[i][j-1] + gap_penalty
                #'←'
                            
            results[i][j] = max(best_candidate_score, 0)
    return results


def smith_waterman(s1:str, s2: str, n: int, substitution_matrix: dict, gap_penalty: int):
    dp_matrix = make_dp_matrix_smith_waterman(s1, s2, substitution_matrix, gap_penalty)

    results: list[tuple[str]] = []
    
    start_positions = np.where(dp_matrix == dp_matrix.max())
    
    def dfs(i: int, j: int, new_s1: str, new_s2: str):
        if len(results) == n:
            return
        if dp_matrix[i][j] == 0:
            results.append((new_s1, new_s2, dp_matrix[-1][-1]))
            return
        if i > 0 and j > 0:
            if dp_matrix[i][j] == dp_matrix[i-1][j-1] + substitution_matrix[s2[i-1]][s1[j-1]]:
                dfs(i-1, j-1, s1[j-1]+new_s1, s2[i-1]+new_s2)
        if i > 0:
            if dp_matrix[i][j] == dp_matrix[i-1][j] + gap_penalty:
                dfs(i-1, j, "-"+new_s1, s2[i-1]+new_s2)
        if j > 0:
            if dp_matrix[i][j] == dp_matrix[i][j-1] + gap_penalty:
                dfs(i, j-1, s1[j-1]+new_s1, "-"+new_s2)
    
    for start_i, start_j in zip(start_positions[0], start_positions[1]):
        dfs(start_i, start_j, "", "")
    
    return results
        
                

            
def main(s1: str, s2: str, n: int, substitution_matrix_path: Path, gap_penalty: int, output_filename: Path):
    substitution_matrix = read_substitution_matrix_from_csv(substitution_matrix_path)
    results = smith_waterman(s1, s2, n, substitution_matrix, gap_penalty)

    with output_filename.open('w') as f:
        for i, (new_s1, new_s2, score) in enumerate(results):
            f.write(f"Global alignment no. {i+1}:\n")
            f.write(new_s1 + "\n")
            f.write(new_s2 + "\n")
            f.write(f"Score: {score}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Needleman-Wunsch algorithm for sequence alignment")
    parser.add_argument("--s1", type=str, help="First sequence")
    parser.add_argument("--s2", type=str, help="Second sequence")
    parser.add_argument("--n", type=int, help="Number of alignments to find", default=10)
    parser.add_argument("--substitution_matrix_path", type=Path, help="Path to the substitution matrix CSV file",default="substitution_matrix_assignment.csv")
    parser.add_argument("--gap_penalty", type=int, help="Gap penalty", default=-2)
    parser.add_argument("--output_filename", type=Path, help="Output filename", default="output.txt")

    args = parser.parse_args()

    main(args.s1, args.s2, args.n, Path(args.substitution_matrix_path), args.gap_penalty, Path(args.output_filename))
    
    