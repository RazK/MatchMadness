from typing import Tuple

import numpy as np

from pieces import SYMBOLS
from solver import BOARD_SIZE


def canonicalize(board: np.array) -> Tuple[np.array, np.int64]:
    board_rotations = [np.rot90(board, i) for i in range(4)]
    rotations_scores = [score(rotation) for rotation in board_rotations]
    min_rotation = np.argmin(rotations_scores)
    canonical = board_rotations[min_rotation]
    canonical_score = rotations_scores[min_rotation]
    return canonical, canonical_score


def heat_at(i, j):
    return (np.maximum(i, j)) ** 2 + np.maximum(i, j) - i + j


def score(board):
    return int(np.einsum('ij,ij', board, HEATMAP))


BASE = len(SYMBOLS)
HEATMAP = BASE ** np.fromfunction(function=heat_at, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int64)
