import numpy as np
from typing import List, Tuple, Set
from copy import copy
from enum import Enum
from colorama import init

init()  # colorama init

from pieces import Piece, Face, PiecesMatrix, ALL_PIECES, SYMBOLS, COLORED

BOARD_SIZE = 4


class Move:
    class Orientation(Enum):
        HORIZONTAL = 0
        VERTICAL = 1

    def __init__(self, piece: Piece, orientation: Orientation, origin: Tuple):
        self.piece = piece
        self.orientation = orientation
        self.origin = origin


class BoardState:
    def __init__(self, board: np.array):
        self.canonical, self.canonical_score = BoardState.canonicalize(board)

    @property
    def is_empty(self) -> bool:
        return not np.any(self.canonical)

    @staticmethod
    def canonicalize(board: np.array) -> Tuple[np.array, np.int64]:
        board_rotations = [np.rot90(board, i) for i in range(4)]
        rotations_scores = [BoardState.__score(rotation) for rotation in board_rotations]
        min_rotation = np.argmin(rotations_scores)
        canonical = board_rotations[min_rotation]
        canonical_score = rotations_scores[min_rotation]
        return canonical, canonical_score

    @staticmethod
    def __heat_at(i, j):
        return (np.maximum(i, j)) ** 2 + np.maximum(i, j) - i + j

    @staticmethod
    def __score(board):
        return int(np.einsum('ij,ij', board, BoardState.__heatmap))

    def __hash__(self):
        return int(self.canonical_score)

    def __repr__(self):
        lines = ""
        for row in self.canonical:
            line = "[" + " ".join([COLORED[SYMBOLS[i]] for i in row]) + "]"
            lines += line + "\n"
        return lines

    def __getitem__(self, row):
        return self.canonical[row]

    def __eq__(self, other):
        return hash(self) == hash(other)

    __base = len(SYMBOLS)
    __heatmap = __base ** np.fromfunction(function=BoardState.__heat_at, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int64)


class GameState:
    def __init__(self, boardState: BoardState, pieces: Set[Piece]):
        self.boardState = boardState
        self.pieces = pieces
        self.piecesMatrix = PiecesMatrix(pieces)

    @property
    def is_goal(self) -> bool:
        return self.boardState.is_empty

    def get_valid_moves(self) -> List[Move]:
        print(b1)
        for iy, ix in np.ndindex((BOARD_SIZE - 1, BOARD_SIZE - 1)):
            horizontal = Face(self.boardState[iy][ix], self.boardState[iy][ix + 1])
            vertical = Face(self.boardState[iy][ix], self.boardState[iy + 1][ix])
            print(horizontal)
            print(vertical)

    def do_move(self, move: Move):
        pass


class MatchMadness:
    def __init__(self):
        pass

    def solve(self, state: GameState) -> List[Move]:
        moves = []
        fringe = [(state, moves)]
        while fringe:
            state, moves = fringe.pop()
            if state.is_goal:
                return moves  # Solved!
            for move in state.get_valid_moves():
                child = state.do_move(move)
                fringe.append((child, moves + move))
        raise Exception("No solution found!")


def test_boardHash():
    print(BoardState.heatmap)
    m = np.array([
        [1, 2, 3, 4],
        [0, 3, 0, 0],
        [0, 4, 0, 0],
        [0, 0, 0, 0]])
    b1 = BoardState(m)
    b2 = BoardState(np.rot90(m))
    print(hash(b1))
    print(b1)
    print(b2)
    print(b1 == b2)


if __name__ == "__main__":
    print("Game On!")
    m = np.array([
        [1, 2, 3, 4],
        [0, 3, 0, 0],
        [0, 4, 0, 0],
        [0, 0, 0, 0]])
    b1 = BoardState(m)
    game = MatchMadness()
    game.solve(GameState(b1, ALL_PIECES))
