from typing import List, Tuple, Set
from itertools import combinations_with_replacement
from colorama import init, Fore, Back, Style

SYMBOLS = [' ', '*', '+', 'x', 'o', '#']
COLORED = {
    ' ': ' ' + Style.RESET_ALL,
    '*': Fore.GREEN + '*' + Style.RESET_ALL,
    '+': Fore.CYAN + '+' + Style.RESET_ALL,
    'x': Fore.MAGENTA + 'x' + Style.RESET_ALL,
    'o': Fore.RED + 'o' + Style.RESET_ALL,
    '#': Fore.YELLOW + '#' + Style.RESET_ALL
}
init()  # colorama init


class Symbol:
    def __init__(self, id):
        self.id = id

    @property
    def symbol(self):
        return SYMBOLS[self.id]

    def __repr__(self):
        return COLORED[self.symbol]


class Face:
    def __init__(self, a: Symbol, b: Symbol):
        self.face = sorted([a, b])

    def __getitem__(self, idx):
        return self.face[idx]

    def __repr__(self):
        return f"{repr(self.face[0])} {repr(self.face[1])}"


class Piece:
    def __init__(self, id: int, faces: List[Face]):
        self.id = id
        self.faces = sorted([sorted(face) for face in faces])

    def __repr__(self):
        return f"Piece {self.id}\n" + "\n".join([f"[{SYMBOLS[face[0]]} : {SYMBOLS[face[1]]}]" for face in self.faces])


class PiecesMatrix:
    def __init__(self, pieces: Set[Piece]):
        self.pieces = pieces
        self.matrix = {tuple(sorted(face)): set() for face in combinations_with_replacement(SYMBOLS, 2)}
        for piece in pieces:
            for face in piece.faces:
                self[face].add(piece)

    def __getitem__(self, face):
        print(face)
        return self.matrix[tuple(sorted(face))]

    def __repr__(self):
        title = f"Symbol Combinations In Pieces {sorted([piece.id for piece in self.pieces])} \n"
        header = "\t" + "\t".join([f"[{COLORED[y]}]" for y in SYMBOLS[1:]]) + "\n\n"
        lines = "\n\n".join(
            [f"[{COLORED[x]}]\t" + "\t".join(
                [",".join(
                    [str(p.id) for p in self[(x, y)]])
                    for y in SYMBOLS[1:]])
             for x in SYMBOLS[1:]])
        return title + header + lines


PIECE_1 = Piece(1, [
    Face(1, 5),
    Face(2, 1),
    Face(2, 4),
    Face(3, 3)])

PIECE_2 = Piece(2, [
    Face(4, 4),
    Face(5, 2),
    Face(3, 1),
    Face(1, 5)])

PIECE_3 = Piece(3, [
    Face(2, 2),
    Face(3, 4),
    Face(4, 5),
    Face(1, 3)])

PIECE_4 = Piece(4, [
    Face(3, 2),
    Face(2, 4),
    Face(5, 5),
    Face(4, 1)])

PIECE_5 = Piece(5, [
    Face(1, 1),
    Face(3, 4),
    Face(2, 5),
    Face(5, 3)])

ALL_PIECES = {PIECE_1, PIECE_2, PIECE_3, PIECE_4, PIECE_5}

if __name__ == "__main__":
    matrix = PiecesMatrix(ALL_PIECES)
    print(matrix)
