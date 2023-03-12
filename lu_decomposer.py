from typing import Iterable, Tuple

import numpy as np


# todo: this class should live in a separate module for matrix operation
class LUDecomposer:
    """class that decomposes a square matrix using LU decomposition,
    namely A = LU where L is a lower triangular matrix and U is an
    upper triangular matrix.
    """
    def __init__(self, verbose=True) -> None:
        self.L = None
        self.U = None
        self.N = None
        self.verbose = verbose

    def set(self, A: Iterable[Iterable]) -> None:
        """set the matrix to be solved. A must be a square matrix
        example:
            A = [
                [7, 2, -3],
                [2, 5, -3],
                [1, -1, -6]
            ]
        """
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=float)
        nrow, ncol = A.shape
        if nrow != ncol:
            raise ValueError("A must be square matrix.")
        self.N = nrow
        self.L = np.identity(self.N)
        self.U = A
        self.print_matrix_if_verbose(self.L, title="L Matrix")
        self.print_matrix_if_verbose(self.U, title="U Matrix")

    def decompose(self) -> Tuple[Iterable[Iterable], Iterable[Iterable]]:
        """returns tuple that contains L and U matrices"""
        # eliminate rows in U matrix below row i
        for i in range(self.N - 1):
            pivot = self.U[i, i]
            for offset, a in enumerate(self.U[(i + 1) :, i]):
                scaling_factor = -1.0 * a / pivot
                self.add_row_u(i, i + offset + 1, scaling_factor)
                self.L[i + offset + 1, i] = -1.0 * scaling_factor
            self.print_matrix_if_verbose(self.L, title=f"L Matrix - Iter {i+1}")
            self.print_matrix_if_verbose(self.U, title=f"U Matrix - Iter {i+1}")
        return self.L, self.U

    def add_row_u(self, i, j, scaling_factor=1):
        """add row i * scaling_factor to row j for U matrix"""
        temp = (self.U[i] * scaling_factor + self.U[j]).copy()
        self.U[j] = temp

    def print_matrix_if_verbose(self, A, title=None):
        """print the given matrix if verbose"""
        if self.verbose:
            print(f"\n========== {title} ============")
            print(A)
