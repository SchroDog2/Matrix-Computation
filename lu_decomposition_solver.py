from typing import Iterable

import numpy as np

from lu_decomposer import LUDecomposer
from solver import GaussianSolver


class LUDecompositionSolver(GaussianSolver):
    """class that decomposes a matrix using LU decomposition,
    namely A = LU where L is a lower triangular matrix and U
    is an upper triangular matrix.
    """

    def __init__(self, verbose=True) -> None:
        super().__init__(verbose)
        self.L = None
        self.U = None
        self.decomposer = None

    # todo: refactor base class init to accept b
    def set(self, A: Iterable[Iterable]) -> None:
        super().set(A)
        self.decomposer = LUDecomposer(self.verbose)
        self.decomposer.set(self.A[:, : self.N])

    def solve(self):
        # LU decomposition
        self.L, self.U = self.decomposer.decompose()

        # solve Ld = b
        b = self.A[:, self.N].reshape(self.N, 1)
        self.A = np.concatenate([self.L, b], axis=1)
        d = self.forward_substitute()

        # solve Ux = d
        d = d.reshape(self.N, 1)
        self.A = np.concatenate([self.U, d], axis=1)
        x = self.backward_substitute()
        return x


def matrix_inv_lu(A, verbose=True):
    """returns inverse of matrix A using LU decompostion method"""
    # check A should be square matrix
    A = np.array(A, dtype=float)
    nrow, ncol = A.shape
    if not nrow == ncol:
        raise ValueError("A must be a square matrix")
    solver = LUDecompositionSolver(verbose)
    
    # each column of the inverse matrix is a separate solution
    # of a linear system by setting proper b vector
    inv = np.zeros(shape=(nrow, ncol), dtype=float)
    for i in range(nrow):
        # set the righthand side b vector
        b = np.zeros(shape=(nrow, 1))
        b[i] = 1
        # solve the linear equations
        solver.set(np.concatenate([A, b], axis=1))
        x = solver.solve()
        # place the result into the solution matrix
        inv[:, i] = x
    
    if verbose:
        print("============ Solved Inverse Matrix ============")
        print(inv)

    return inv
