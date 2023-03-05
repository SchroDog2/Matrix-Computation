import numpy as np

from typing import Iterable


class GaussianEliminator:
    """class that solves linear system using Gaussian elimination method
    that implements partial pivoting. Currently only applies to well conditioned
    n variable n equations system.
    """
    def __init__(self, verbose=True) -> None:
        self.A = None
        self.b = None
        self.N = None
        self.verbose = verbose

    def set(self, A: Iterable[Iterable]) -> None:
        """set up the linear equations to be sovled.
        A is the extended coefficient matrix of the system that includes right hand side coefficients.
        example: 
            A = [
                [2, -6, -1, -38],
                [-3, -1, 7, -34],
                [-8, 1, -2, -20]
            ]
        """
        self.A = np.array(A, dtype=float)
        nrow, ncol = self.A.shape
        if not nrow + 1 == ncol:
            raise ValueError("Expecting square matrix for coefficient A")
        self.N = nrow
        if self.verbose:
            print("\n========== Linear System Set Up ============") 
            print(self.A)

    def solve(self) -> Iterable:
        """for each row in matrix A, swap rows so that element (i, i) is the largest in abs
        value. then use row i to cancel rows below. returns solution to the linear system
        """
        for i in range(self.N - 1):
            self.partial_pivot_and_swap(i)
            self.eliminate(i)
        return self.back_substitute()

    def swap_row(self, i, j):
        """swaps row i and row j in matrix A. i and j are zero index based"""
        temp = self.A[i].copy()
        self.A[i] = self.A[j]
        self.A[j] = temp

    def add_row(self, i, j, scaling_factor=1):
        """add row i * scaling_factor to row j"""
        temp = (self.A[i] * scaling_factor + self.A[j]).copy()
        self.A[j] = temp

    def partial_pivot_and_swap(self, i):
        """find largest element below element (i, i). then swap the rows
        so that the largest element is on the pivot of row i
        """
        row_to_swap = abs(self.A[i:, i]).argmax() + i
        if row_to_swap != i:
            self.swap_row(i, row_to_swap)
        if self.verbose:
            print("\n============ Pivoting and Swap ==============") 
            print(self.A)

    def eliminate(self, i):
        """eliminate coefficients below element (i, i) by computing the scaling
        factors and add row i to following rows
        """
        pivot = self.A[i, i]
        for offset, a in enumerate(self.A[(i+1):, i]):
            scaling_factors = -1.0 * a / pivot
            self.add_row(i, i + offset + 1, scaling_factors)
        if self.verbose:
            print("\n============== Eliminating ================") 
            print(self.A)

    def back_substitute(self):
        """backward substitute the gaussian eliminated upper trangular matrix
        to obtain the solution"""
        # solution vector
        x = np.ndarray(self.N, float)
        for i in range(self.N)[::-1]:
            x[i] = (self.A[i, self.N] - sum(self.A[i, (i+1):self.N] * x[i+1:])) / self.A[i, i]
        if self.verbose:
            print("\n============== Solved Solution ================") 
            print(x)
        return x