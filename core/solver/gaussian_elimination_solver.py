from typing import Iterable

from core.solver.solver import LUSolver


class GaussianEliminationSolver(LUSolver):
    """class that solves linear system using Gaussian elimination method
    that implements partial pivoting. Currently only applies to well conditioned
    n variable n equations system.
    """

    def solve(self) -> Iterable:
        """for each row in matrix A, swap rows so that element (i, i) is the largest in abs
        value. then use row i to cancel rows below. returns solution to the linear system
        """
        for i in range(self.N - 1):
            self.partial_pivot_and_swap(i)
            self.eliminate(i)
        return self.backward_substitute()

    def partial_pivot_and_swap(self, i):
        """find largest element below element (i, i). then swap the rows
        so that the largest element is on the pivot of row i
        """
        row_to_swap = abs(self.A[i:, i]).argmax() + i
        if row_to_swap != i:
            self.swap_row(i, row_to_swap)
        self.print_matrix_if_verbose(self.A, title="Pivoting and Swap")

    def eliminate(self, i):
        """eliminate coefficients below element (i, i) by computing the scaling
        factors and add row i to following rows
        """
        pivot = self.A[i, i]
        for offset, a in enumerate(self.A[(i + 1) :, i]):
            scaling_factors = -1.0 * a / pivot
            self.add_row(i, i + offset + 1, scaling_factors)
        self.print_matrix_if_verbose(self.A, title="Eliminating")

    def swap_row(self, i, j):
        """swaps row i and row j in matrix A. i and j are zero index based"""
        temp = self.A[i].copy()
        self.A[i] = self.A[j]
        self.A[j] = temp

    def add_row(self, i, j, scaling_factor=1):
        """add row i * scaling_factor to row j"""
        temp = (self.A[i] * scaling_factor + self.A[j]).copy()
        self.A[j] = temp
