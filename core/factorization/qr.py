from typing import Tuple
import numpy as np


def classical_gram_schmidt(A) -> Tuple[np.matrix, np.matrix]:
    """
    For each vector in the matrix, remove its projection onto the data set, 
    normalize what is left, and include it in the orthogonal set. 

    X is the original set of vectors 
    Q is the resulting set of orthogonal vectors
    R is the set of coefficients, organized into an upper triangular matrix.
    """
    if not isinstance(A, np.matrix):
        A = np.matrix(A)

    m, n = A.shape

    Q = np.matrix(np.zeros([m, n]))
    R = np.matrix(np.zeros([n, n]))

    # loop column vectors in A
    for j in range(n):
        # inner product of A column vector j to previous vectors in Q
        R[:j, j] = Q[:, :j].T * A[:, j]
        
        # subtract projections
        Q[:, j] = A[:, j] - Q[:, :j] * R[:j, j]
        
        # compute norm and normalize
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / R[j, j]

    return Q, R


def modified_gram_schmidt(A) -> Tuple[np.matrix, np.matrix]:
    """a modified variant of classical Gram-Schmidt method.
    The idea is to orthogonalize against emerging set of vectors 
    instead of against the orignal set. This method is expected
    to be more numerically stable compared to the classical one.
    """
    if not isinstance(A, np.matrix):
        A = np.matrix(A)

    m, n = A.shape

    Q = np.matrix(np.zeros([m, n]))
    R = np.matrix(np.zeros([n, n]))

    for j in range(n):
        Q[:, j] = A[:, j]

        # loop previous orthoganolized column vectors
        for i in range(j):
            R[i, j] = Q[:, i].T * A[:, j]
            Q[:, j] = Q[:, j] - Q[:, i] * R[i, j]
        
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / R[j, j]

    return Q, R


# Householder reflection
# reference: https://blogs.mathworks.com/cleve/2016/10/03/householder-reflections-and-the-qr-decomposition/

def generate_house_v(x):
    """ Generate Householder reflection vector v
    returns v with norm(v) = sqrt(2).
    
    v is the Householder reflection vector that is used 
    in the reflection operator H such that:
    H(v, x) = x - v * (v.T * x) = -+ norm(x) * e1
    """
    normx = np.linalg.norm(x)
    if normx != 0:
        # make v unit vector
        v = x / normx

        # add unit vector e1 to v. sign of e1 is chosen to be the one
        # that is opposite to sign of v[0]. when v is vertical to x-axis
        # use positive sign by default
        v[0] = v[0] + np.sign(v[0]) + (v[0] == 0)

        # (a+1)^2 + b^2 + c^2 = 2*(a+1)
        v = v / np.sqrt(np.abs(v[0]))
    else:
        v = x
        v[0] = np.sqrt(2)

    return v


def house_apply(U, X):
    """apply householder reflectiosn
    Z = _house_apply(U, X) with U from `house_qr`
    computes Q*X without actually computing Q
    """
    Z = X
    _, n = U.shape
    for j in reversed(range(n)):
        Z = H(U[:, j], Z)
    return Z


def house_apply_transpose(U, X):
    """apply householder transposed reflections
    Z = house_apply(U, X) with U from `house_qr`
    computes Q.T * X without actually computing Q.T
    """
    Z = X.copy()
    _, n = U.shape
    for j in range(n):
        Z = H(U[:, j], Z)
    return Z


def H(u, x):
    """reflection operator"""
    return x - u * (u.T * x)


def house_qr(A, compute_q=True):
    """Compute R by applying Householder reflections to matrix A
    a column at a time. The reflection will create zeros below
    diagonal for each column j.

    When compute_q is True, return Q, R.
    Otherwise, return the U matrix that stores the v vectors.
    
    A very good vedio that explains this algorithm
    https://www.youtube.com/watch?v=yyOXDSlY8d4&t=1002s
    """
    if not isinstance(A, np.matrix):
        A = np.matrix(A)
    
    m, n = A.shape

    U = np.matrix(np.zeros([m, n]))
    R = A.copy()

    # the square matrix embedded in the rectangular matrix
    for j in range(min(m, n)):
        u = generate_house_v(R[j:, j])
        U[j:, j] = u
        R[j:, j:] = H(u, R[j:, j:])
        R[j+1:, j] = 0

    # compute Q and return Q, R
    if compute_q:
        if m == n:
            I = np.identity(n)
        elif m > n:
            I = np.row_stack([np.identity(n), np.zeros([1, n])])
        else:
            raise NotImplementedError()
        Q = house_apply(U, I)
        return Q, R[:min(m, n), :min(m, n)]
    
    # return U, R
    else:
        return U, R[:min(m, n), :min(m, n)]

# alias
householder_reflection = house_qr
