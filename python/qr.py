import numpy as np


def classical_gram_schmidt(A):
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


def modified_gram_schmidt(A):
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

if __name__ == "__main__":
    A = np.matrix([
        [1, -1, 4],
        [1, 4, -2],
        [1, 4,  2],
        [1, -1, 0]
    ])
    Q, R = modified_gram_schmidt(A)
    print(Q.T * Q)
    print(Q * R)
    print()