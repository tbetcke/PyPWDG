'''
Created on 23 Mar 2011

Module of various preconditioners for problems

@author: rdodd
'''
from scipy.sparse.linalg import LinearOperator, factorized

def diagonal(matrix):
    """LinearOperator corresponding to diagonal preconditioner"""
    Pm1 = 1. / matrix.diagonal()
    def matvec(v):
        return Pm1 * v
    return LinearOperator(matrix.shape, matvec=matvec, rmatvec=matvec)

def block_diagonal(matrix, idxs):
    """LinearOperator corresponding to a block diagonal preconditioner"""
    LU = [factorized(matrix[idx,:][:,idx].tocsc()) for idx in idxs]    
    def matvec(v):
        for i in range(len(idxs)):
            v[idxs[i]] = LU[i](v[idxs[i]])
        return v
    return LinearOperator(matrix.shape, matvec=matvec)