'''
Created on 23 Mar 2011

@author: rdodd
'''
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import lil_matrix
import numpy as np

def diagonal(stiffness):
    """LinearOperator corresponding to diagonal preconditioner"""
    Pm1 = 1. / stiffness.diagonal()
    def matvec(v):
        return Pm1 * v
    return LinearOperator(stiffness.shape, matvec=matvec, rmatvec=matvec)

def block_diagonal(stiffness):
    """LinearOperator corresponding to a block diagonal preconditioner"""
    def matvec(v):
        pass
    return LinearOperator(stiffness.shape, matvec)