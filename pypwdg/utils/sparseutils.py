'''
Created on Oct 2, 2010

@author: joel
'''
import numpy
import scipy.sparse as sparse

def getintptr(indices, n):
    """ Calculate an intptr matrix that indices one element in each row in indices"""
    intptrjumps = numpy.zeros(n+1, dtype=numpy.int)
    intptrjumps[indices+1] = 1
    return numpy.cumsum(intptrjumps)

def sparseindex(rows, cols, m, n=None):
    """ Return a csr matrix with a one at all the points given in rows and cols """
    if n is None: n = m
    return sparse.csr_matrix((numpy.ones(len(rows)), cols, getintptr(rows, n)),shape=(m,n))
