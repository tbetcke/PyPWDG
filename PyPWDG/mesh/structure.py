'''
Created on Aug 6, 2010

@author: joel
'''
import numpy
import scipy.sparse as sparse

def getintptr(indices, n):
    """ Calculate an intptr matrix that indices one element in each row in indices"""
    intptrjumps = numpy.zeros(n+1, dtype=numpy.int)
    intptrjumps[indices+1] = 1
    return numpy.cumsum(intptrjumps)

def sparseindex(rows, cols, n):
    """ Return a csr matrix with a one at all the points given in rows and cols """
    return sparse.csr_matrix((numpy.ones(len(rows)), cols, getintptr(rows, n)))

class StructureMatrices(object):
    
    def __init__(self, mesh):
        
        # All the following structure matrices map from double-sided faces to double-sided faces
        self.connectivity = sparseindex(mesh.intfaces, mesh.facemap[mesh.intfaces], mesh.nfaces)
        self.internal = sparseindex(mesh.intfaces, mesh.intfaces, mesh.nfaces)
        self.boundary = sparseindex(mesh.bndfaces, mesh.bndfaces, mesh.nfaces)
        self.average = (self.connectivity + self.internal)/2
        self.jump = self.internal - self.connectivity
        
        # The structure is at the level of double faces.  We'll need to sum the contributions from 
        # each face onto each element
        edfi = numpy.hstack([numpy.ones(len(fs))*e for e,fs in enumerate(mesh.etof.values())])
        self.eltstofaces = sparse.csr_matrix((numpy.ones(mesh.nfaces), edfi, range(0, mesh.nfaces+1)))

        self.AD = self.average
        self.AN = self.jump / 2
        self.JD = self.jump
        self.JN = self.average * 2
        
    def sumfaces(self, S):
        return (S * self.eltstofaces).__rmul__(self.eltstofaces.transpose())