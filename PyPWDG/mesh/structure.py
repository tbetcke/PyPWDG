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
    
    def __init__(self, mesh, bndentities=[]):
        
        # All the following structure matrices map from double-sided faces to double-sided faces
        self.connectivity = sparseindex(mesh.intfaces, mesh.facemap[mesh.intfaces], mesh.nfaces)
        self.internal = sparseindex(mesh.intfaces, mesh.intfaces, mesh.nfaces)
        self.boundary = sparseindex(mesh.bndfaces, mesh.bndfaces, mesh.nfaces)
        self.average = (self.connectivity + self.internal)/2
        self.jump = self.internal - self.connectivity
        
        # The structure matrix approach works because at a structure level, we make the vandermondes
        # look like the identity.  This means that we create dim+1 vandermondes for each elt - effectively
        # we split each shape function into 4 components.  These need to be summed, which is what the 
        # eltstofaces matrix is for.
        self.eltstofaces = sparse.csc_matrix((numpy.ones(mesh.nfaces), numpy.hstack(mesh.etof), numpy.cumsum([0] + map(len, mesh.etof))))
        self.allfaces = sparse.csr_matrix(numpy.ones((mesh.nfaces, 1)))

        self.AD = self.average
        self.AN = self.jump / 2
        self.JD = self.jump
        self.JN = self.average * 2
        
        self.B = {}
        for b in bndentities:
            bf = numpy.array(filter(lambda f : mesh.bnd_entities[f] == b, mesh.bndfaces))
            self.B[b] = sparseindex(bf, bf, mesh.nfaces)
            
        
    def sumfaces(self, S):
        "Sum all the faces that contribute to each element"
        return (S * self.eltstofaces).__rmul__(self.eltstofaces.transpose())
    
    def sumrhs(self, G):
        "For the rows, sum the faces that contribute to each element; for the cols, sum everything"
        return (G * self.allfaces).__rmul__(self.eltstofaces.transpose())