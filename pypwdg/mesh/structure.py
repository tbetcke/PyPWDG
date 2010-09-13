'''
Created on Aug 6, 2010

@author: joel
'''
import numpy
import scipy.sparse as sparse
from pypwdg.utils.timing import print_timing

def getintptr(indices, n):
    """ Calculate an intptr matrix that indices one element in each row in indices"""
    intptrjumps = numpy.zeros(n+1, dtype=numpy.int)
    intptrjumps[indices+1] = 1
    return numpy.cumsum(intptrjumps)

def sparseindex(rows, cols, n):
    """ Return a csr matrix with a one at all the points given in rows and cols """
    return sparse.csr_matrix((numpy.ones(len(rows)), cols, getintptr(rows, n)),shape=(n,n))

class StructureMatrices(object):
    """ Calculate the structure matrices for a given mesh.
    
    Structure matrices operate at the level of the structure of the mesh, i.e. one row or column
    of a structure matrix corresponds to a face or element in the mesh.  
    
    
    AD: average of Dirichlet values across neighbouring faces
    AN: average of Neumann values across neighbouring faces
    JD: jump of Dirichlet values across neighbouring faces
    JN: jump of Neumann values across neighboring faces
    B: pull out the boundary faces 
    BE: BE[n] pulls out all the faces associated with entity n
    """
    
    def __init__(self, mesh, bndentities=[], facepartition=None):
        """ bndentities is a list of entities to build BE matrices for. """

        self.__nfaces = mesh.nfaces
        self._setFP(facepartition)
        # All the following structure matrices map from double-sided faces to double-sided faces        
        self.connectivity = sparseindex(mesh.intfaces, mesh.facemap[mesh.intfaces], mesh.nfaces)
        self.internal = sparseindex(mesh.intfaces, mesh.intfaces, mesh.nfaces)
        self.boundary = sparseindex(mesh.bndfaces, mesh.bndfaces, mesh.nfaces)
        self.average = (self.connectivity + self.internal)/2
        self.jump = self.internal - self.connectivity

        self.__AD = self.average
        self.__AN = self.jump / 2
        self.__JD = self.jump
        self.__JN = self.average * 2
                
        self.__BE = {}
        for b in bndentities:
            bf = numpy.array(filter(lambda f : mesh.bnd_entities[f] == b, mesh.bndfaces))
            self.__BE[b] = sparseindex(bf, bf, mesh.nfaces)
            
                # The structure matrix approach works because at a structure level, we make the vandermondes
        # look like the identity.  This means that we create dim+1 vandermondes for each elt - effectively
        # we split each shape function into 4 components.  These need to be summed, which is what the 
        # eltstofaces matrix is for.
        self.eltstofaces = sparse.csc_matrix((numpy.ones(mesh.nfaces), numpy.hstack(mesh.etof), numpy.cumsum([0] + map(len, mesh.etof))))
        self.allfaces = numpy.ones((mesh.nfaces))
    
    def _setFP(self, facepartition):
        self.__FP = None if facepartition is None else sparseindex(numpy.array(facepartition), numpy.array(facepartition), self.__nfaces)
    
    def withFP(self, facepartition):
        import copy
        SMFP = copy.copy(self)
        SMFP._setFP(facepartition)
        return SMFP
    
    def applyFP(self, sm):
        return sm if self.__FP is None else self.__FP * sm
        
    @print_timing    
    def sumfaces(self, S):
        """Sum all the faces that contribute to each element
        
        This reduces a faces x faces structure to an elts x elts structure
        """
        return (S * self.eltstofaces).__rmul__(self.eltstofaces.transpose())
    
    @print_timing    
    def sumrhs(self, G):
        """For the rows, sum the faces that contribute to each element; for the cols, sum everything
        
        This reduces a faces x faces structure to an elts x 1 structure
        """
        return G.__rmul__(self.eltstofaces.transpose()) * self.allfaces
    
    
    
    AD = property(lambda self: self.applyFP(self.__AD))
    AN = property(lambda self: self.applyFP(self.__AN))
    JD = property(lambda self: self.applyFP(self.__JD))
    JN = property(lambda self: self.applyFP(self.__JN))
    I = property(lambda self: self.applyFP(self.internal))
    B = property(lambda self: self.applyFP(self.boundary))
    BE = property(lambda self: dict(zip(self.__BE.keys(), map(self.applyFP, self.__BE.values()))))
            
    

            
