'''
Created on Aug 6, 2010

@author: joel
'''
import numpy as np
import scipy.sparse as ss

class AveragesAndJumps(object):
    def __init__(self, mesh):
        self.average = (mesh.connectivity + mesh.internal)/2
        self.jump = mesh.internal - mesh.connectivity

        self.AD = self.average
        self.AN = self.jump / 2
        self.JD = self.jump
        self.JN = self.average * 2
        self.Z = ss.csr_matrix(self.average.shape)
        d = np.zeros(mesh.nelements)
        d[mesh.partition] = 1
        self.I = ss.dia_matrix((d, [0]), shape = (mesh.nelements,)*2).tocsr()

        
def sumfaces(mesh,S):
    """Sum all the faces that contribute to each element
    
    This reduces a faces x faces structure to an elts x elts structure
    """
    return (S * mesh.elttofaces.transpose()).__rmul__(mesh.elttofaces)

def sumrhs(mesh,G):
    """For the rows, sum the faces that contribute to each element; for the cols, sum everything
    
    This reduces a faces x faces structure to an elts x 1 structure
    """
    return G.__rmul__(mesh.elttofaces) * ss.csr_matrix(np.ones((G.shape[1],1)))


            
