'''
Created on Jul 27, 2010

@author: tbetcke
'''
import numpy

class Mesh(object):
    """Mesh - Object that stores all necessary mesh information
       
       Usage:
       mesh=Mesh(meshDict), where meshDict is a Dictionary returned
       from GmshReader
       
    """
    
    def __init__(self,mesh_dict):
        
        self.__mesh=mesh_dict
        # Extract all faces and create face to vertex map
        self.__ftov = ([tuple(sorted(vs['nodes'][0:i]+vs['nodes'][i+1:4])) for vs in self.__mesh['elements'] for i in range(0,4) if vs['type']==4])
                
        # Create Face to Vertex Sparse Matrix
        from scipy.sparse import csr_matrix
        ij=[[3*i for i in range(len(self.__ftov))],
            [vs[i]-1 for vs in self.__ftov for i in range(3)]] # vs[i]-1 since nodes should start at zero
        data=numpy.ones(len(ij[0]))
        FToV=csr_matrix((data,ij),dtype='i')
        

        
    
        
    