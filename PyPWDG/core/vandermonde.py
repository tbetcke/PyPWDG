'''
Created on Jul 14, 2010

@author: joel
'''

import numpy

class LocalVandermondes(object):
    """ Calculate Vandermonde matrices for each face in a mesh.
            
        The data is calculated lazily and cached.  One consequence of this is that this class could still be used
        in a distributed architecture where we don't want all the Vandermondes in every process.  
        
        attributes: 
            numbases: list of the number of basis functions on each face
    """
    
    def __init__(self, mesh, elttobasis, quadpoints, usecache=True):
        """ Initialise the Vandermondes (nothing serious is calculated yet)
        
        mesh: the mesh.
        elttobasis: a list of lists of Basis objects (see .core.bases.PlaneWaves).  Order should correspond to 
        quadpoints: callable that returns quadrature points for each face
        usecache: cache vandermondes (disable to save memory)
        """
        self.__mesh = mesh
        self.__elttobasis = elttobasis
        self.__quadpoints = quadpoints
        self.__cache = [None] * mesh.nfaces if usecache else None 
        self.__numbases = [sum([b.n for b in elttobasis[face[0]]]) for face in mesh.faces]     
        
    def getVandermondes(self, faceid):
        """ Returns a tuple of (values, derivatives, weights) for functions on the face indexed by faceid """
         
        vandermondes = None if self.__cache is None else self.__cache[faceid] 
        if vandermondes==None:       
            face = self.__mesh.faces[faceid]
            normal = self.__mesh.normals[faceid]
            points = self.__quadpoints(faceid)
            vals = numpy.hstack([b.values(points, normal) for b in self.__elttobasis[face[0]]])
            derivs = numpy.hstack([b.derivs(points, normal) for b in self.__elttobasis[face[0]]])
            vandermondes = (vals,derivs)
            if self.__cache is not None: self.__cache[faceid] = vandermondes 
            
        return vandermondes
        
    def getValues(self, faceid):
        return self.getVandermondes(faceid)[0]
    
    def getDerivs(self, faceid):
        return self.getVandermondes(faceid)[1]

    numbases = property(lambda self: self.__numbases)
        
        
class LocalInnerProducts(object):
    """ A class to calculate inner products and matrix vector multiplication based on local vandermonde matrices """
    
    def __init__(self, vleft, vright, quadweights):
        """ vleft and vright are callables that return a vandermonde matrix for a given id
            quadweights is a callable returning the quadrature weights for each id"""  
        self.__vleft = vleft
        self.__vright = vright
        self.__weights = quadweights
        self.__cache = {}
    
    def product(self, i,j):
        """ Return the inner product of the ith thing against the jth thing """
        # It should be the case that weights(i) = weights(j), otherwise the
        # inner product makes no sense.
        p = self.__cache.get((i,j))
        if p == None:
            p = numpy.dot(numpy.multiply(self.__vleft(i).conj().transpose(),self.__weights(i).flatten()), self.__vright(j))    
            self.__cache[(i,j)] = p
        
        return p        