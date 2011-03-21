'''
Created on Jul 14, 2010

@author: joel
'''

import numpy
from pypwdg.parallel.decorate import distribute, parallelmethod, immutable

@distribute()
class LocalVandermondes(object):
    """ Calculate Vandermonde matrices for each face in a mesh.
            
        The data is calculated lazily and cached.  One consequence of this is that this class could still be used
        in a distributed architecture where we don't want all the Vandermondes in every process.  
        
        attributes: 
            numbases: list of the number of basis functions on each face
    """
    
    def __init__(self, mesh, elttobasis, quadrule, usecache=True):
        """ Initialise the Vandermondes (nothing serious is calculated yet)
        
        mesh: the mesh.
        elttobasis: a list of lists of Basis objects (see .core.bases.PlaneWaves).  Order should correspond to 
        quadrule: Object containing a quadrature rule
        usecache: cache vandermondes (disable to save memory)
        """
        self.__mesh = mesh
        self.__elttobasis = elttobasis
        self.__quadpoints = quadrule.quadpoints
        self.__cache = {} if usecache else None 
        self.numbases = elttobasis.getSizes()[mesh.ftoe]
        self.indices = elttobasis.getIndices()[mesh.ftoe]
        
        
    def getVandermondes(self, faceid):
        """ Returns a tuple of (values, derivatives, weights) for functions on the face indexed by faceid """
         
        vandermondes = None if self.__cache is None else self.__cache.get(faceid) 
        if vandermondes==None:       
            e = self.__mesh.ftoe[faceid]
            normal = self.__mesh.normals[faceid]
            points = self.__quadpoints(faceid)
            vals = self.__elttobasis.getValues(e, points)
            derivs = self.__elttobasis.getDerivs(e, points, normal)
            vandermondes = (vals,derivs)
            if self.__cache is not None: self.__cache[faceid] = vandermondes 
            
        return vandermondes
    def getValues(self, faceid):
        return self.getVandermondes(faceid)[0]
    
    def getDerivs(self, faceid):
        return self.getVandermondes(faceid)[1]

    def getCachesize(self):
        return 0 if self.__cache is None else len(self.__cache)

class ElementVandermondes(object):
    """ Calculate vandermonde matrices at the element level."""
    def __init__(self, mesh, elttobasis, quadrule):
        self.__mesh = mesh
        self.__elttobasis = elttobasis
        self.__points = quadrule.quadpoints
        self.numbases = elttobasis.getSizes()
        self.indices = elttobasis.getIndices()
        
    def getValues(self, eltid):
        return self.__elttobasis.getValues(eltid, self.__points(eltid))
    
    def getDerivs(self, eltid):
        return self.__elttobasis.getDerivs(eltid, self.__points(eltid), None)
        
        
class LocalInnerProducts(object):
    """ A class to calculate inner products and matrix vector multiplication based on local vandermonde matrices """
    
    def __init__(self, vleft, vright, quadweights, axes = (0,0)):
        """ vleft and vright are callables that return a vandermonde matrix for a given id
            quadweights is a callable returning the quadrature weights for each id"""  
        self.__vleft = vleft
        self.__vright = vright
        self.__weights = quadweights
        self.axes = axes
        self.__cache = {}
    
    def product(self, i,j):
        """ Return the inner product of the ith thing against the jth thing """
        # It should be the case that weights(i) = weights(j), otherwise the
        # inner product makes no sense.
        p = self.__cache.get((i,j))
        if p is None:
            vl = self.__vleft(i).conj()
            vr = self.__vright(j)
            w = self.__weights(i).reshape((-1,) + (1,)*(len(vl.shape)-1))
            p = numpy.tensordot(vl * w, vr, self.axes)    
            
            if len(p.shape)==0: p = p.reshape(1,1)
            self.__cache[(i,j)] = p
#            print vl.shape, vr.shape, w.shape, p.shape
        
        return p        
