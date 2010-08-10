'''
Created on Jul 14, 2010

@author: joel
'''

import numpy

class LocalVandermondes(object):
    
    def __init__(self, mesh, elttobasis, quadratures):
        self.__mesh = mesh
        self.__elttobasis = elttobasis
        self.__quads = quadratures
        self.__cache = [None] * mesh.nfaces   
        self.__numbases = [sum([b.n for b in elttobasis[face[0]]]) for face in mesh.faces]     
        
    def getVandermondes(self, faceid):
        """ Returns a tuple of (values, derivatives, weights) for functions on the face indexed by faceid
        
        The data is calculated lazily and cached.  The point is to support a distributed architecture
        where we don't want all the vandermondes in every process.
        """
         
        vandermondes = self.__cache[faceid]
        if vandermondes==None:       
            face = self.__mesh.faces[faceid]
            normal = self.__mesh.normals[faceid]
            points = self.__quads(faceid)
            vals = numpy.hstack([b.values(points, normal) for b in self.__elttobasis[face[0]]])
            derivs = numpy.hstack([b.derivs(points, normal) for b in self.__elttobasis[face[0]]])
            self.__cache[faceid] = vandermondes = (vals,derivs)
            
        return vandermondes
        
    def getValues(self, faceid):
        return self.getVandermondes(faceid)[0]
    
    def getDerivs(self, faceid):
        return self.getVandermondes(faceid)[1]

    numbases = property(lambda self: self.__numbases)
        
        
class LocalInnerProducts(object):

    """ A class to calculate inner products and matrix vector multiplication based on local vandermonde matrices """
    def __init__(self, vleft, vright, quadweights):
        """ vleft and vright are callables that return a vandermonde matrix for a given faceid
            quadweights returns the quadrature weights for each faceid"""  
        self.__vleft = vleft
        self.__vright = vright
        self.__weights = quadweights
        self.__cache = {}
    
    def product(self, i,j):
        """ Return the inner product of the ith face against the jth face """
        # It should be the case that weights(i) = weights(j), otherwise the
        # inner product makes no sense.
        p = self.__cache.get((i,j))
        if p == None:
            p = numpy.dot(numpy.multiply(self.__vleft(i).conj().transpose(),self.__weights(i).flatten()), self.__vright(j))    
            self.__cache[(i,j)] = p
        
        return p        

    def matvec(self, g):
        return lambda i,j : numpy.dot(self.__vleft(i).H, numpy.multiply(self.__weights(i).reshape(-1,1), g[j]))