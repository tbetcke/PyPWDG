'''
Created on Aug 5, 2010

@author: joel

All basis classes should have values and derivs methods and a property n giving the number of functions

'''

import numpy
import math

def cubeDirections(n):
    """ Return n^2 directions roughly parallel to (1,0,0)"""
    
    r = [2.0*t/(n+1)-1 for t in range(1,n+1)]
    return [v / math.sqrt(numpy.dot(v,v)) for v in [numpy.array([1,y,z]) for y in r for z in r]]

def cubeRotations(directions):
    """ Rotate each direction through the faces of the cube"""
    M = numpy.array(directions)
    return numpy.vstack([numpy.vstack([M,-M])[:,i] for i in [(0,1,2),(1,2,0),(2,0,1)] ])

def circleDirections(n):
    """ return n equi-spaced directions on a circle """
    theta = numpy.arange(n).reshape((-1,1)) * 2*math.pi / n
    return numpy.hstack((numpy.cos(theta), numpy.sin(theta)))

class PlaneWaves(object):
    
    def __init__(self, directions, k):
        """ directions should be a n x dim array of directions.  k is the wave number """
        self.__directions = directions.transpose()
        self.__k = k
    
    def values(self,x,n=None):
        """ return the values of the plane-waves at points x 
        
        x should be a m x dim array of points.
        n is ignored
        The return value is a m x self.n array
        """
        return numpy.exp(1j * self.__k * numpy.dot(x, self.__directions))
    
    def derivs(self,x,n):
        """ return the directional derivatives of the plane-waves at points x and direction n 
        
        x should be a m x dim array.
        n should be a vector of length dim
        The return value is a m x self.n array
        """
        return 1j*self.__k*numpy.multiply(numpy.dot(n, self.__directions), self.values(x,n))
    
    """ the number of functions """
    n=property(lambda self: self.__directions.shape[1])
    
    

