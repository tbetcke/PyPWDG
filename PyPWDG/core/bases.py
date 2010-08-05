'''
Created on Aug 5, 2010

@author: joel

All basis classes should have values and derivs methods and a property n giving the number of functions

'''

import numpy

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
    
    

