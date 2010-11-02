'''
Created on Aug 5, 2010

@author: joel

All basis classes should have values and derivs methods and a property n giving the number of functions

'''

import numpy
import math
import scipy.special as ss

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
        self.directions = directions.transpose()
        self.__k = k
    
    def values(self,x,n=None):
        """ return the values of the plane-waves at points x 
        
        x should be a m x dim array of points.
        n is ignored
        The return value is a m x self.n array
        """
        return numpy.exp(1j * self.__k * numpy.dot(x, self.directions))
    
    def derivs(self,x,n):
        """ return the directional derivatives of the plane-waves at points x and direction n 
        
        x should be a m x dim array.
        n should be a vector of length dim
        The return value is a m x self.n array
        """
        return 1j*self.__k*numpy.multiply(numpy.dot(n, self.directions), self.values(x,n))
    
    def __str__(self):
        return "PW "+ str(self.directions)
    
    """ the number of functions """
    n=property(lambda self: self.directions.shape[1])

    
class FourierHankelBessel(object):
    
    def __init__(self, origin, orders, k):
        self.__origin = origin.reshape(1,2)
        self.__orders = orders.reshape(1,-1)
        self.__k = k

    def rtheta(self, points):
        r = numpy.sqrt(numpy.sum(points**2, axis=1)).reshape(-1,1)
        theta = numpy.arctan2(points[:,1],points[:,0]).reshape(-1,1)
#        theta[numpy.isnan(theta)]=0
        return r, theta
    
    def values(self, points, n=None):
        r,theta = self.rtheta(points-self.__origin)
#        print numpy.hstack((points, points - self.__origin, theta, r, numpy.exp(1j * self.__orders * theta)))
        return self.rfn(self.__orders,self.__k * r) * numpy.exp(1j * self.__orders * theta)
    
    def derivs(self, points, n):
        poffset = points-self.__origin
        r,theta = self.rtheta(poffset)
        ent = numpy.exp(1j * self.__orders * theta)
        dr = self.__k * self.rfnd(self.__orders, self.__k * r, 1) * ent
        du = 1j * self.__orders * self.rfn(self.__orders, self.__k * r) * ent
        x = poffset[:,0].reshape(-1,1)
        y = poffset[:,1].reshape(-1,1)
        r2 = r**2
        Js = numpy.hstack((x/r, -y/r2, y/r, x/r2)).reshape((-1,1,2,2))
        nJs = numpy.sum(n.reshape(-1,1,2,1) * Js, axis=2)        
        dru = numpy.concatenate((dr[:,:,numpy.newaxis], du[:,:,numpy.newaxis]), axis=2)
        return numpy.sum(nJs * dru, axis=2)
    
    n=property(lambda self: self.__orders.shape[1])

class EmptyBasis(object):
    
    def __init__(self,n):
        """Create an empty placeholder basis that returns size n"""
        
        self.__n=n
        
    n=property(lambda self: self.__n)
 

class FourierBessel(FourierHankelBessel):
    
    def __init__(self, origin, orders, k):
        FourierHankelBessel.__init__(self, origin, orders, k)

    def rfn(self, n, x):
        return ss.jn(n,x)
    def rfnd(self, n, x, d):
        return ss.jvp(n,x,d)
        
class FourierHankel(FourierHankelBessel):
    
    def __init__(self, origin, orders, k):
        FourierHankelBessel.__init__(self, origin, orders, k)
        
    def rfn(self, n, x):
        return ss.hankel1(n,x)
    def rfnd(self, n, x, d):
        return ss.h1vp(n,x,d)        


class BasisReduce(object):
    """ Reduce a basis object to return just one function """
    def __init__(self, pw, x):
        self.pw = pw
        self.x = x
        self.n = 1
        
    def values(self, points, n=None):
        return numpy.dot(self.pw.values(points, n), self.x).reshape(-1,1)

    def derivs(self, points, n):
        return numpy.dot(self.pw.derivs(points, n), self.x).reshape(-1,1)

class BasisCombine(object):
    """ Combine several (reduced) basis objects"""
    def __init__(self, bases, x):
        self.bases = bases
        self.x = x
        self.n = 1
        
    def values(self, points, n=None):
        return numpy.dot(numpy.hstack([b.values(points, n) for b in self.bases]), self.x).reshape(-1,1)
        
    def derivs(self, points, n):
        return numpy.dot(numpy.hstack([b.derivs(points, n) for b in self.bases]), self.x).reshape(-1,1)
