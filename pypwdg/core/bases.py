'''
Created on Aug 5, 2010

@author: joel

All basis classes should have values and derivs methods and a property n giving the number of functions

'''

import abc
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

def planeWaveBases(mesh, k, nplanewaves):        
    if mesh.dim==2:
        dirs = circleDirections(nplanewaves)
    else:
        dirs = cubeRotations(cubeDirections(nplanewaves))
    pw = PlaneWaves(dirs,k)
    etob=ElementToBases(mesh)
    for e in range(mesh.nelements):
        etob.addBasis(e,PlaneWaves(dirs,k))
    return etob


def fourierBesselBases(mesh, k, orders):
    if not mesh.dim==2: raise Exception("Bessel functions are only defined in 2D.")
    etob = ElementToBases(mesh)
    for e in range(mesh.nelements):
        origin=mesh.nodes[mesh.elements[e][0]]
        etob.addBasis(e, FourierBessel(origin,orders,k))
    return etob

class ElementToBases(object):
    def __init__(self, mesh):
        self.mesh = mesh
        self.etob = {}
        self.sizes = None     
        self.indices = None 
        self.version = 0  
        
    def getValues(self, eid, points, normal=None):
        """ Return the values of the basis for element eid at points"""
        bases = self.etob.get(eid)
        if bases==None:
            return numpy.zeros(len(points),0)
        else:
            return numpy.hstack([b.values(points, normal) for b in bases])
    
    def getDerivs(self, eid, points, normal):
        """ Return the directional derivatives of the basis for element eid at points"""
        bases = self.etob.get(eid)
        if bases==None:
            return numpy.zeros(len(points),0)
        else:
            return numpy.hstack([b.derivs(points, normal) for b in bases])
    
    def _reset(self):
        self.sizes = None     
        self.indices = None 
        self.version +=1
    
    def addBasis(self, eid, b):
        """ Add a basis object to element eid"""
        bases = self.etob.setdefault(eid, [])
        bases.append(b)
        self._reset()
        return self
    
    def addUniformBasis(self, b):
        for e in range(self.mesh.nelements):
            self.addBasis(e, b)   
        self._reset()
        return self
    
    def setEtoB(self, etob = {}):
        self.etob = etob
        self._reset()
    
    def getSizes(self):
        if self.sizes is None:
            self.sizes = numpy.array([sum([b.n for b in self.etob.get(e,[])]) for e in range(self.mesh.nelements)])
        return self.sizes
        
    def getIndices(self):
        """ Return the global index for element eid"""
        if self.indices is None:
            sizes = self.getSizes()
            self.indices = numpy.cumsum(numpy.concatenate(([0], sizes)))
        return self.indices 

    def setRefractiveElement(self,eid,refr):
        """Set refractive index of all basis objects on element eid"""
        bases=self.etob.setdefault(eid,[])
        for b in bases: b.setRefractive(refr)


    def setRefractive(self,geomDict):
        """Set refractive indices of the elements

           geomDict is a dictionary that maps geometric entities
           to the corresponding refractive indices.
        """
        for eid in range(self.mesh.nelements):
            self.setRefractiveElement(eid,geomDict[self.mesh.elemIdentity[eid]])
       
    def getRefractive(self):
        for eid in range(self.mesh.nelements):
            for b in self.etob[eid]:
                print b.getRefractive()


class Basis(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def values(self,x,n=None):
        pass

    @abc.abstractmethod
    def derivs(self,x,n):
        pass

    def setRefractive(self,val):
        self.refr=val

    def getRefractive(self):
        return self.refr

class PlaneWaves(Basis):
    
    def __init__(self, directions, k):
        """ directions should be a n x dim array of directions.  k is the wave number """
        self.directions = directions.transpose()
        self.__k = k
        self.refr=1
    
    def values(self,x,n=None):
        """ return the values of the plane-waves at points x 
        
        x should be a m x dim array of points.
        n is ignored
        The return value is a m x self.n array
        """
        return numpy.exp(1j * self.__k * self.refr * numpy.dot(x, self.directions))
    
    def derivs(self,x,n):
        """ return the directional derivatives of the plane-waves at points x and direction n 
        
        x should be a m x dim array.
        n should be a vector of length dim
        The return value is a m x self.n array
        """
        return 1j*self.__k*self.refr*numpy.multiply(numpy.dot(n, self.directions), self.values(x,n))
    
    def __str__(self):
        return "PW "+ str(self.directions)
    
    """ the number of functions """
    n=property(lambda self: self.directions.shape[1])

    
class FourierHankelBessel(Basis):
    
    def __init__(self, origin, orders, k):
        self.__origin = origin.reshape(1,2)
        self.__orders = orders.reshape(1,-1)
        self.__k = k
        self.refr=1

    def rtheta(self, points):
        r = numpy.sqrt(numpy.sum(points**2, axis=1)).reshape(-1,1)
        theta = numpy.arctan2(points[:,1],points[:,0]).reshape(-1,1)
#        theta[numpy.isnan(theta)]=0
        return r, theta
    
    def values(self, points, n=None):
        r,theta = self.rtheta(points-self.__origin)
#        print numpy.hstack((points, points - self.__origin, theta, r, numpy.exp(1j * self.__orders * theta)))
        return self.rfn(self.__orders,self.__k * self.refr* r) * numpy.exp(1j * self.__orders * theta)
    
    def derivs(self, points, n):
        poffset = points-self.__origin
        r,theta = self.rtheta(poffset)
        ent = numpy.exp(1j * self.__orders * theta)
        dr = (self.__k * self.refr* self.rfnd(self.__orders, self.__k *
                self.refr* r, 1) * ent)
        du = 1j * self.__orders * self.rfn(self.__orders, self.__k *self.refr * r) * ent
        x = poffset[:,0].reshape(-1,1)
        y = poffset[:,1].reshape(-1,1)
        r2 = r**2
        Js = numpy.hstack((x/r, -y/r2, y/r, x/r2)).reshape((-1,1,2,2))
        nJs = numpy.sum(n.reshape(-1,1,2,1) * Js, axis=2)        
        dru = numpy.concatenate((dr[:,:,numpy.newaxis], du[:,:,numpy.newaxis]), axis=2)
        return numpy.sum(nJs * dru, axis=2)
    
    n=property(lambda self: self.__orders.shape[1])

class EmptyBasis(Basis):
    
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


class BasisReduce(Basis):
    """ Reduce a basis object to return just one function """
    def __init__(self, pw, x):
        self.pw = pw
        self.x = x
        self.n = 1
        
    def values(self, points, n=None):
        return numpy.dot(self.pw.values(points, n), self.x).reshape(-1,1)

    def derivs(self, points, n):
        return numpy.dot(self.pw.derivs(points, n), self.x).reshape(-1,1)

    def setRefractive(self,val):
        self.pw.setRefractive(val)



class BasisCombine(Basis):
    """ Combine several (reduced) basis objects"""
    def __init__(self, bases, x):
        self.bases = bases
        self.x = x
        self.n = 1
        
    def values(self, points, n=None):
        return numpy.dot(numpy.hstack([b.values(points, n) for b in self.bases]), self.x).reshape(-1,1)
        
    def derivs(self, points, n):
        return numpy.dot(numpy.hstack([b.derivs(points, n) for b in self.bases]), self.x).reshape(-1,1)

    def setRefractive(self,val):
        for b in self.bases: b.setrefractive(val)



