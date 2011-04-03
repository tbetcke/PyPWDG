'''
Created on Aug 5, 2010

@author: joel

All basis classes should have values and derivs methods and a property n giving the number of functions

'''

import abc
import numpy
import math
import scipy.special as ss
import numpy as np

import pypwdg.parallel.decorate as ppd
import pypwdg.parallel.distributeddict as ppdd
from pypwdg.parallel.mpiload import mpiloaded
import pypwdg.mesh.meshutils as pmmu

def cubeDirections(n):
    """ Return n^2 directions roughly parallel to (1,0,0)"""
    
    r = [2.0*t/(n+1)-1 for t in range(1,n+1)]
    return [v / math.sqrt(np.dot(v,v)) for v in [np.array([1,y,z]) for y in r for z in r]]

def cubeRotations(directions):
    """ Rotate each direction through the faces of the cube"""
    M = np.array(directions)
    return np.vstack([np.vstack([M,-M])[:,i] for i in [(0,1,2),(1,2,0),(2,0,1)] ])

def circleDirections(n):
    """ return n equi-spaced directions on a circle """
    theta = np.arange(n).reshape((-1,1)) * 2*math.pi / n
    return np.hstack((np.cos(theta), np.sin(theta)))

def uniformdirs(dim, npw):
    if dim==2:
        return circleDirections(npw)
    else:
        return cubeRotations(cubeDirections(npw))
    
def planeWaveBases(dim, k, nplanewaves):
    dirs = uniformdirs(dim, nplanewaves)
    pw = [PlaneWaves(dirs,k)]
    return UniformBases(pw)

@ppd.distribute()
class UniformBases(object):
    def __init__(self, b):
        self.b = b

    @ppd.parallelmethod(None, None)    
    def populate(self, mesh, etob):    
        for e in mesh.partition:
            etob[e] = self.b

@ppd.distribute()
class FourierBesselBases(object):

    def __init__(self, k, orders):
        self.orders = orders
        self.k = k
    
    @ppd.parallelmethod(None, None)
    def populate(self, mesh, etob):
        if not mesh.dim==2: raise Exception("Bessel functions are only defined in 2D.")
        for e in mesh.partition:
            origin=mesh.nodes[mesh.elements[e][0]]
            etob[e] = [FourierBessel(origin,self.orders,self.k)]

@ppd.distribute()
class PlaneWaveVariableN(object):
    def __init__(self, k, dirs, eton):
        self.k = k
        self.eton = eton
        self.dirs = dirs
        
    @ppd.parallelmethod(None, None)
    def populate(self, mesh, etob):
        for e in mesh.partition:
            etob[e] = [PlaneWaves(self.dirs, self.k * self.eton[mesh.elemIdentity[e]])]        
    
@ppd.distribute()
class ReferenceBases(object):
    """ A basis that uses a (affine) transformation to a reference element (old skool)"""
    def __init__(self, reference):
        self.reference = reference
        
    @ppd.parallelmethod(None, None)
    def populate(self, mesh, etob):
        mems = pmmu.MeshElementMaps(mesh)
        for e in mesh.partition:
            etob[e] = [Reference(mems.getMap(e), self.reference)]

def getSizes(etob, mesh):
    return np.array([sum([b.n for b in etob.get(e,[])]) for e in range(mesh.nelements)])    

def constructBasis(mesh, basisrule):
    manager = ppdd.ddictmanager(ppdd.elementddictinfo(mesh), True)
    etob = manager.getDict()
    basisrule.populate(mesh, etob)  
    manager.sync()   
    return ElementToBases(etob, mesh)

class ElementToBases(object):
    def __init__(self, etob, mesh):
        self.etob = etob
        self.sizes = np.array([sum([b.n for b in etob.get(e,[])]) for e in range(mesh.nelements)])     
        self.indices = np.cumsum(np.concatenate(([0], self.sizes))) 
        
    def getValues(self, eid, points):
        """ Return the values of the basis for element eid at points"""
        bases = self.etob.get(eid)
        if bases==None:
            return np.zeros(len(points),0)
        else:
            return np.hstack([b.values(points) for b in bases])
    
    def getDerivs(self, eid, points, normal = None):
        """ Return the directional derivatives of the basis for element eid at points
        
           if normal == None, returns the gradient on the standard cartesian grid
        """
        bases = self.etob.get(eid)
        if bases==None:
            return np.zeros(len(points),0) if normal is not None else np.zeros(len(points), 0, points.shape[1])
        else:
            return np.hstack([b.derivs(points, normal) for b in bases])
    
    def getSizes(self):
        return self.sizes
        
    def getIndices(self):
        return self.indices 


class Basis(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def values(self,x):
        pass

    @abc.abstractmethod
    def derivs(self,x,n = None):
        pass

class Reference(Basis):
    
    def __init__(self, map, reference):
        self.mapi = map.inverse
        self.n = reference.size
    
    def values(self, x):
        return self.reference.values(self.mapi.apply(x))
    
    def derivs(self, x, n = None):        
        derivs = np.dot(self.reference.derivs(self.map.inverse.apply(x)),self.mapi.linear.transpose())
        return derivs if n is None else np.dot(derivs, n)  
        

class PlaneWaves(Basis):
    
    def __init__(self, directions, k):
        """ directions should be a n x dim array of directions.  k is the wave number """
        self.directions = directions.transpose()
        self.__k = k
    
    def values(self,x):
        """ return the values of the plane-waves at points x 
        
        x should be a m x dim array of points.
        n is ignored
        The return value is a m x self.n array
        """
        return np.exp(1j * self.__k * np.dot(x, self.directions))
    
    def derivs(self,x,n=None):
        """ return the directional derivatives of the plane-waves at points x and direction n 
        
        x should be a m x dim array.
        n should be a vector of length dim
        The return value is a m x self.n array
        """
        vals = self.values(x)
        if n == None:
            return 1j * self.__k * np.multiply(vals[..., np.newaxis], self.directions.transpose()[np.newaxis, ...])
        else:
            return 1j * self.__k*np.multiply(np.dot(n, self.directions), vals)
    
    def __str__(self):
        return "PW basis "+ str(self.directions)
    
    """ the number of functions """
    n=property(lambda self: self.directions.shape[1])
    
class FourierHankelBessel(Basis):
    
    def __init__(self, origin, orders, k):
        self.__origin = np.array(origin).reshape(1,2)
        self.__orders = np.array(orders).reshape(1,-1)
        self.__k = k

    def rtheta(self, points):
        r = np.sqrt(np.sum(points**2, axis=1)).reshape(-1,1)
        theta = np.arctan2(points[:,1],points[:,0]).reshape(-1,1)
#        theta[np.isnan(theta)]=0
        return r, theta
    
    def values(self, points):
        r,theta = self.rtheta(points-self.__origin)
#        print np.hstack((points, points - self.__origin, theta, r, np.exp(1j * self.__orders * theta)))
        return self.rfn(self.__orders,self.__k * r) * np.exp(1j * self.__orders * theta)
    
    def derivs(self, points, n):
        poffset = points-self.__origin
        r,theta = self.rtheta(poffset)
        ent = np.exp(1j * self.__orders * theta)
        dr = (self.__k * self.rfnd(self.__orders, self.__k * r, 1) * ent)
        du = 1j * self.__orders * self.rfn(self.__orders, self.__k *r) * ent
        x = poffset[:,0].reshape(-1,1)
        y = poffset[:,1].reshape(-1,1)
        r2 = r**2
        Js = np.hstack((x/r, -y/r2, y/r, x/r2)).reshape((-1,1,2,2))
        nJs = np.sum(n.reshape(-1,1,2,1) * Js, axis=2)        
        dru = np.concatenate((dr[:,:,np.newaxis], du[:,:,np.newaxis]), axis=2)
        return np.sum(nJs * dru, axis=2)

    def __str__(self):
        return "FHB basis "+ str(self.__orders)

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
        assert len(x)==pw.n
        self.n = 1
        
    def values(self, points):
        return np.dot(self.pw.values(points), self.x).reshape(-1,1)

    def derivs(self, points, n):
        return np.dot(self.pw.derivs(points, n), self.x).reshape(-1,1)


class BasisCombine(object):
    """ Combine several basis objects"""
    def __init__(self, bases):
        self.bases = bases
        self.n = sum([b.n for b in bases])
        
    def values(self, points):
        return np.hstack([b.values(points) for b in self.bases])
        
    def derivs(self, points, n):
        return np.hstack([b.derivs(points, n) for b in self.bases])

    
    def __str__(self):
        return "".join(map(str,self.bases))

