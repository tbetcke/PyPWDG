'''
Utility methods to create Bases and Basis objects

Created on Apr 12, 2011

@author: joel
'''
import math
import numpy as np

import pypwdg.parallel.decorate as ppd
import pypwdg.parallel.distributeddict as ppdd
import pypwdg.mesh.meshutils as pmmu


from pypwdg.parallel.mpiload import mpiloaded

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
    ''' Return an object that populates a plane wave basis'''
    import pypwdg.core.bases.definitions as pcbb
    dirs = uniformdirs(dim, nplanewaves)
    pw = [pcbb.PlaneWaves(dirs,k)]
    return UniformBasisRule(pw)        

class UniformBasisRule(object):
    ''' Creates bases that are uniform across all elements in the mesh'''
    def __init__(self, b):
        self.b = b

    def populate(self, einfo):    
        return self.b

class FourierBesselBasisRule(object):
    ''' Creates bases consisting of a Fourier-Bessel basis on each element'''
    def __init__(self, k, orders, mesh):
        self.orders = orders
    
    def populate(self, einfo):
        import pypwdg.core.bases.definitions as pcbb        
        return [pcbb.FourierBessel(einfo.origin,self.orders,einfo.k)]

class ProductBasisRule(object):
    ''' Creates bases which are the product of two underlying bases on each element'''
    def __init__(self, bases1, bases2):
        self.bases1 = bases1
        self.bases2 = bases2
        
    def populate(self, einfo):        
        import pypwdg.core.bases.definitions as pcbb        
        return [pcbb.Product(pcbb.BasisCombine(self.bases1.populate(einfo)), pcbb.BasisCombine(self.bases2.populate(einfo)))]
   
def getSizes(etob, mesh):
    return np.array([sum([b.n for b in etob.get(e,[])]) for e in range(mesh.nelements)])    

class Info(object):
    ''' Helper class to make the einfo parameters to the basis rules look like structs'''
    def __init__(self, elementid, populator):
        self.elementid = elementid
        self.populator = populator
        
    def __getattr__( self, name ):
        return self.populator.__getattribute__(name)(self.elementid)

class ElementInfo(object):
    ''' General abstraction that provides information about elements used by basis objects.  This is the
        common ground between the Problem class and the above Bases classes.  Can be sub-classed to provide
        more information (see p.c.b.variable)'''
    
    def __init__(self, mesh, k):
        self.mesh = mesh
        self.k = k
        self.mems = pmmu.MeshElementMaps(mesh)

    def k(self, e):
        return self.k
    
    def kp(self, e):
        return lambda p: self.k
        
    def origin(self, e):
        return np.sum(self.mesh.nodes[self.mesh.elements[e]], axis=0) / len(self.mesh.elements[e])
    
    def refmap(self, e):
        return self.mems.getMap(e)
    
    def info(self, e):
        return Info(e, self)
    
@ppd.parallel(None, None)
def localConstructBasis(mesh, etob, basisrule, elementinfo):
    ''' Helper function to initialise the element to basis map in each partition'''  
    for e in mesh.partition:
        etob[e] = basisrule.populate(None if elementinfo is None else elementinfo.info(e))

def constructBasis(mesh, basisrule, elementinfo = None):
    ''' Build an element to basis (distributed) map based on a basisrule'''
    manager = ppdd.ddictmanager(ppdd.elementddictinfo(mesh), True)
    etob = manager.getDict()
    localConstructBasis(mesh, etob, basisrule, elementinfo)
    manager.sync()   
    return ElementToBases(etob, mesh)

class ElementToBases(object):
    ''' Information about, and evaluation of, bases on each element'''
    
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