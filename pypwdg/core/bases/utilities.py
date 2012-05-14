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
    
    if dim==1:
        if npw==1:
            return np.array([[1],[-1]],dtype=np.float64)
        else:
            return np.array([[1]],dtype=np.float64)
    elif dim==2:
        return circleDirections(npw)
    elif dim==3:
        return cubeRotations(cubeDirections(npw))
    
def planeWaveBases(dim, k, nplanewaves=10):
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
    def __init__(self, orders):
        self.orders = orders
    
    def populate(self, einfo):
        import pypwdg.core.bases.definitions as pcbb        
        return [pcbb.FourierBessel(einfo.origin,self.orders,einfo.k)]
    
class FourierHankelBasisRule(object):
    ''' Creates bases consisting of Hankel fct. on each element'''
    def __init__(self,origins,orders):
        self.orders=orders
        self.origins=origins
        
    def populate(self, einfo):
        import pypwdg.core.bases.definitions as pcbb
        return ([ pcbb.FourierHankel(o,self.orders,einfo.k) for o in self.origins])

class PlaneWaveFromDirectionsRule(object):
    """Takes a list of directions for each element and returns a corresponding
       Plane Wave basis.
       
       If there is no direction for an element a Fourier-Bessel fct. of degree
       zero is used instead.
    """
    
    def __init__(self, etods):
        self.etods=etods
        
    def populate(self, einfo):
        import pypwdg.core.bases.definitions as pcbb
        id=einfo.info.elementid
        if len(self.etods[id])==0: 
            return [pcbb.FourierBessel(einfo.origin,0,einfo.k)]
        else:
            return [pcbb.PlaneWaves(self.etods[id],einfo.k)]
        
class UnionBasisRule(object):
    """Return the union of two basis rules"""
    
    def __init__(self,rules):
        self.rules=rules
        
    def populate(self, einfo):
        import pypwdg.core.bases.definitions as pcbb
        bs = []
        for rule in self.rules:
            for b in rule.populate(einfo):
                bs.append(b)
        return [pcbb.BasisCombine(bs)]
    
class GeomIdBasisRule(object):
    '''Takes a dictionary {Id:BasisRule} to define different basis rules for different geometric identities'''
    
    def __init__(self,basisDict):
        self.basisDict=basisDict
        
    def populate(self,einfo):
        return self.basisDict[einfo.geomId].populate(einfo)
        

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

class SingleElementInfo(object):
    ''' Helper class to make the einfo parameters to the basis rules look like structs'''
    def __init__(self, elementid, elementinfo):
        self.elementid = elementid
        self.elementinfo = elementinfo
        
    def __getattr__( self, name ):
        return self.elementinfo.__getattribute__(name)(self.elementid)

class ElementInfo(object):
    ''' General abstraction that provides information about elements used by basis objects.  This is the
        common ground between the Problem class and the above Bases classes.  Can be sub-classed to provide
        more information (see p.c.b.variable)'''
    
    def __init__(self, mesh):
        self.mesh = mesh
        self.mems = pmmu.MeshElementMaps(mesh)
            
    def origin(self, e):
        return np.sum(self.mesh.nodes[self.mesh.elements[e]], axis=0) / len(self.mesh.elements[e])
    
    def refmap(self, e):
        return self.mems.getMap(e)
    
    def info(self, e):
        return SingleElementInfo(e, self)
    
    def geomId(self,e):
        return self.mesh.elemIdentity[e]
    
    def volume(self, e):
        def vq(quadrule):
            meqs = pmmu.MeshElementQuadratures(self.mesh, quadrule) 
            return meqs.quadpoints(e), meqs.quadweights(e)
        return vq
    
    def boundary(self, e):
        def bq(quadrule):
            mfqs = pmmu.MeshQuadratures(self.mesh, quadrule) 
            fs = self.mesh.etof[e]
            qp = np.vstack((mfqs.quadpoints(f) for f in fs))
            fqw = [mfqs.quadweights(f) for f in fs]
            qw = np.concatenate(fqw)
            ns = self.mesh.normals[fs].repeat(map(len, fqw), axis=0)
            return qp,qw,ns
        return bq
    
    def vertices(self, e):
        return self.mesh.elements[e]
            
class KElementInfo(ElementInfo):
    def __init__(self, mesh, k):
        ElementInfo.__init__(self, mesh)
        self.kk = k

    def k(self, e):
        return self.kk    
    
    def kp(self, e):
        return lambda p: self.kk
            
#    
#@ppd.parallel(None, None)
#def localConstructBasis(mesh, etob, basisrule, elementinfo):
#    ''' Helper function to initialise the element to basis map in each partition'''  
#    for e in mesh.partition:
#        etob[e] = basisrule.populate(None if elementinfo is None else elementinfo.info(e))
#
#def constructBasis(mesh, basisrule, elementinfo = None):
#    ''' Build an element to basis (distributed) map based on a basisrule'''
#    manager = ppdd.ddictmanager(ppdd.elementddictinfo(mesh), True)
#    etob = manager.getDict()
#    localConstructBasis(mesh, etob, basisrule, elementinfo)
#    manager.sync()   
#    return CellToBases(etob, range(mesh.nelements))

class CellToBases(object):
    ''' Information about, and evaluation of, bases on a cell (i.e. an element or a face)'''
    
    def __init__(self, ctob, cids):
        self.ctob = ctob
        self.sizes = np.array([sum([b.n for b in ctob.get(e,[])]) for e in cids])     
        self.indices = np.cumsum(np.concatenate(([0], self.sizes))) 
    
    def getValues(self, cid, points):
        """ Return the values of the basis for element cid at points"""
        bases = self.ctob.get(cid)
        if bases==None:
            return np.zeros((len(points),0))
        else:
            return np.hstack([b.values(points) for b in bases])
    
    def getDerivs(self, cid, points, normal = None):
        """ Return the directional derivatives of the basis for element cid at points
        
           if normal == None, returns the gradient on the standard cartesian grid
        """
        bases = self.ctob.get(cid)
        if bases==None:
            return np.zeros((len(points),0)) if normal is not None else np.zeros((len(points), 0, points.shape[1]))
        else:
            return np.hstack([b.derivs(points, normal) for b in bases])

    def getLaplacian(self, cid, points):
        """ Return the laplacian of the basis for element cid at points"""
        bases = self.ctob.get(cid)
        if bases==None:
            return np.zeros((len(points),0))
        else:
            return np.hstack([b.laplacian(points) for b in bases])

    
    def getSizes(self):
        return self.sizes
        
    def getIndices(self):
        return self.indices 
    
class UniformElementToBases(object):
    def __init__(self, b, mesh):
        self.b = b
        self.sizes = np.ones(mesh.nelements, dtype=int) * b.n
        self.indices = np.arange(mesh.nelements) * b.n
    
    def getValues(self, eid, points):
        return self.b.values(points)
    
    def getDerivs(self, eid, points, normal = None):
        return self.b.derivs(points, normal)
    
    def getSizes(self):
        return self.sizes
        
    def getIndices(self):
        return self.indices 
    

class UniformFaceToBases(object):
    def __init__(self, b, mesh, entityton = None):
        self.mesh = mesh
        self.b = b
        self.numbases = np.ones(mesh.nfaces, dtype=int) * b.n
        self.indices = np.arange(mesh.nelements) * b.n
        
    def evaluate(self, faceid, points):
        normal = self.mesh.normals[faceid]
        vals = self.b.values(points)
        derivs = self.b.derivs(points, normal)
        return (vals, derivs)

class FaceToBasis(object):
    def __init__(self, mesh, elttobasis):
        self.mesh = mesh
        self.elttobasis = elttobasis
        self.numbases = elttobasis.getSizes()[mesh.ftoe]
        self.indices = elttobasis.getIndices()[mesh.ftoe]
    
    def evaluate(self, faceid, points):
        e = self.mesh.ftoe[faceid] 
        normal = self.mesh.normals[faceid]
        vals = self.elttobasis.getValues(e, points)
        derivs = self.elttobasis.getDerivs(e, points, normal)
        return (vals,derivs)

class FaceToScaledBasis(FaceToBasis):
    def __init__(self, entityton, *args, **kwargs):
        super(FaceToScaledBasis, self).__init__(*args, **kwargs)
        self.entityton = entityton

    def evaluate(self,faceid, points):
        e = self.mesh.ftoe[faceid] 
        (vals, derivs) = super(FaceToScaledBasis, self).evaluate(faceid, points)
        return (vals * self.entityton[e](points).reshape(-1,1), derivs)    
    