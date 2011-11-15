'''
Created on Oct 24, 2011

@author: joel
'''
import numpy as np
import scipy.linalg as sl
import scipy.sparse as ss
import math

def norm(a):
    return np.sqrt(np.sum(a**2,axis=1))

class gradient:
    ''' Utility that creates a gradient function from a function using forward differences 
    Args:
        fn: function to take a gradient of
        h:  step-size for forward difference
    '''
    def __init__(self, fn, h):
        self.h = h
        self.fn = fn
    
    def __call__(self,x):
        xh = x + np.identity(x.shape[1])[:,np.newaxis,:]*self.h
        fnx = self.fn(x)
        fnxh = map(lambda xh : self.fn(xh) - fnx, xh)
        return np.array(fnxh).transpose() / self.h

class Recip:
    def __init__(self, f):
        self.f = f
    def __call__(self, p):
        return 1.0/ self.f(p)

def onestep(x, p, slowness, gradslowness, deltat):
    ''' Advance a wavefront by one step.  Currently just uses forward Euler
    Args:
        x:    points along the wavefront
        p:    corresponding phases at the points
        slowness:    slowness function
        gradslowness:    gradient of the slowness
        deltat:    timestep
    Returns:
        (xk,pk): the new positions and phases along the wavefront
    '''
    s = slowness(x)[:,np.newaxis]
    pnorm = norm(p)[:,np.newaxis]
    pp = p * s / pnorm    
    gs = gradslowness(x)        
    xk = x + deltat * pp / s**2
    pk = pp + deltat * gs / s
    
    return xk,pk

def ninterp(x,p,tol):
    """ Calculate the number of interpolation points needed along a wavefront 
    Args: 
        x: points along the wavefront
        p: corresponding phases
        tol: maximum permitted distance between points / phases
    Returns:
        The number of interpolation points for each interval x[i],x[i+1]
    """
    dxt = np.int32(norm(x[1:] - x[:-1]) / tol)
    dpt = np.int32(norm(p[1:] - p[:-1]) / tol)
    return np.max((dxt,dpt),axis=0)
    
def fillin(x,p,tol):
    """ Interpolate a wavefront
    Args: 
        x: points along the wavefront
        p: corresponding phases
        tol: maximum permitted distance between points / phases
    Returns:
        A tuple (xi,pi,forwardidx) where xi and pi are the interpolated points and corresponding
        phases and forwardidx maps the indices of the original points in x to the new array, xi.
        N.B. for efficiency, if no interpolation is performed, forwardidx = None
    """    
    ni = ninterp(x,p,tol)
    cs = np.cumsum(ni)
    nni = cs[-1]
    dim = x.shape[1]
    if nni > 0:
        n = nni + len(x)
        xi = np.empty((n,dim))
        pi = np.empty((n,dim))
        for k, (i, c, xk,xk1,pk,pk1) in enumerate(zip(ni,cs, x[:-1],x[1:],p[:-1],p[1:])):
            w = np.linspace(0,1,i+1,False)[:,np.newaxis]
            
            xi[k+c-i:k+1 + c] = xk * (1-w) + xk1 * w
            pi[k+c-i:k+1 + c] = pk * (1-w) + pk1 * w
        xi[-1] = x[-1]
        pi[-1] = p[-1]
        forwardidx = np.arange(len(x))
        forwardidx[1:]+=cs
        return xi,pi, forwardidx
    else: return x,p,None

def wavefront(x0,p0,slowness,gradslowness,deltat,Tmax,tol):
    """ Compute wavefronts using a Lagrangian method
    Args:
        x0:    Initial points on the wavefront
        p0:    Initial phases
        slowness:    slowness function
        gradslowness:    gradient of the slowness function (if None, forward differences are used)
        deltat:    timestep
        Tmax:    time to integrate until
        tol:    tolerance to fill in points on the wavefront
    Returns:
        (wavefronts, forwardidxs) where wavefronts is a list of tuples (x,p) giving the 
        positions and phases for each wavefront and forwardidxs maps points on one wave front
        to the corresponding point on the next wavefront
    Todo:
        Caclulate tol (and deltat?) automatically.  
    """
    nsteps = int(math.ceil(Tmax / deltat))
    x,p = x0,p0
    wavefronts = []
    forwardidxs = []
    if gradslowness is None:
        gradslowness = gradient(slowness)
    
    s = slowness(x)[:,np.newaxis] # todo: incorporate this into fillin
    pnorm = norm(p)[:,np.newaxis]
    p = p * s / pnorm    

    for _ in range(nsteps):
        xi,pi,fwdidx = fillin(x,p,tol)
        wavefronts.append((xi,pi))
        forwardidxs.append(fwdidx)
        x,p = onestep(xi,pi,slowness,gradslowness,deltat)
    return wavefronts, forwardidxs

class WavefrontQuadsPointTest():
    """ For a set of quadrilaterals along a given wavefront, test whether they contain a set of points"""
    def __init__(self, x0,x1):
        """ x0,x1 are corresponding Lagrangian points on 2 successive wavefronts"""
        x0p = np.hstack((x0, np.ones((len(x0),1)))) 
        x1p = np.hstack((x1, np.ones((len(x1),1))))
        tri1 = np.dstack((x0p[:-1],x0p[1:],x1p[:-1])) # divide each quadrilateral into 2 triangles
        tri2 = np.dstack((x1p[:-1],x1p[1:],x0p[1:])) 
        self.lup1 = map(sl.lu_factor, tri1) # the test will be whether the barycentric coordinates are all in [0,1]
        self.lup2 = map(sl.lu_factor, tri2)
    
    def test(self, p):
        """ Determine which quadrilateral(s) each point is in
        Args:
            p: the points to test
        
        Returns:
            A N x len(p) boolean array, where N is the number of quadrilaterals
        """
        p1 = np.vstack((p.T, np.ones(len(p)))) # set up the points for the barycentric calculation
        bary1 = np.array(map(lambda lup: sl.lu_solve(lup, p1), self.lup1)) # calculate the barycentric coords for the first set of triangles
        bary2 = np.array(map(lambda lup: sl.lu_solve(lup, p1), self.lup2)) # ... and the second
        in1 = np.all((bary1 >=0) * (bary1 <=1), axis=1) # test that they are all in [0,1]
        in2 = np.all((bary2 >=0) * (bary2 <=1), axis=1)
        return in1+in2 # + is "or"

    def unique(self, p):
        """ Give a unique quadrilateral for each point"""
        M = self.test(p)
        (P,Q) = M.T.nonzero()
        ptoq = np.ones(len(p),dtype=np.int32) * -1
        ptoq[P[0]] = Q[0]
        dP = np.diff(P) > 0
        ptoq[P[1:][dP]] = Q[1:][dP]
        return ptoq
    

class WavefrontInterpolate():
    def __init__(self, x0,x1,p0,p1):
        """ x0,x1 are corresponding Lagrangian points on 2 successive wavefronts"""
        x0p = np.hstack((x0, np.ones((len(x0),1)))) 
        x1p = np.hstack((x1, np.ones((len(x1),1))))
        tri1 = np.dstack((x0p[:-1],x0p[1:],x1p[:-1])) # divide each quadrilateral into 2 triangles
        tri2 = np.dstack((x1p[:-1],x1p[1:],x0p[1:])) 
        self.lup1 = map(sl.lu_factor, tri1) # the test will be whether the barycentric coordinates are all in [0,1]
        self.lup2 = map(sl.lu_factor, tri2)
        self.ptri1 = np.dstack((p0[:-1],p0[1:],p1[:-1]))
        self.ptri2 = np.dstack((p1[:-1],p1[1:],p0[1:]))
        self.plen = 1 if len(p0.shape) == 1 else p0.shape[1]

    def interpolate(self, v):
        """ Find all the triangles that each v is in, then average the linear interpolation of the phases"""
        v1 = np.vstack((v.T, np.ones(len(v)))) # set up the points for the barycentric calculation
        bary1 = np.array(map(lambda lup: sl.lu_solve(lup, v1), self.lup1)) # calculate the barycentric coords for the first set of triangles
        bary2 = np.array(map(lambda lup: sl.lu_solve(lup, v1), self.lup2)) # ... and the second
        in1 = np.all((bary1 >=0) * (bary1 <=1), axis=1) # test that they are all in [0,1]
        in2 = np.all((bary2 >=0) * (bary2 <=1), axis=1)
        nTris = np.sum(in1,axis=0) + np.sum(in2, axis=0)
        vfound = nTris > 0
        phases = np.zeros((len(v), self.plen))
        for tidx,vidx in zip(*in1.nonzero()):
            phases[vidx] += np.dot(self.ptri1[tidx], bary1[tidx, :, vidx]) 
        for tidx,vidx in zip(*in2.nonzero()):
            phases[vidx] += np.dot(self.ptri2[tidx], bary2[tidx, :, vidx]) 
        return vfound, phases[vfound] / nTris[vfound].reshape(-1,1)        


def nodesToPhases(wavefronts, forwardidxs, mesh, bdys):
    nodephases = [[] for _ in range(mesh.nnodes)]
    bdylist = sum([list(nodes) for (i, nodes) in mesh.boundaries if i in bdys], [])    
    ftov = ss.csr_matrix((np.ones(mesh.nfaces * 2, dtype=np.int8), mesh.faces.ravel(), np.arange(0, mesh.nfaces+1)*2), dtype=np.int8)
    vtov = ftov.T * ftov
    vtov.data = vtov.data / vtov.data
 
    bdynodes = np.zeros(mesh.nnodes, dtype=bool)
    bdynodes[bdylist] = True   
    nextnodes = np.zeros(mesh.nnodes, dtype=bool)
       
    for ((x0,p0),(x1,p1),idxs) in zip(wavefronts[:-1], wavefronts[1:], forwardidxs[1:]):
        # For each wavefront, we'll search for nodes that are within it.
        # We start with curnodes.  
        curnodes = nextnodes if nextnodes.any() else bdynodes # we start with the neighbours of the previous wavefront (or the boundary)
        nextnodes = np.zeros(mesh.nnodes, dtype=bool)
        
        checkednodes = np.zeros(mesh.nnodes, dtype=bool) 
        (x1p, p1p) = (x1,p1) if idxs is None else (x1[idxs], p1[idxs]) 
        wi = WavefrontInterpolate(x0,x1p,p0,p1p)
        while curnodes.any():
            cnz = curnodes.nonzero()[0]
            found, phases = wi.interpolate(mesh.nodes[cnz])
            nextnodes[cnz[np.logical_not(found)]] = True
            if len(phases): # we found something
                for p, idx in zip(phases, cnz[found]):
                    nodephases[idx].append(p)
                checkednodes[cnz] = True        
                curnodes = np.logical_and((vtov * curnodes), np.logical_not(checkednodes))
            else:
                break # we didn't find anything.  Stop working with this wavefront
        
    return nodephases
        
        
        
        
        
        
    