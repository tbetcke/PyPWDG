'''
Created on Aug 31, 2011

@author: joel
'''
import scipy.optimize as so
import scipy.interpolate as si
import numpy as np

def nearestpoint(x, S):
    ''' S is a chart.  x is a point.  returns y, where y minimises |S(y) - x|'''
    x = x.reshape(1,-1)
    return so.fmin_powell(lambda y: np.sum((S(y) - x)**2), np.zeros(x.shape[1]-1), disp=False)

def uniformreferencepoints(dim, n):
    ''' Gives a uniform grid of points on the reference dim-simplex with spacing 1/n'''
    a = np.linspace(0,1,n)
    p = np.array([np.tile(a.reshape(1 - 2*s), n*(1-s)+s) for s in np.eye(dim)]).reshape(dim,-1).transpose() # This produces a grid of points on the unit hypercube of dimension dim (I promise!)
    return p[p.sum(1) <=1] # We only want the points on the unit simplex
    
def appendcolumn(a, v=1):
    ''' Append a uniform column to the matrix a (useful for barycentric coordinates)'''
    return np.hstack((a, np.ones((len(a),1))*v))        

class CurvedMapping:
    ''' Given an m-simplex K, a sub-simplex F (i.e. vertex, edge, face, etc) and a (n-1)-manifold, M, find
        a (smooth) mapping of K that maps the F onto the M, but is the identity
        on any sub-simplices not intersecting F.
        
        The manifold is given by the chart S:U -> M \subset R^n, where U \subset R^{n-1}.        
    
        vertices: determine the simplex, K (an m+1 x n array)
        subidx: the indices of the vertices that determine the sub-simplex
        surface: a callable giving a parameterisation of the surface, S
        interpolationpoints: see below                
        
        TODO: 
        - Allow for other "nearest point" functions.
        - Allow for other interpolation options for p         
    '''
    
    def __init__(self, vertices, subidx, surface, interpolationpoints = 10):
        
        self.dim = len(vertices)-1
        subdim = len(subidx)-1
        self.barycentric = np.linalg.pinv(appendcolumn(vertices)) # matrix that transforms a point into barycentric coordinates 
                                                                  # using .pinv allows us to work with simplices that are not n-dimensional
        
        if subdim >0:
            refpoints = uniformreferencepoints(subdim, interpolationpoints) # Get a grid of points on the reference sub-simplex
                                                                            # Note that these are the first m-1 barycentric coordinates 
            subpoints = np.dot(np.hstack((refpoints, 1-np.sum(refpoints, axis=1).reshape(-1,1))), vertices[subidx, :]) # And map them into meat-space
            
            subparams = np.array(map(nearestpoint, subpoints, [surface]*len(subpoints))) # For each point, x \in F, find the point in y \in U that such that S(y) minimises |S(y) - x|
            # Now construct the mapping, self.f, which maps from points on the reference sub-simplex to M
            if subdim==1:
                p = si.interp1d(refpoints.ravel(), subparams)
                self.f = lambda x : surface(p(x))
            else:
                ps = [si.LinearNDInterpolator(refpoints, params) for params in subparams.transpose()]
                self.f = lambda x : surface(np.hstack([p(x)for p in ps]))
        else:
            y = nearestpoint(vertices[subidx], surface)
            self.f = lambda x : y
        
        self.subidx = subidx
        self.otheridx = np.setdiff1d(np.arange(self.dim+1), self.subidx, True) # the indexes of the non-mapped vertices
        self.othervertices = vertices[self.otheridx]
        
    def apply(self, x):
        barycoords = np.dot(appendcolumn(x), self.barycentric) # convert x into barycentric coordinates
        barycoords[barycoords < 0] = 0
        barycoords[barycoords > 1] = 1 # otherwise the interpolator can get upset - rounding error comes from pinv
        ssbarycoords = barycoords[:,self.subidx] # extract the barycentric coordinates associated with the mapped sub-simplex
        curvemappoints = self.f(ssbarycoords[:,:-1]) # map them (we use the first m-1 barycentric coordinates, because those were the reference points)
        noncurvemappoints = np.dot(barycoords[:,self.otheridx], self.othervertices) # now map the barycentric coordinates not associated with the sub-simplex
        return curvemappoints*np.sum(ssbarycoords,axis=1).reshape(-1,1) + noncurvemappoints # And add everything up
        
def jacobians(f, points, eps=1E-6):
    ''' Numerical approximation of the Jacobian of f at points'''
    n,dim = points.shape    
    h = np.eye(dim)[np.newaxis, ...]*eps
    points = points[:,np.newaxis,:]
    df = f((points + h).reshape(-1,dim)) - f((points - h).reshape(-1,dim))
    jacs = df.reshape(n,dim,-1) / (2*eps)
    return jacs
         
def determinants(jacobians):
    j = jacobians
    n, d1,d2 = j.shape
    if d1==1 and d2==1:
        return jacobians.ravel()
    if d1==2 and d2==2:
        return j[:,0,0] * j[:,1,1] - j[:,0,1]*j[:,1,0]
    else:
        return np.array(map(np.linalg.det, jacobians))
        
